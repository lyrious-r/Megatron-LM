import argparse
import json
import os
import sys
import math
import time
import shutil
import subprocess
from dataclasses import dataclass
from string import Template
from typing import Optional
import pickle
import datetime
import redis

EXP_CONFIG_DIR = "./experiment_configs"
TEMPLATE_PATH = os.path.join(EXP_CONFIG_DIR, "finetune_{}.template")
DEEPSPEED_TEMPLATE_PATH = os.path.join(
    EXP_CONFIG_DIR, "deepspeed_config.template"
)
EXPERIMENT_DIR_PREFIX = "./experiments/"
EXPERIMENT_PROGRESS_TIMEOUT = 180  # 3 mins
EXPERIMENT_PROGRESS_POLL_INTERVAL = 5  # 5s

EXP_REDIS_PORT = 9876
KVREDIS_INIT_POLLING_INTERVAL = 0.5
KVREDIS_CONNECT_TIMEOUT = 30
KVREDIS_POLLING_INTERVAL = 0.5

print_fn = print

class RedisKVStore(object):
    # a blocking local redis client
    def __init__(self, args):
        self.node_rank = args.node_rank
        self.is_master = args.node_rank == 0
        self.host = args.master_addr
        self.port = EXP_REDIS_PORT
        self.n_processes = args.nnodes
        self.barrier_cnt = 0
        if self.is_master:
            self.server = self._run_redis_server()
        # wait for redis server to start
        t = time.time()
        while True:
            try:
                self.client = redis.Redis(host=self.host, port=self.port, db=0)
                self.client.ping()
                break
            except redis.exceptions.ConnectionError:
                time.sleep(KVREDIS_INIT_POLLING_INTERVAL)
                if time.time() - t > KVREDIS_CONNECT_TIMEOUT:
                    raise RuntimeError(
                        "WARNING: Cannot connect to KV Server. "
                        "Is PLOPT_KV_HOST and PLOPT_KV_PORT set correctly?"
                    )
                continue

    def __del__(self):
        if self.is_master:
            if self.server.poll() is not None:
                return
            self.server.send_signal(subprocess.signal.SIGINT)
            self.server.wait()

    def _run_redis_server(self):
        # run a redis server
        p = subprocess.Popen(
            [
                "redis-server",
                "--save",
                "",
                "--port",
                str(self.port),
                "--bind",
                str(self.host),
            ],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        return p

    def wait(self, keys, timeout=None):
        # wait for a key to be set
        time_start = datetime.datetime.now()
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        while True:
            if self.client.exists(*keys):
                break
            if (
                timeout is not None
                and datetime.datetime.now() - time_start > timeout
            ):
                # match torch kvstore behavior
                raise RuntimeError("Timeout")
            time.sleep(KVREDIS_POLLING_INTERVAL)

    def barrier(self):
        key = "barrier_{}".format(self.barrier_cnt)
        self.client.incr(key)
        while True:
            count = self.client.get(key)
            if count == self.n_processes:
                break
            time.sleep(KVREDIS_POLLING_INTERVAL)
        if self.node_rank == 0:
            self.client.delete(key)
        self.barrier_cnt += 1

    def blocking_get(self, key):
        self.wait(key)
        return self.client.get(key)

    def set(self, key, value):
        # match torch kvstore behavior
        self.client.set(key, value)

    def get(self, key):
        return self.client.get(key)

    def add(self, key, value: int):
        # match torch kvstore behavior
        return self.client.incr(key, value)

    def delete_key(self, key):
        return self.client.delete(key)

    def gather(self, obj):
        # synchronous gather
        ack_key = "gather_ack"
        if self.node_rank == 0:
            recved_objs = [obj]
            # read from all keys
            for i in range(1, self.n_processes):
                key = "gather_r{}".format(i)
                self.wait(key)
                recved_objs.append(pickle.loads(self.client.get(key)))
                self.delete_key(key)
            # set ack key
            self.set(ack_key, "1")
            return recved_objs
        else:
            # delete ack key
            self.delete_key(ack_key)
            key = "gather_r{}".format(self.node_rank)
            self.client.set(key, pickle.dumps(obj))
            # wait for ack key before returning
            self.wait(ack_key)
        return

    def send_abort_signal(self):
        self.client.set("abort", 1)

    def check_abort_signal(self):
        if self.client.get("abort") == 1:
            return True
        return False

def _postprocess_group_args(
    args: argparse.Namespace,
    group: argparse._ArgumentGroup,
    config_attr: Optional[str] = None,
    optional_args: Optional[list] = None,
    switch_arg: Optional[str] = None,
):
    if optional_args is None:
        optional_args = ["help", config_attr]

    required_args = [
        act.dest
        for act in group._group_actions
        if act.dest not in optional_args
    ]
    required_args_index = [
        i
        for i, act in enumerate(group._group_actions)
        if act.dest not in optional_args
    ]
    optional_args = [
        act.dest for act in group._group_actions if act.dest in optional_args
    ]
    optional_args_index = [
        i
        for i, act in enumerate(group._group_actions)
        if act.dest in optional_args
    ]
    if config_attr and getattr(args, config_attr) is not None:
        # read config file and set args
        with open(getattr(args, config_attr), "r") as f:
            config: dict = json.load(f)
        for k, v in config.items():
            if k in required_args:
                action_index = required_args.index(k)
                action = group._group_actions[
                    required_args_index[action_index]
                ]
            elif k in optional_args:
                action_index = optional_args.index(k)
                action = group._group_actions[
                    optional_args_index[action_index]
                ]
            else:
                continue
            if hasattr(action, "type") and action.type is not None:
                try:
                    v = action.type(v)
                except ValueError:
                    raise ValueError(
                        f"Invalid value {v} for argument {k}, "
                        f"expected type {action.type}"
                    )
            setattr(args, k, v)
    # only check required args if switch_arg is not set or
    # switch_arg is set and is True
    if switch_arg is not None:
        # check if switch_arg is a valid arg
        if getattr(args, switch_arg) is None:
            raise ValueError(f"Switch argument {switch_arg} is not set.")
    if switch_arg is None or getattr(args, switch_arg):
        for arg in required_args:
            if getattr(args, arg) is None:
                raise ValueError(f"Argument {arg} is required.")


def _add_cluster_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Cluster Config")
    group.add_argument("--node_rank", type=int, default=0, help="Node rank.")
    group.add_argument(
        "--cluster_config",
        type=str,
        help="Load cluster spec from config file.",
    )
    # if cluster_config is not specified, require the following args
    group.add_argument(
        "--master_addr", type=str, default="localhost", help="Master address."
    )
    group.add_argument(
        "--master_port", type=str, default="18230", help="Master port."
    )
    group.add_argument(
        "--nnodes", type=int, default=1, help="Number of nodes."
    )
    group.add_argument(
        "--gpus_per_node", type=int, default=4, help="Number of GPUs per node."
    )
    group.add_argument(
        "--plopt_kv_host", type=str, default="localhost", help="KV store host."
    )
    group.add_argument(
        "--plopt_kv_port", type=int, default=6379, help="KV store port."
    )
    return parser, group


def _check_cluster_args(args):
    if args.nnodes > 1:
        assert (
            args.master_addr != "localhost"
        ), "master_addr must be specified for multi-node training."
        assert (
            args.node_rank < args.nnodes and args.node_rank >= 0
        ), "node_rank must be in [0, nnodes)."
    return args


def _add_model_args(parser):
    group = parser.add_argument_group(title="Model Config")
    group.add_argument(
        "--model_config", type=str, help="Load model spec from config file."
    )
    # if model_config is not specified, require the following args
    group.add_argument(
        "--num_layers", type=int, help="Number of layers in the model."
    )
    group.add_argument(
        "--encoder_num_layers", type=int, help="Number of encoder layers."
    )
    group.add_argument(
        "--decoder_num_layers", type=int, help="Number of decoder layers."
    )
    group.add_argument("--hidden_size", type=int, help="Model hidden size.")
    group.add_argument(
        "--num_attn_heads", type=int, help="Number of attention heads."
    )
    group.add_argument(
        "--kv_channels", type=int, help="Number of KV channels."
    )
    group.add_argument("--ffn_hidden_size", type=int, help="FFN hidden size.")
    return parser, group


def _add_data_args(parser):
    group = parser.add_argument_group(title="Data Config")
    group.add_argument(
        "--data_config", type=str, help="Load data spec from config file."
    )
    # if data_config is not specified, require the following args
    group.add_argument("--data_path", type=str, help="Path to dataset.")
    group.add_argument(
        "--targets_data_path", type=str, help="Path to target dataset."
    )
    group.add_argument("--vocab_file", type=str, help="Path to vocab file.")
    group.add_argument("--merge_file", type=str, help="Path to merge file.")
    return parser, group


def _add_training_args(parser):
    group = parser.add_argument_group(title="Training Config")
    group.add_argument(
        "--training_config",
        type=str,
        help="Load training spec from config file.",
    )
    group.add_argument("--seq_length", type=int, help="Max sequence length.")
    group.add_argument(
        "--encoder_seq_length", type=int, help="Max encoder sequence length."
    )
    group.add_argument(
        "--decoder_seq_length", type=int, help="Max decoder sequence length."
    )
    group.add_argument(
        "--tokens_per_global_batch",
        type=int,
        help="Tokens per global batch (used when enabling plopt).",
    )
    # if training_config is not specified, require the following args
    group.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    group.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Pipeline parallel size.",
    )
    group.add_argument(
        "--pp_split_rank",
        type=int,
        default=0,
        help="Rank where encoder and decoder is split",
    )
    group.add_argument(
        "--micro_batch_size",
        type=int,
        default=8,
        help="Micro-batch size (not used when enabling plopt).",
    )
    group.add_argument(
        "--global_batch_size",
        type=int,
        default=32,
        help="Global batch size (not used when enabling plopt).",
    )
    group.add_argument(
        "--max_pos_embeddings",
        type=int,
        default=1024,
        help="Max position embeddings "
        "(filled automatically using max seq len).",
    )
    group.add_argument(
        "--train_iters",
        type=int,
        default=1000,
        help="Number of training iterations.",
    )
    group.add_argument(
        "--recompute_level",
        type=str,
        choices=["none", "selective", "full"],
        default="none",
        help="Enable static recompute (not used when enabling plopt).",
    )
    group.add_argument(
        "--enable_deepspeed",
        type=bool,
        default=False,
        help="Run model using deepspeed.",
    )
    group.add_argument(
        "--deepspeed_zero_stage",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Which stage of ZeRO to use.",
    )
    return parser, group


def _get_expected_gbs(args):
    # check micro_batch_size and global_batch_size
    if args.model_type == "t5":
        effective_tokens_per_sequence = (
            args.encoder_seq_length + args.decoder_seq_length
        )
    else:
        effective_tokens_per_sequence = args.seq_length
    expected_global_batch_size = (
        args.tokens_per_global_batch // effective_tokens_per_sequence
    )
    return expected_global_batch_size


def _check_training_args(args):
    if args.enable_plopt:
        # reset recompute level
        args.recompute_level = "none"
    if args.enable_deepspeed:
        if args.deepspeed_zero_stage is None:
            raise ValueError(
                "Argument deepspeed_zero_stage is required "
                "when enable_deepspeed is True."
            )
    # automatically set max_pos_embeddings
    if args.model_type == "gpt":
        args.max_pos_embeddings = args.seq_length
    else:
        args.max_pos_embeddings = max(
            args.encoder_seq_length, args.decoder_seq_length
        )
    if args.enable_plopt:
        # micro batch size/global batch size is not used
        # make it 1 so it does not interfere with the check in Megatron-LM
        args.micro_batch_size = 1
    else:
        # check micro_batch_size and global_batch_size
        expected_global_batch_size = _get_expected_gbs(args)
        if args.global_batch_size != expected_global_batch_size:
            print_fn("Override global batch size to {}".format(expected_global_batch_size))
            args.global_batch_size = expected_global_batch_size
    return args


def _add_plopt_args(parser):
    group = parser.add_argument_group(title="PLOPT Config")
    # training_config is used for plopt
    group.add_argument(
        "--plopt_dump_stats",
        type=bool,
        help="Dump memory, ep, and execution logs to file.",
    )
    group.add_argument(
        "--enable_plopt", type=bool, default=False, help="Enable PLOPT."
    )
    group.add_argument(
        "--plopt_cost_model",
        type=str,
        default="default",
        help="Path to the cost model.",
    )
    group.add_argument(
        "--plopt_device_to_node",
        type=str,
        help="Mapping from rank to node, e.g. 0:0,1:0,2:1,3:1",
    )
    group.add_argument(
        "--plopt_device_memory_limit",
        type=int,
        help="Memory limit per device in MB.",
    )
    group.add_argument(
        "--plopt_intra_node_bw",
        type=int,
        default=4800,
        help="Intra-node bandwidth in gbps.",
    )
    group.add_argument(
        "--plopt_inter_node_bw",
        type=int,
        default=100,
        help="Inter-node bandwidth in gbps.",
    )
    group.add_argument(
        "--plopt_layer_to_device",
        type=str,
        help="A list of device ids for each layer, e.g. 0,1,2,3,0,1,2,3",
    )
    group.add_argument(
        "--plopt_prefetch_planner_num_workers",
        type=int,
        default=64,
        help="Number of workers for preprocessing (per node).",
    )
    group.add_argument(
        "--plopt_debug_level", type=str, default="INFO", help="Debug level."
    )
    group.add_argument(
        "--plopt_debug_logging_dir", type=str, help="Debug logging dir."
    )
    group.add_argument(
        "--plopt_debug_dump_ep_prefix",
        type=str,
        help="Directory to dump ep stats.",
    )
    group.add_argument(
        "--plopt_debug_dump_memory_prefix",
        type=str,
        help="Directory to dump memory stats.",
    )
    group.add_argument(
        "--plopt_enable_packing",
        type=bool,
        default=False,
        help="Enable packing.",
    )
    return parser, group


def _add_experiment_args(parser):
    group = parser.add_argument_group(title="Experiment Config")
    group.add_argument(
        "--batch_experiments",
        type=bool,
        default=False,
        help="Run experiment items in batch.",
    )
    group.add_argument(
        "--sequence_length_range",
        type=str,
        default="512,1024,2048,4096,8192,16384,32768",
        help="Sequence length range.",
    )
    group.add_argument(
        "--global_batch_size_range",
        type=str,
        default="16384,32768,65536,131072",
        help="Global batch size range.",
    )
    return parser, group


def _check_logging_args(args):
    nranks = args.nnodes * args.gpus_per_node
    pp_times_tp = args.tensor_parallel_size * args.pipeline_parallel_size
    dp_size = nranks // pp_times_tp
    exp_spec_name = "dp{}_tp{}_pp{}".format(
        dp_size, args.tensor_parallel_size, args.pipeline_parallel_size
    )
    if args.model_type == "gpt":
        exp_spec_name += "_sl{}_gbs{}".format(
            args.seq_length, args.tokens_per_global_batch
        )
    else:
        exp_spec_name += "_encsl{}_decsl{}_gbs{}".format(
            args.encoder_seq_length,
            args.decoder_seq_length,
            args.tokens_per_global_batch,
        )
    if not args.enable_plopt:
        exp_spec_name += "_mbs{}_rc{}".format(
            args.micro_batch_size, args.recompute_level
        )
    if args.enable_deepspeed:
        exp_spec_name += "_zero{}".format(args.deepspeed_zero_stage)
    if args.enable_plopt and args.plopt_enable_packing:
        exp_spec_name += "_spp" # for seqlen preserving packing
    exp_logging_dir = os.path.join(
        EXPERIMENT_DIR_PREFIX,
        args.experiment_name,
        exp_spec_name,
    )
    if os.path.exists(exp_logging_dir):
        # exp dir already exists
        return args, exp_logging_dir, True
    if not os.path.exists(exp_logging_dir):
        os.makedirs(exp_logging_dir)
    if args.enable_plopt:
        if args.plopt_dump_stats:
            args.plopt_debug_level = "DEBUG"
        else:
            args.plopt_debug_level = "INFO"
        args.plopt_debug_logging_dir = os.path.join(
            exp_logging_dir, "plopt_logs"
        )
        args.plopt_debug_dump_ep_prefix = os.path.join(
            exp_logging_dir, "plopt_ep_stats"
        )
        args.plopt_debug_dump_memory_prefix = os.path.join(
            exp_logging_dir, "plopt_memory_stats"
        )
    else:
        args.plopt_debug_logging_dir = "UNUSED"
        args.plopt_debug_dump_ep_prefix = "UNUSED"
        args.plopt_debug_dump_memory_prefix = "UNUSED"
    args.stdout_stderr_log = os.path.join(exp_logging_dir, "stdout_stderr.log")
    # dump all args to a file
    args_file = os.path.join(exp_logging_dir, "args.json")
    with open(args_file, "w") as f:
        args_dict = vars(args)
        dump_dict = {k: v for k, v in args_dict.items() if k != "kvstore"}
        json.dump(dump_dict, f, indent=2)
    return args, exp_logging_dir, False


def _create_deepspeed_config(args, exp_logging_dir):
    if args.enable_deepspeed:
        # generate deepspeed config
        nranks = args.nnodes * args.gpus_per_node
        pp_times_tp = args.tensor_parallel_size * args.pipeline_parallel_size
        assert nranks % pp_times_tp == 0, (
            f"Number of ranks {nranks} must be divisible by "
            f"pipeline_parallel_size {args.pipeline_parallel_size} "
            f"times tensor_parallel_size {args.tensor_parallel_size}"
        )
        dp_size = nranks // pp_times_tp
        per_gpu_batch_size = args.global_batch_size // dp_size
        assert per_gpu_batch_size % args.micro_batch_size == 0, (
            f"Per GPU batch size {per_gpu_batch_size} must be divisible by "
            f"micro batch size {args.micro_batch_size}"
        )
        n_micro_batches = per_gpu_batch_size // args.micro_batch_size
        # if running with pipeline, assert that we are using zero 1
        if args.pipeline_parallel_size > 1 and args.deepspeed_zero_stage >= 2:
            raise ValueError(
                "ZeRO2 and ZeRO3 are not supported with pipeline parallelism."
            )
        # disable overlap comm if using pipeline parallelism
        if args.pipeline_parallel_size > 1 or args.enable_plopt:
            overlap_comm = "false"
        else:
            overlap_comm = "true"
        # create deepspeed config
        with open(DEEPSPEED_TEMPLATE_PATH, "r") as f:
            template = Template(f.read())
        config_str = template.substitute(
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=n_micro_batches,
            zero_stage=args.deepspeed_zero_stage,
            overlap_comm=overlap_comm,
        )
        deepspeed_config_path = os.path.join(
            exp_logging_dir, "deepspeed_config.json"
        )
        with open(deepspeed_config_path, "w") as f:
            f.write(config_str)
        args.deepspeed_config = deepspeed_config_path
    else:
        args.deepspeed_config = None
    return args


def _get_pow_of_2s_up_to(n):
    """Get powers of 2 up to n.

    Args:
        n (int): Upper bound.

    Returns:
        list: List of powers of 2.
    """
    # to reduce the number of configs, we manually specify candidates
    if n <= 4:
        return [2**i for i in range(math.floor(math.log2(n)) + 1)]
    elif n == 8:
        return [1, 4, 8]
    elif n == 16:
        return [1, 8, 16]
    elif n == 32:
        return [1, 16, 32]
    elif n == 64:
        return [1, 16, 32, 64]
    elif n == 128:
        return [1, 32, 64, 128]
    elif n == 256:
        return [1, 64, 128, 256]
    else:
        raise ValueError(f"Invalid n: {n}")


def grid_search_parallelism(args):
    assert (args.gpus_per_node != 0) and (
        args.gpus_per_node & (args.gpus_per_node - 1) == 0
    ), "Number of GPUs per node must be a power of 2."
    # only allow intra-node tp
    for tp in _get_pow_of_2s_up_to(args.gpus_per_node):
        gpus_per_tp_group = args.gpus_per_node * args.nnodes // tp
        for pp in _get_pow_of_2s_up_to(gpus_per_tp_group):
            if args.num_layers and args.num_layers % pp != 0:
                continue
            if args.encoder_num_layers and args.encoder_num_layers % pp != 0:
                continue
            dp = gpus_per_tp_group // pp
            assert (
                dp * tp * pp == args.gpus_per_node * args.nnodes
            ), "Invalid parallelism configuration."
            yield (dp, tp, pp)


def grid_search_ds_stage(args):
    if args.model_type == "t5" and args.pipeline_parallel_size > 1:
        # since we use interleaved schedule for t5, we cannot use DS
        yield (False, 0)
        return
    if args.pipeline_parallel_size > 1 or args.enable_plopt:
        # we can only use zero 1 with pipeline parallelism or with plopt
        stage_candidates = [0, 1]
    else:
        # we can use zero 1, 2 with data and tensor parallelism
        stage_candidates = [0, 1, 2]
    for ds_stage in stage_candidates:
        enable_ds = ds_stage > 0
        yield (enable_ds, ds_stage)


def grid_search_microbatch_size(dp_size, args):
    # this should be run after setting sequence length and global batch size
    if args.enable_plopt:
        # plopt use dynamic micro batch size
        yield 1
        return
    expected_gbs = _get_expected_gbs(args)
    per_gpu_batch_size = expected_gbs // dp_size
    if per_gpu_batch_size == 0:
        # cannot run
        return
    for mbs in reversed(_get_pow_of_2s_up_to(per_gpu_batch_size)):
        if expected_gbs % mbs == 0:
            yield mbs


def grid_search_recomputation(args):
    if args.enable_plopt:
        # plopt use dynamic recomputation
        yield "none"
        return
    for recompute_level in ["none", "selective", "full"]:
        yield recompute_level


def get_pp_split_rank(pp_size):
    return pp_size // 2


def get_pp_device_to_node_str(args):
    rank_separation = (
        args.nnodes * args.gpus_per_node // args.pipeline_parallel_size
    )
    mappings = []
    for pp_rank_id in range(args.pipeline_parallel_size):
        gpu_id = pp_rank_id * rank_separation
        node_id = gpu_id // args.gpus_per_node
        mappings.append("{}:{}".format(pp_rank_id, node_id))
    return ",".join(mappings)


def get_layer_to_device(args):
    if args.pipeline_parallel_size == 1:
        return ",".join(
            ["0"]
            * (
                args.encoder_num_layers + args.decoder_num_layers
                if args.model_type == "t5"
                else args.num_layers
            )
        )
    pp_size = args.pipeline_parallel_size
    if args.model_type == "t5":
        # we use interleaved schedule for t5 to balance encoder and decoders
        # for simplicity here we use a fixed virtual pipeline parallel size 2
        assert (
            args.encoder_num_layers % pp_size == 0
        ), "Number of layers must be divisible by pipeline parallel size."
        n_encoder_layers_per_device = args.encoder_num_layers // pp_size
        n_decoder_layers_per_device = args.decoder_num_layers // pp_size
        layer_to_device = []
        for i in range(pp_size):
            for _ in range(n_encoder_layers_per_device):
                layer_to_device.append(f"{i}")
        for i in range(pp_size):
            for _ in range(n_decoder_layers_per_device):
                layer_to_device.append(f"{i}")
    else:
        # we use sequential schedule for gpt
        assert (
            args.num_layers % pp_size == 0
        ), "Number of layers must be divisible by pipeline parallel size."
        n_layers_per_device = args.num_layers // pp_size
        layer_to_device = []
        for i in range(pp_size):
            for _ in range(n_layers_per_device):
                layer_to_device.append(f"{i}")
    return ",".join(layer_to_device)


RC_MAP = {
    "none": 0,
    "selective": 1,
    "full": 2,
}


@dataclass(eq=True)
class ExperimentConfig:
    enc_seqlen: int = 0
    dec_seqlen: int = 0
    gbs: int = 0
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    mbs: int = 1
    rc: str = "none"
    ds_level: int = 0
    spp: bool = False
    status: str = "unknown"

    def speed_dominates(self, other):
        assert isinstance(
            other, ExperimentConfig
        ), "Can only compare with ExperimentConfig"
        # Config A speed_dominates config B if A is almost surely faster than B
        # if sequence length, gbs or parallelism are different, no dominance
        if (
            self.enc_seqlen != other.enc_seqlen
            or self.dec_seqlen != other.dec_seqlen
            or self.gbs != other.gbs
            or self.dp_size != other.dp_size
            or self.tp_size != other.tp_size
            or self.pp_size != other.pp_size
            or self.spp != other.spp
        ):
            return False
        # dominance happens if ds_level is lower, and mbs is higher,
        # and rc level is lower
        if (
            self.ds_level <= other.ds_level
            and self.mbs >= other.mbs
            and RC_MAP[self.rc] <= RC_MAP[other.rc]
        ):
            return True
        return False

    def memory_dominates(self, other):
        assert isinstance(
            other, ExperimentConfig
        ), "Can only compare with ExperimentConfig"
        # Config A memory_dominates config B if A is almost surely
        # consumes more memory than B
        # if gbs or parallelism are different, no dominance
        if (
            self.gbs != other.gbs
            or self.dp_size != other.dp_size
            or self.tp_size != other.tp_size
            or self.pp_size != other.pp_size
            or self.spp != other.spp
        ):
            return False
        # dominance happens if sequence length is higher, mbs is higher,
        # rc level is lower, and ds_level is lower
        if (
            self.enc_seqlen >= other.enc_seqlen
            and self.dec_seqlen >= other.dec_seqlen
            and self.mbs >= other.mbs
            and RC_MAP[self.rc] <= RC_MAP[other.rc]
            and self.ds_level <= other.ds_level
        ):
            return True
        return False

    @staticmethod
    def parse_experiment_status(exp_dir):
        log_path = os.path.join(exp_dir, "stdout_stderr.log")
        if not os.path.exists(log_path):
            return "unknown"
        with open(log_path, "r") as f:
            contents = f.read()
        if "[after training is done]" in contents:
            return "success"
        else:
            return "failure"

    @staticmethod
    def parse_history_experiments(exp_dir):
        exp_spec = os.path.basename(os.path.normpath(exp_dir))
        config_items = exp_spec.split("_")
        config = ExperimentConfig()
        for item in config_items:
            if item.startswith("dp"):
                config.dp_size = int(item[2:])
            elif item.startswith("tp"):
                config.tp_size = int(item[2:])
            elif item.startswith("pp"):
                config.pp_size = int(item[2:])
            elif item.startswith("sl"):
                config.enc_seqlen = int(item[2:])
            elif item.startswith("encsl"):
                config.enc_seqlen = int(item[5:])
            elif item.startswith("decsl"):
                config.dec_seqlen = int(item[5:])
            elif item.startswith("gbs"):
                config.gbs = int(item[3:])
            elif item.startswith("mbs"):
                config.mbs = int(item[3:])
            elif item.startswith("rc"):
                config.rc = item[2:]
            elif item.startswith("zero"):
                config.ds_level = int(item[4:])
            elif item.startswith("spp"):
                config.spp = True
        # test status
        config.status = ExperimentConfig.parse_experiment_status(exp_dir)
        return config


def generate_dynapipe_exp_configs(args):
    seqlens = [int(sl) for sl in args.sequence_length_range.split(",")]
    gbs_tokens = [int(gbs) for gbs in args.global_batch_size_range.split(",")]
    ####  Sequence length  ####
    for seqlen in seqlens:
        if args.model_type == "t5":
            # TODO: may run more experiments with decoder length
            # differenet from encoder length
            args.encoder_seq_length = seqlen
            args.decoder_seq_length = seqlen
        else:
            args.seq_length = seqlen
        ####  global batch size  ####
        for gbs in gbs_tokens:
            args.tokens_per_global_batch = gbs
            ####  Parallelism  ####
            for dp, tp, pp in grid_search_parallelism(args):
                args.tensor_parallel_size = tp
                args.pipeline_parallel_size = pp
                if pp > 1:
                    args.pp_split_rank = get_pp_split_rank(pp)
                args.plopt_device_to_node = get_pp_device_to_node_str(args)
                args.plopt_layer_to_device = get_layer_to_device(args)
                ####  Micro batch size  ####
                for mbs in grid_search_microbatch_size(dp, args):
                    args.micro_batch_size = mbs
                    ###  Recomputation  ####
                    for recompute_level in grid_search_recomputation(args):
                        args.recompute_level = recompute_level
                        ####  ZeRO Stage  ####
                        for enable_ds, ds_stage in grid_search_ds_stage(args):
                            args.enable_deepspeed = enable_ds
                            args.deepspeed_zero_stage = ds_stage
                            # config
                            config = ExperimentConfig()
                            if args.model_type == "t5":
                                config.enc_seqlen = args.encoder_seq_length
                                config.dec_seqlen = args.decoder_seq_length
                            else:
                                config.enc_seqlen = args.seq_length
                            config.gbs = args.tokens_per_global_batch
                            config.dp_size = dp
                            config.tp_size = tp
                            config.pp_size = pp
                            config.mbs = mbs
                            config.rc = args.recompute_level
                            config.ds_level = args.deepspeed_zero_stage
                            yield args, config


def pgrep():
    out = os.popen("pgrep redis").read().strip()
    return list(map(int, out.splitlines()))

def kill_non_controller_redis_servers(args):
    if args.node_rank != 0:
        os.system("pkill -9 -f 'redis-server'")
    else:
        kv: RedisKVStore = args.kvstore
        server_pid = kv.server.pid if kv.server else None
        redis_pids = list(map(int, os.popen("pgrep redis").read().strip().splitlines()))
        for pid in redis_pids:
            if pid != server_pid:
                os.system(f"kill -9 {pid}")


def cleanup_plopt_job(args):
    os.system("pkill -9 -f 'pretrain_t5'")
    os.system("pkill -9 -f 'pretrain_gpt'")
    kill_non_controller_redis_servers(args)


def run_batch_experiments(args):
    global print_fn
    past_success_configs = []
    past_failures_configs = []
    exp_dir = os.path.join(EXPERIMENT_DIR_PREFIX, args.experiment_name)
    if os.path.isdir(exp_dir):
        for exp_spec_dir in [
            x for x in os.listdir(exp_dir) if os.path.isdir(x)
        ]:
            full_exp_spec_dir = os.path.join(exp_dir, exp_spec_dir)
            config = ExperimentConfig.parse_history_experiments(
                full_exp_spec_dir
            )
            if config.status == "success":
                past_success_configs.append(config)
            else:
                past_failures_configs.append(config)
    config_iterator = generate_dynapipe_exp_configs(args)
    if args.node_rank == 0:
        from tqdm import tqdm
        config_iterator = tqdm(config_iterator)
        print_fn = config_iterator.write
    else:
        print_fn = lambda *args, **kwargs: None
    for current_args, current_exp_config in config_iterator:
        should_skip = False
        for past_success_config in past_success_configs:
            past_success_config: ExperimentConfig
            if past_success_config.speed_dominates(current_exp_config):
                print_fn(
                    f"Skip {current_exp_config} because it is slower than {past_success_config}"
                )
                should_skip = True
                break
        if should_skip:
            continue
        for past_failure_config in past_failures_configs:
            past_failure_config: ExperimentConfig
            if current_exp_config.memory_dominates(past_failure_config):
                print_fn(
                    f"Skip {current_exp_config} because it consumes more memory than {past_failure_config}"
                )
                should_skip = True
                break
        if should_skip:
            continue
        if args.enable_plopt:
            initial_memlimit = current_args.plopt_device_memory_limit
        while True:
            current_args = _check_training_args(current_args)
            current_args, exp_logging_dir, should_skip = _check_logging_args(
                current_args
            )
            spec_basename = os.path.basename(exp_logging_dir)
            if args.node_rank == 0:
                config_iterator.set_description("Running experiment {}".format(spec_basename))
            if should_skip:
                # the experiment has been run
                break
            # barrier before starting the experiment
            assert hasattr(args, "kvstore") and args.kvstore is not None
            kv: RedisKVStore = args.kvstore
            kv.barrier()
            # exchange exp config
            gathered_exp_configs = kv.gather(current_exp_config)
            if kv.node_rank == 0:
                # check if all nodes have the same exp config
                if not all(
                    [
                        gathered_exp_config == current_exp_config
                        for gathered_exp_config in gathered_exp_configs
                    ]
                ):
                    print("ERROR: All nodes must have the same experiment config.")
                    kv.send_abort_signal()
                    sys.exit(1)
            kv.barrier()
            current_args = _create_deepspeed_config(
                current_args, exp_logging_dir
            )
            shell_script = _get_shell_script(current_args)
            shell_script_path = os.path.join(exp_logging_dir, "run.sh")
            with open(shell_script_path, "w") as f:
                f.write(shell_script)
            # all stdout and stderr are redirected to current_args.stdout_stderr_log
            p = subprocess.Popen(f"bash {shell_script_path}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            assert (
                current_args.stdout_stderr_log is not None
            ), "stdout_stderr_log must be specified for batch experiments."
            prev_content = None
            last_progress = time.time()
            should_restart = False
            while p.poll() is None:
                # check if the stdout/stderr log has progress
                if not os.path.exists(current_args.stdout_stderr_log):
                    # the job has not started yet
                    time.sleep(EXPERIMENT_PROGRESS_POLL_INTERVAL)
                    continue
                should_abort = False
                with open(current_args.stdout_stderr_log, "r") as f:
                    current_content = f.read()
                if ("Failed to generate microbatches." in current_content or 
                    "No feasible schedule" in current_content or 
                    "OutOfMemoryError" in current_content):
                    # error
                    should_abort = True
                    if args.enable_plopt:
                        should_restart = True
                elif current_content != prev_content:
                    # progress
                    prev_content = current_content
                    last_progress = time.time()
                else:
                    # no progress
                    if (
                        time.time() - last_progress
                        > EXPERIMENT_PROGRESS_TIMEOUT
                    ):
                        # timeout
                        should_abort = True
                        if args.enable_plopt:
                            should_restart = True
                if should_abort and should_restart:
                    kv.set(spec_basename + "status", "restart")
                elif should_abort:
                    kv.set(spec_basename + "status", "abort")
                # get the most updated status from all nodes
                current_status = kv.get(spec_basename + "status")
                should_abort = current_status in ["abort", "restart"]
                should_restart = current_status == "restart"
                if should_abort:
                    # kill the job on all nodes
                    if p.poll() is None:
                        p.kill()
                    cleanup_plopt_job(args)
                    break
                time.sleep(EXPERIMENT_PROGRESS_POLL_INTERVAL)
            kv.barrier()
            # check restart status again incase some nodes exit early
            current_status = kv.get(spec_basename + "status")
            should_restart = current_status == "restart"
            cleanup_plopt_job(args)
            if not should_restart:
                break
            else:
                # decrease memory limit and restart
                if current_args.plopt_device_memory_limit < 10000:
                    break
                current_args.plopt_device_memory_limit -= 1000
                # nuke the result dir
                shutil.rmtree(exp_logging_dir)

        # check current experiment status
        current_exp_config.status = ExperimentConfig.parse_experiment_status(
            exp_logging_dir
        )
        if current_exp_config.status == "success":
            past_success_configs.append(current_exp_config)
        elif current_exp_config.status == "failure":
            past_failures_configs.append(current_exp_config)
        else:
            raise ValueError(
                f"Failed to parse experiment status: {exp_logging_dir}"
            )
        if args.enable_plopt:
            current_args.plopt_device_memory_limit = initial_memlimit


def _parse_args():
    parser = argparse.ArgumentParser("Experiment runner for T5 and GPT.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--stdout_stderr_log",
        type=str,
        help="Path to the stdout/stderr log file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["t5", "gpt"],
        help="Type of model to benchmark on.",
    )
    parser, cluster_group = _add_cluster_args(parser)
    parser, model_group = _add_model_args(parser)
    parser, data_group = _add_data_args(parser)
    parser, training_group = _add_training_args(parser)
    parser, plopt_group = _add_plopt_args(parser)
    parser, exp_group = _add_experiment_args(parser)
    args = parser.parse_args()

    # if experiment config exists, load it
    config_path = os.path.join(EXP_CONFIG_DIR, args.experiment_name + ".json")
    print_fn(f"Loading experiment config from {config_path}")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        required_args = [
            ("cluster_config", str),
            ("model_config", str),
            ("data_config", str),
            ("training_config", str),
        ]
        for key, type_func in required_args:
            assert (
                key in config
            ), f"Missing key {key} in experiment config file {config_path}"
            value = type_func(config[key])
            setattr(args, key, value)
        # add other optional args
        for key, value in config.items():
            if key not in [k for k, _ in required_args]:
                setattr(args, key, value)

    # load config files
    _postprocess_group_args(args, cluster_group, "cluster_config")
    _postprocess_group_args(
        args,
        model_group,
        "model_config",
        optional_args=(
            ["num_layers"]
            if args.model_type == "t5"
            else ["encoder_num_layers", "decoder_num_layers"]
        ),
    )
    _postprocess_group_args(
        args,
        data_group,
        "data_config",
        optional_args=["merge_file"] if args.model_type == "t5" else None,
    )
    # if running experiment in batch, don't check training args that we
    # are going to search through
    training_optional_args = []
    plopt_optional_args = ["plopt_enable_packing"]
    if args.batch_experiments:
        assert (
            args.sequence_length_range is not None
        ), "Must specify sequence length range when running experiments in batch."
        assert (
            args.global_batch_size_range is not None
        ), "Must specify global batch size range when running experiments in batch."
        training_optional_args += ["deepspeed_zero_stage"]
        training_optional_args += [
            "seq_length",
            "encoder_seq_length",
            "decoder_seq_length",
        ]
        training_optional_args += ["tokens_per_global_batch"]
        plopt_optional_args += [
            "plopt_device_to_node",
            "plopt_layer_to_device",
        ]
    else:
        training_optional_args = ["deepspeed_zero_stage"]
        if args.model_type == "t5":
            training_optional_args += ["seq_length"]
        else:
            training_optional_args += [
                "encoder_seq_length",
                "decoder_seq_length",
            ]
    _postprocess_group_args(
        args,
        training_group,
        "training_config",
        optional_args=training_optional_args,
    )
    _postprocess_group_args(
        args,
        plopt_group,
        "training_config",
        optional_args=plopt_optional_args
        + [
            "plopt_debug_logging_dir",
            "plopt_debug_dump_ep_prefix",
            "plopt_debug_dump_memory_prefix",
        ],
        switch_arg="enable_plopt",
    )

    # init kvstore for exp control
    kvstore = RedisKVStore(args)
    args.kvstore = kvstore

    # check args
    args = _check_cluster_args(args)

    if args.batch_experiments:
        # training and logging args are generated for each experiment
        return args, None, False
    else:
        args = _check_training_args(args)
        args, exp_logging_dir, should_skip = _check_logging_args(args)

    return args, exp_logging_dir, should_skip


def _get_shell_script(args):
    # construct pipeline_args
    if args.model_type == "t5" and args.pipeline_parallel_size > 1:
        pipeline_args = (
            f"--pipeline-model-parallel-split-rank {args.pp_split_rank}"
        )
        # interleaved schedule
        n_encoder_layers_per_device = args.encoder_num_layers // args.pipeline_parallel_size
        n_decoder_layers_per_device = args.decoder_num_layers // args.pipeline_parallel_size
        assert n_encoder_layers_per_device == n_decoder_layers_per_device, (
            "Different number of layers per device for encoder and decoder "
            "is not supported."
        )
        pipeline_args += (
            f" --num-layers-per-virtual-pipeline-stage {n_encoder_layers_per_device}"
        )
    else:
        pipeline_args = ""
    # construct recompute args
    recompute_args = ""
    if args.enable_plopt:
        assert (
            args.recompute_level == "none"
        ), "Plopt uses dynamic recomputation, recompute_level is overriden."
        # although recompute_level is none, we still need to set recompute
        # method, which is required if full recompute is chosen
        recompute_args = "--recompute-method uniform"
    if args.recompute_level == "selective":
        recompute_args = "--recompute-activations"
    elif args.recompute_level == "full":
        recompute_args = (
            "--recompute-granularity full --recompute-method uniform"
        )
    else:
        assert (
            args.recompute_level == "none"
        ), f"Invalid recompute level {args.recompute_level}"
    # construct dynamic batch args
    if args.enable_plopt:
        batching_args = (
            "--dynamic-batchsize "
            + f"--tokens-per-global-batch {args.tokens_per_global_batch}"
        )
    else:
        batching_args = "--pack-dataset"
    # construct plopt args
    if not args.enable_plopt:
        plopt_args = ""
    else:
        plopt_args = [
            "--use-plopt",
            f"--plopt-cost-model {args.plopt_cost_model}",
            f"--plopt-device-to-node {args.plopt_device_to_node}",
            f"--plopt-device-memory-limit {args.plopt_device_memory_limit}",
            f"--plopt-intra-node-bw {args.plopt_intra_node_bw}",
            f"--plopt-inter-node-bw {args.plopt_inter_node_bw}",
            f"--plopt-layer-to-device {args.plopt_layer_to_device}",
            "--plopt-prefetch-planner-num-workers "
            + f"{args.plopt_prefetch_planner_num_workers}",
            f"--plopt-zero-stage {args.deepspeed_zero_stage}",
            "--plopt-reserve-all-memory",
            "--plopt-custom-allocator",
        ]
        if args.model_type == "gpt":
            plopt_args.append("--plopt-seqlen-offset 1")
        if args.plopt_enable_packing:
            plopt_args.append("--plopt-enable-packing")
        plopt_args = " ".join(plopt_args)
    # construct deepspeed args
    if not args.enable_deepspeed:
        deepspeed_args = ""
    else:
        deepspeed_args = [
            "--deepspeed",
            f"--deepspeed_config {args.deepspeed_config}",
        ]
        deepspeed_args = " ".join(deepspeed_args)
    template_args = vars(args)
    template_args.update(
        {
            "pipeline_args": pipeline_args,
            "recompute_args": recompute_args,
            "batching_args": batching_args,
            "plopt_args": plopt_args,
            "deepspeed_args": deepspeed_args,
        }
    )
    with open(TEMPLATE_PATH.format(args.model_type), "r") as f:
        template = Template(f.read())
    return template.substitute(template_args)


def main():
    args, exp_logging_dir, should_skip = _parse_args()
    if should_skip:
        print_fn("Experiment directory already exists, skipping.")
        return
    if args.batch_experiments:
        run_batch_experiments(args)
    else:
        args = _create_deepspeed_config(args, exp_logging_dir)
        # get shell script
        shell_script = _get_shell_script(args)
        # run shell script
        shell_script_path = os.path.join(exp_logging_dir, "run.sh")
        with open(shell_script_path, "w") as f:
            f.write(shell_script)
        subprocess.run(f"bash {shell_script_path}", shell=True)
    args.kvstore.barrier()
    if args.kvstore.is_master:
        if args.kvstore.server.poll() is not None:
            return
        os.system("pkill -f 'redis-server'")


if __name__ == "__main__":
    main()
