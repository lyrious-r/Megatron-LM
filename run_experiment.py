import argparse
import json
import os
import subprocess
from string import Template
from typing import Optional

EXP_CONFIG_DIR = "./experiment_configs"
TEMPLATE_PATH = os.path.join(EXP_CONFIG_DIR, "finetune_t5.template")
DEEPSPEED_TEMPLATE_PATH = os.path.join(
    EXP_CONFIG_DIR, "deepspeed_config.template"
)
EXPERIMENT_DIR_PREFIX = "./experiments/"


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
    return parser, group


def _add_training_args(parser):
    group = parser.add_argument_group(title="Training Config")
    group.add_argument(
        "--training_config",
        type=str,
        help="Load training spec from config file.",
    )
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
        choices=[0, 1, 2, 3],
        help="Which stage of ZeRO to use.",
    )
    return parser, group


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
    args.max_pos_embeddings = max(
        args.encoder_seq_length, args.decoder_seq_length
    )
    # check micro_batch_size and global_batch_size
    effective_tokens_per_sequence = (
        args.encoder_seq_length + args.decoder_seq_length
    )
    expected_global_batch_size = (
        args.tokens_per_global_batch // effective_tokens_per_sequence
    )
    if args.global_batch_size != expected_global_batch_size:
        print("Override global batch size to", expected_global_batch_size)
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
    # if plopt_config is not specified, require the following args
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
        help="Memory limit per device in GB.",
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
        "--plopt_debug_level", type=str, default="DEBUG", help="Debug level."
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
    return parser, group


def _check_logging_args(args):
    exp_logging_dir = os.path.join(
        EXPERIMENT_DIR_PREFIX,
        args.experiment_name,
        "encsl{}_decsl{}_gbs{}".format(
            args.encoder_seq_length,
            args.decoder_seq_length,
            args.global_batch_size,
        ),
    )
    if not os.path.exists(exp_logging_dir):
        os.makedirs(exp_logging_dir)
    if args.enable_plopt:
        if args.plopt_dump_stats:
            args.plopt_debug_level = "DEBUG"
        else:
            args.plopt_debug_level = "INFO"
        if args.plopt_debug_logging_dir is None:
            args.plopt_debug_logging_dir = os.path.join(
                exp_logging_dir, "plopt_logs"
            )
        if args.plopt_debug_dump_ep_prefix is None:
            args.plopt_debug_dump_ep_prefix = os.path.join(
                exp_logging_dir, "plopt_ep_stats"
            )
        if args.plopt_debug_dump_memory_prefix is None:
            args.plopt_debug_dump_memory_prefix = os.path.join(
                exp_logging_dir, "plopt_memory_stats"
            )
    else:
        args.plopt_debug_logging_dir = "UNUSED"
        args.plopt_debug_dump_ep_prefix = "UNUSED"
        args.plopt_debug_dump_memory_prefix = "UNUSED"
    if args.stdout_stderr_log is None:
        args.stdout_stderr_log = os.path.join(
            exp_logging_dir, "stdout_stderr.log"
        )
    # dump all args to a file
    args_file = os.path.join(exp_logging_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    return args, exp_logging_dir


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
        # create deepspeed config
        with open(DEEPSPEED_TEMPLATE_PATH, "r") as f:
            template = Template(f.read())
        config_str = template.substitute(
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=n_micro_batches,
            zero_stage=args.deepspeed_zero_stage,
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


def _parse_args():
    parser = argparse.ArgumentParser("Experiment runner for T5 model.")
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
    parser, cluster_group = _add_cluster_args(parser)
    parser, model_group = _add_model_args(parser)
    parser, data_group = _add_data_args(parser)
    parser, training_group = _add_training_args(parser)
    parser, plopt_group = _add_plopt_args(parser)
    args = parser.parse_args()

    # if experiment config exists, load it
    config_path = os.path.join(EXP_CONFIG_DIR, args.experiment_name + ".json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        required_args = [
            ("cluster_config", str),
            ("model_config", str),
            ("data_config", str),
            ("training_config", str),
        ]
        if args.enable_plopt:
            required_args.append(("plopt_dump_stats", bool))
        for key, type_func in required_args:
            assert (
                key in config
            ), f"Missing key {key} in experiment config file {config_path}"
            value = type_func(config[key])
            setattr(args, key, value)

    # load config files
    _postprocess_group_args(args, cluster_group, "cluster_config")
    _postprocess_group_args(args, model_group, "model_config")
    _postprocess_group_args(args, data_group, "data_config")
    _postprocess_group_args(
        args,
        training_group,
        "training_config",
        optional_args=["deepspeed_zero_stage"],
    )
    _postprocess_group_args(
        args,
        plopt_group,
        "training_config",
        optional_args=[
            "plopt_debug_logging_dir",
            "plopt_debug_dump_ep_prefix",
            "plopt_debug_dump_memory_prefix",
        ],
        switch_arg="enable_plopt",
    )

    # check args
    args = _check_cluster_args(args)
    args = _check_training_args(args)
    args, exp_logging_dir = _check_logging_args(args)

    return args, exp_logging_dir


def _get_shell_script(args):
    # construct pipeline_args
    pipeline_args = (
        f"--pipeline-model-parallel-split-rank {args.pp_split_rank}"
    )
    # construct recompute args
    if args.recompute_level == "none":
        recompute_args = ""
    elif args.recompute_level == "selective":
        recompute_args = "--recompute-activations"
    elif args.recompute_level == "full":
        recompute_args = (
            "--recompute-granularity full --recompute-method uniform"
        )
    else:
        raise ValueError(f"Invalid recompute level {args.recompute_level}")
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
            "--plopt-reserve-all-memory",
            "--plopt-custom-allocator",
        ]
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
    with open(TEMPLATE_PATH, "r") as f:
        template = Template(f.read())
    return template.substitute(template_args)


def main():
    args, exp_logging_dir = _parse_args()
    args = _create_deepspeed_config(args, exp_logging_dir)
    # get shell script
    shell_script = _get_shell_script(args)
    # run shell script
    shell_script_path = os.path.join(exp_logging_dir, "run.sh")
    with open(shell_script_path, "w") as f:
        f.write(shell_script)
    subprocess.run(f"bash {shell_script_path}", shell=True)


if __name__ == "__main__":
    main()
