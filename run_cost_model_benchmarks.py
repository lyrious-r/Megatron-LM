import argparse
import math
import multiprocessing as mp
import os
import json
import time
from dataclasses import dataclass

import torch
from tqdm import tqdm

from run_single_gpu_benchmark_gpt import run_benchmark as run_benchmark_gpt
from run_single_gpu_benchmark_t5 import run_benchmark as run_benchmark_t5


def parse_args():
    parser = argparse.ArgumentParser("Benchmark GPT to obtain cost model.")
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Output directory for benchmark results"
    )
    parser.add_argument("--model_config", type=str, help="Model config path")

    args = parser.parse_args()
    args.log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.exists(args.model_config):
        raise ValueError(f"Model config path {args.model_config} does not exist.")
    # read model config
    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    model_config_basename = os.path.basename(args.model_config)
    if model_config_basename.startswith("t5"):
        args.model_type = "t5"
    elif model_config_basename.startswith("gpt"):
        args.model_type = "gpt"
    else:
        raise ValueError(f"Unrecognized model type {args.model_type}.")
    args.hidden_size = model_config["hidden_size"]
    args.num_attn_heads = model_config["num_attn_heads"]
    args.kv_channels = model_config["kv_channels"]
    args.ffn_hidden_size = model_config["ffn_hidden_size"]
    return args


RC_MAP = {
    "None": 0,
    "Selective": 1,
    "Full": 2,
}


@dataclass(eq=True)
class BenchmarkConfig:
    mbs: int
    seqlen: int
    seqlen_dec: int
    rc: str

    def dominates(self, other):
        assert isinstance(
            other, BenchmarkConfig
        ), "Can only compare with BenchmarkConfig"
        if (
            self.mbs >= other.mbs
            and self.seqlen >= other.seqlen
            and self.seqlen_dec >= other.seqlen_dec
            and RC_MAP[self.rc] <= RC_MAP[other.rc]
        ):
            return True


def _profile_func(
    queue: mp.Queue,
    tp_size,
    devices,
    assigned_mbs,
    assigned_seqlen,
    assigned_recompute_type,
    model_type,
    hidden_size,
    num_attn_heads,
    ffn_hidden_size,
    out_dir,
    log_dir,
):
    if model_type == "gpt":
        # assigned_seqlen_dec is unused for gpt
        assigned_seqlen_dec = [0]
    else:
        assigned_seqlen_dec = assigned_seqlen
    oom_configs = []
    for mbs in assigned_mbs:
        for seqlen in assigned_seqlen:
            for seqlen_dec in assigned_seqlen_dec:
                for recompute_type in assigned_recompute_type:
                    current_config = BenchmarkConfig(mbs, seqlen, seqlen_dec, recompute_type)
                    should_skip = False
                    for past_oom in oom_configs:
                        if current_config.dominates(past_oom):
                            # skip this config since it must also oom
                            should_skip = True
                            break
                    if not should_skip:
                        oom = True
                        for n_layers in [3, 2, 1]:
                            print(f"Running benchmark with {n_layers} layers.")
                            if model_type == "gpt":
                                retcode = run_benchmark_gpt(
                                    tp_size,
                                    seqlen,
                                    mbs,
                                    n_layers,
                                    hidden_size=hidden_size,
                                    n_attn_heads=num_attn_heads,
                                    ffn_hidden_size=ffn_hidden_size,
                                    output_dir=out_dir,
                                    devices=devices,
                                    recompute_type=recompute_type,
                                    use_flash_attn=False,
                                    log_file=os.path.join(
                                        log_dir, f"microbenchmark_{devices}.log"
                                    ),
                                    benchmark_iters=20,
                                )
                            elif model_type == "t5":
                                retcode = run_benchmark_t5(
                                    tp_size,
                                    seqlen,
                                    seqlen_dec,
                                    mbs,
                                    n_layers,
                                    n_layers,
                                    output_dir=out_dir,
                                    devices=devices,
                                    recompute_type=recompute_type,
                                    use_flash_attn=False,
                                    log_file=os.path.join(
                                        log_dir, f"microbenchmark_{devices}.log"
                                    ),
                                    benchmark_iters=20,
                                )
                            if retcode == 0:
                                oom = False
                                break
                        if oom:
                            oom_configs.append(current_config)
                    queue.put("Progress")


if __name__ == "__main__":
    args = parse_args()
    n_gpus = torch.cuda.device_count()

    tensor_parallel_size = [
        2**i for i in range(math.floor(math.log2(n_gpus)) + 1)
    ]
    candidate_mbs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    candidate_seqlen = [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        6144,
        8192,
    ]
    candidate_recompute_type = ["None", "Selective", "Full"]

    subprocesses = []
    q = mp.Queue()
    for tp_size_idx, tp_size in enumerate(tensor_parallel_size):
        t = time.time()
        device_groups = [
            list(range(tp_size * i, tp_size * (i + 1)))
            for i in range(n_gpus // tp_size)
        ]
        per_device_group_mbs = (len(candidate_mbs) + len(device_groups) - 1) // len(device_groups)
        # round robin assignment
        mbs_args = []
        all_mbs_args = []
        for i in range(len(device_groups)):
            current_group_mbs = []
            for n in range(per_device_group_mbs):
                if n % 2 == 0:
                    mbs_id = i + n * len(device_groups)
                else:
                    mbs_id = (len(device_groups) - i - 1) + n * len(device_groups)
                if mbs_id >= len(candidate_mbs):
                    continue
                current_group_mbs.append(
                    candidate_mbs[mbs_id]
                )
            mbs_args.append(current_group_mbs)
            all_mbs_args += current_group_mbs
        assert sorted(all_mbs_args) == sorted(candidate_mbs)
        for device_group_id, devices in enumerate(device_groups):
            p = mp.Process(
                target=_profile_func,
                args=(
                    q,
                    tp_size,
                    devices,
                    mbs_args[device_group_id],
                    candidate_seqlen,
                    candidate_recompute_type,
                    args.model_type,
                    args.hidden_size,
                    args.num_attn_heads,
                    args.ffn_hidden_size,
                    args.out_dir,
                    args.log_dir,
                ),
            )
            p.start()
            subprocesses.append(p)
        total_jobs = (
            len(candidate_mbs)
            * len(candidate_seqlen)
            * len(candidate_recompute_type)
        )
        if args.model_type == "t5":
            total_jobs *= len(candidate_seqlen)
        with tqdm(
            total=total_jobs,
            desc="[{}/{}] TP size: {}".format(
                tp_size_idx + 1, len(tensor_parallel_size), tp_size
            ),
        ) as pbar:
            while True:
                q.get()
                pbar.update(1)
                if pbar.n == total_jobs:
                    break
        for p in subprocesses:
            p.join()
        PROFILE_DUR_OUT_PATH = "./profile_duration.txt"
        if not os.path.exists(PROFILE_DUR_OUT_PATH):
            with open("./profile_duration.txt", "w") as f:
                f.write("tp_size, duration\n")
        with open("./profile_duration.txt", "a"):
            with open("./profile_duration.txt", "a") as f:
                f.write(f"{tp_size}, {time.time() - t}\n")
