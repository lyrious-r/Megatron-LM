import argparse
import math
import multiprocessing as mp
import os
from dataclasses import dataclass

import torch
from tqdm import tqdm

from run_single_gpu_benchmark_gpt import run_benchmark


def parse_args():
    parser = argparse.ArgumentParser("Benchmark GPT to obtain cost model.")
    parser.add_argument(
        "out_dir", type=str, help="Output directory for benchmark results"
    )
    parser.add_argument("--log_dir", type=str, help="Log directory")

    args = parser.parse_args()
    if args.log_dir is None:
        args.log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(args.log_dir, exist_ok=True)
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
    rc: str

    def dominates(self, other):
        assert isinstance(
            other, BenchmarkConfig
        ), "Can only compare with BenchmarkConfig"
        if (
            self.mbs >= other.mbs
            and self.seqlen >= other.seqlen
            and RC_MAP[self.rc] >= RC_MAP[other.rc]
        ):
            return True


def _profile_func(
    queue: mp.Queue,
    tp_size,
    devices,
    assigned_mbs,
    assigned_seqlen,
    assigned_recompute_type,
    out_dir,
    log_dir,
):
    oom_configs = []
    for mbs in assigned_mbs:
        for seqlen in assigned_seqlen:
            for recompute_type in assigned_recompute_type:
                current_config = BenchmarkConfig(mbs, seqlen, recompute_type)
                should_skip = False
                for past_oom in oom_configs:
                    if current_config.dominates(past_oom):
                        # skip this config since it must also oom
                        should_skip = True
                        break
                if not should_skip:
                    oom = True
                    for n_layers in [3, 2, 1]:
                        retcode = run_benchmark(
                            tp_size,
                            seqlen,
                            mbs,
                            n_layers,
                            output_dir=out_dir,
                            devices=devices,
                            recompute_type=recompute_type,
                            use_flash_attn=False,
                            log_file=os.path.join(
                                log_dir, f"microbenchmark_{devices}.log"
                            ),
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
        12288,
        16384,
        24576,
        32768,
    ]
    candidate_recompute_type = ["None", "Selective", "Full"]

    subprocesses = []
    q = mp.Queue()
    for tp_size_idx, tp_size in enumerate(tensor_parallel_size):
        device_groups = [
            list(range(tp_size * i, tp_size * (i + 1)))
            for i in range(n_gpus // tp_size)
        ]
        per_device_group_mbs = len(candidate_mbs) // len(device_groups)
        last_group_remainder = len(candidate_mbs) % len(device_groups)
        mbs_args = [
            candidate_mbs[
                i * per_device_group_mbs : (i + 1) * per_device_group_mbs
            ]
            for i in range(len(device_groups))
        ]
        if last_group_remainder > 0:
            mbs_args[-1] += candidate_mbs[-last_group_remainder:]
        assert sum([len(mbs) for mbs in mbs_args]) == len(candidate_mbs)
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
