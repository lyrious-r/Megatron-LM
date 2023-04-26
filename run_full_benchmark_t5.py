import argparse
import math
import multiprocessing as mp
import os

import torch
from tqdm import tqdm

from run_single_gpu_benchmark_t5 import run_benchmark


def parse_args():
    parser = argparse.ArgumentParser("Benchmark T5 to obtain cost model.")
    parser.add_argument(
        "out_dir", type=str, help="Output directory for benchmark results"
    )
    parser.add_argument("--log_dir", type=str, help="Log directory")

    args = parser.parse_args()
    if args.log_dir is None:
        args.log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(args.log_dir, exist_ok=True)
    return args


def _profile_func(
    queue: mp.Queue,
    tp_size,
    devices,
    assigned_mbs,
    assigned_enc_seqlen,
    assigned_dec_seqlen,
    assigned_recompute_type,
    out_dir,
    log_dir,
):
    for mbs in assigned_mbs:
        for enc_seqlen in assigned_enc_seqlen:
            for dec_seqlen in assigned_dec_seqlen:
                for recompute_type in assigned_recompute_type:
                    for n_layers in [2, 1]:
                        retcode = run_benchmark(
                            tp_size,
                            enc_seqlen,
                            dec_seqlen,
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
                        )
                        if retcode == 0:
                            break
                    queue.put("Progress")


if __name__ == "__main__":
    args = parse_args()
    n_gpus = torch.cuda.device_count()

    tensor_parallel_size = [
        2**i for i in range(math.floor(math.log2(n_gpus)) + 1)
    ]
    candidate_mbs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    candidate_enc_seqlen = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    candidate_dec_seqlen = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
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
                    candidate_enc_seqlen,
                    candidate_dec_seqlen,
                    candidate_recompute_type,
                    args.out_dir,
                    args.log_dir,
                ),
            )
            p.start()
            subprocesses.append(p)
        total_jobs = (
            len(candidate_mbs)
            * len(candidate_enc_seqlen)
            * len(candidate_dec_seqlen)
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
