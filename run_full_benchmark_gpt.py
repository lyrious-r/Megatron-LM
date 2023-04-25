import argparse
import torch
import multiprocessing as mp
from run_single_gpu_benchmark_gpt import run_benchmark
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("Benchmark GPT to obtain cost model.")
    parser.add_argument("out_dir", type=str, help="Output directory for benchmark results")
    return parser.parse_args()

def _profile_func(queue: mp.Queue, gpu_index, assigned_mbs, assigned_seqlen, 
                  assigned_recompute_type, out_dir):
    for mbs in assigned_mbs:
        for seqlen in assigned_seqlen:
            for recompute_type in assigned_recompute_type:
                for n_layers in [3, 2, 1]:
                    retcode = run_benchmark(
                        seqlen, mbs, n_layers,
                        output_dir=out_dir,
                        device=gpu_index,
                        recompute_type=recompute_type,
                        use_flash_attn=False,
                        log_file=f"microbenchmark_{gpu_index}.log"
                    )
                    if retcode == 0:
                        break
                queue.put("Progress")

if __name__ == "__main__":
    args = parse_args()
    n_gpus = torch.cuda.device_count()

    candidate_mbs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    candidate_seqlen = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
    candidate_recompute_type = ["None", "Selective", "Full"]

    per_gpu_mbs = len(candidate_mbs) // n_gpus
    mbs_args = [candidate_mbs[i * per_gpu_mbs: (i + 1) * per_gpu_mbs] for i in range(n_gpus)]

    subprocesses = []
    q = mp.Queue()
    for i in range(n_gpus):
        p = mp.Process(target=_profile_func, args=(q, i, mbs_args[i], candidate_seqlen, candidate_recompute_type, args.out_dir))
        p.start()
        subprocesses.append(p)

    total_jobs = len(candidate_mbs) * len(candidate_seqlen) * len(candidate_recompute_type)
    with tqdm(total=total_jobs) as pbar:
        while True:
            q.get()
            pbar.update(1)
            if pbar.n == total_jobs:
                break