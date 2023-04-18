import argparse
import torch
import multiprocessing as mp
from run_single_gpu_benchmark import run_benchmark
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("Benchmark T5 to obtain cost model.")
    parser.add_argument("out_dir", type=str, help="Output directory for benchmark results")
    return parser.parse_args()

def _profile_func(queue: mp.Queue, gpu_index, assigned_mbs, assigned_enc_seqlen, 
                  assigned_dec_seqlen, assigned_recompute_type,
                  out_dir):
    for mbs in assigned_mbs:
        for enc_seqlen in assigned_enc_seqlen:
            for dec_seqlen in assigned_dec_seqlen:
                for recompute_type in assigned_recompute_type:
                    run_benchmark(
                        enc_seqlen, dec_seqlen, mbs, 
                        output_dir=out_dir,
                        device=gpu_index,
                        recompute_type=recompute_type,
                        use_flash_attn=False,
                        log_file=f"microbenchmark_{gpu_index}.log"
                    )
                    queue.put("Progress")

if __name__ == "__main__":
    args = parse_args()
    n_gpus = torch.cuda.device_count()

    candidate_mbs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    candidate_enc_seqlen = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    candidate_dec_seqlen = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    candidate_recompute_type = ["None", "Selective", "Full"]

    per_gpu_mbs = len(candidate_mbs) // n_gpus
    mbs_args = [candidate_mbs[i * per_gpu_mbs: (i + 1) * per_gpu_mbs] for i in range(n_gpus)]

    subprocesses = []
    q = mp.Queue()
    for i in range(n_gpus):
        p = mp.Process(target=_profile_func, args=(q, i, mbs_args[i], candidate_enc_seqlen, candidate_dec_seqlen, candidate_recompute_type, args.out_dir))
        p.start()
        subprocesses.append(p)

    total_jobs = len(candidate_mbs) * len(candidate_enc_seqlen) * len(candidate_dec_seqlen) * len(candidate_recompute_type)
    with tqdm(total=total_jobs) as pbar:
        while True:
            q.get()
            pbar.update(1)
            if pbar.n == total_jobs:
                break