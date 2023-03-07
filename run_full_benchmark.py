from run_single_gpu_benchmark import run_benchmark
from tqdm import tqdm

for mbs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    for enc_seqlen in tqdm([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], leave=False):
        for dec_seqlen in tqdm([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], leave=False):
            for recompute_type in ["None", "Selective", "Full"]:
                run_benchmark(
                    enc_seqlen, dec_seqlen, mbs, 
                    output_dir="/root/Megatron-LM/microbench_t5_11b",
                    device=0,
                    recompute_type=recompute_type
                )