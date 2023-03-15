from run_single_gpu_benchmark import run_benchmark
from tqdm import tqdm

candidate_mbs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
candidate_enc_seqlen = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
candidate_dec_seqlen = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
candidate_recompute_type = ["None", "Selective", "Full"]

for mbs in candidate_mbs:
    for enc_seqlen in tqdm(candidate_enc_seqlen, leave=False):
        for dec_seqlen in tqdm(candidate_dec_seqlen, leave=False):
            for recompute_type in candidate_recompute_type:
                run_benchmark(
                    enc_seqlen, dec_seqlen, mbs, 
                    output_dir="/root/Megatron-LM/microbench_t5_11b",
                    device=0,
                    recompute_type=recompute_type
                )