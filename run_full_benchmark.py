from run_single_gpu_benchmark import run_benchmark

for mbs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    for enc_seqlen in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        for dec_seqlen in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            for recompute_type in ["None", "Selective", "Full"]:
                run_benchmark(
                    enc_seqlen, dec_seqlen, mbs, recompute_type=recompute_type
                )
