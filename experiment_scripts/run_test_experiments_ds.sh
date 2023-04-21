#!/bin/bash

for tokens in 16384 32768 65536
do
    for seqlen in 512 1024
    do
        for micro_batch_size in 8 4 2 1
        do
            for recompute_level in 'none' 'selective' 'full'
            do
                for deepspeed_zero_stage in 0 2
                do
                    python3 run_experiment.py \
                        --experiment_name t5_11b_8l_deepspeed_small \
                        --tokens_per_global_batch $tokens \
                        --encoder_seq_length $seqlen \
                        --decoder_seq_length $seqlen \
                        --micro_batch_size $micro_batch_size \
                        --recompute_level $recompute_level \
                        --deepspeed_zero_stage $deepspeed_zero_stage
                done
            done
        done
    done
done