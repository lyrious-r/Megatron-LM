#!/bin/bash

for tokens in 16384 32768 65536
do
    for seqlen in 512 1024 2048 4096
    do
        python3 run_experiment.py --experiment_name t5_11b_8l_plopt_small --tokens_per_global_batch $tokens --encoder_seq_length $seqlen --decoder_seq_length $seqlen --plopt_dump_stats true
        echo "Sleeping for 5 seconds before killing processes"
        sleep 5
        pkill -f "pretrain_t5"
        pkill -f "redis-server"
        echo "Sleeping for 2 seconds"
        sleep 2
    done
done