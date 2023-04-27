#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# for tokens in 16384 32768 65536
for tokens in 16384
do
    # for seqlen in 512 1024 2048 4096
    for seqlen in 1024
    do
        python3 run_experiment.py --model_type gpt --experiment_name gpt_6.7b_16l_tp2_pp2_plopt_small --tokens_per_global_batch $tokens --seq_length $seqlen
        echo "Sleeping for 5 seconds before killing processes"
        sleep 5
        pkill -f "pretrain_gpt"
        pkill -f "pretrain_t5"
        pkill -f "redis-server"
        echo "Sleeping for 2 seconds"
        sleep 2
    done
done