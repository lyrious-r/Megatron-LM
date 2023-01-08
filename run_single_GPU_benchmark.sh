#!/bin/bash

EFF=$1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=8000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/root/Megatron-LM/cleaned_supervised_proportional_inputs_document
TARGETS_DATA_PATH=/root/Megatron-LM/cleaned_supervised_proportional_targets_document
CHECKPOINT_PATH=/root/Megatron-LM/checkpoints

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# the following model config is for T5-11B
# to benchmark different sequence length and batch sizes, change
# encoder-seq-length, decoder-seq-length, and micro-batch-size
# the benchmark will be run train-iters iterations
# with the first 10 iterations discarded for warmup
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 1 \
       --hidden-size 1024 \
       --num-attention-heads 128 \
       --kv-channels 128 \
       --ffn-hidden-size 65536 \
       --encoder-seq-length 4096 \
       --decoder-seq-length 4096 \
       --micro-batch-size 8 \
       --global-batch-size 128 \
       --max-position-embeddings 8192 \
       --train-iters 40 \
       --train-epochs 1 \
       --lr-decay-iters 100 \
       --vocab-file /root/t5-base-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 50 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 5 \
       --fp16  \
        --vocab-extra-ids 100 \
       --num-workers 2 \
       --dataloader-type ordered \
       --memory-model fixed \
       --preprocess-workers 512 \
       --tokens-per-global-batch 16384 \
       2>&1 | tee log_t5_microbenchmark_stats.txt