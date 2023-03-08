
#!/bin/bash

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=18230
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/root/Megatron-LM/cleaned_supervised_proportional_inputs_document
TARGETS_DATA_PATH=/root/Megatron-LM/cleaned_supervised_proportional_targets_document
CHECKPOINT_PATH=/root/Megatron-LM/checkpoints

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --use-env"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 4 \
       --encoder-num-layers 12 \
       --decoder-num-layers 12 \
       --hidden-size 1024 \
       --num-attention-heads 32 \
       --kv-channels 128 \
       --ffn-hidden-size 16384 \
       --encoder-seq-length 1024 \
       --decoder-seq-length 1024 \
       --micro-batch-size 1 \
       --global-batch-size 128 \
       --max-position-embeddings 8192 \
       --no-async-tensor-model-parallel-allreduce \
       --train-iters 255 \
       --train-epochs 1 \
       --lr-decay-iters 100 \
       --data-path $DATA_PATH \
       --targets-data-path $TARGETS_DATA_PATH \
       --vocab-file /root/t5-base-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 20 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 5 \
       --fp16  \
       --vocab-extra-ids 100 \
       --num-workers 2 \
       --pipeline-model-parallel-split-rank 2 \
       --dataloader-type ordered \
       --recompute-method uniform \
       --dynamic-batchsize \
       --tokens-per-global-batch 65536 \
       --pack-dataset \
       2>&1 | tee log_t5_mlm_linear_gbs65536_mbs1.txt
