DATA_PATH="/root/Megatron-LM/datasets/cleaned_supervised_proportional_inputs_document"
TARGET_DATA_PATH="/root/Megatron-LM/datasets/cleaned_supervised_proportional_targets_document"
PER_DEVICE_BATCH_SIZE=2
GRAD_ACCUM_STEPS=1
GRADIENT_CHECKPOINTING=false

export CUDA_VISIBLE_DEVICES=0


# accelerate launch --config_file megatron_t5_config.yaml run_summarization.py \
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -c cudaProfilerApi --capture-range-end stop-shutdown -o t5_11b_2l_single_gpu_hf -f true \
deepspeed run_summarization.py \
    --deepspeed t5_deepspeed.json \
    --model_name_or_path t5-11b \
    --output_dir t5-11b-output \
    --data_path $DATA_PATH \
    --target_data_path $TARGET_DATA_PATH \
    --vocab_file /root/t5-base-vocab.txt \
    --global_batch_size 4096 \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --do_train true \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_train_epochs 1 \
    --max_steps 500 \
    --save_strategy no \
    --fp16 true \
    --dataloader_num_workers 2 \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --remove_unused_columns false \
    --enable_nsys_profile true
