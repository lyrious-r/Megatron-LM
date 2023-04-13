DATA_PATH="/root/Megatron-LM/datasets/cleaned_supervised_proportional_inputs_document"
TARGET_DATA_PATH="/root/Megatron-LM/datasets/cleaned_supervised_proportional_targets_document"
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=8

deepspeed run_summarization.py \
    --deepspeed t5_deepspeed.json \
    --model_name_or_path t5-11b \
    --output_dir t5-11b-output \
    --data_path $DATA_PATH \
    --target_data_path $TARGET_DATA_PATH \
    --vocab_file /root/t5-base-vocab.txt \
    --global_batch_size 65536 \
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
    --remove_unused_columns false
