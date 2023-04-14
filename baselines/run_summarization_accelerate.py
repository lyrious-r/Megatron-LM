#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

from dataclasses import dataclass
import logging
import argparse
import os
import sys
import math
from tqdm import tqdm

import torch
import datasets
import transformers
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    TrainerCallback,
    SchedulerType,
    get_scheduler,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.utils import MegatronLMDummyScheduler


from dataset_from_mlm import get_train_ds, get_tokenizer, DataCollatorForPackedDataset

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to the input dataset to use.",
    )
    parser.add_argument(
        "--target_data_path",
        type=str,
        help="The path to the target dataset to use.",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="The path to the vocab file to use.",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        required=True,
        help="The global batch size to use.",
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--enable_nsys_profile",
        type=bool,
        default=False,
        help="Whether to enable nsys profiling."
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument(
        "--resize_position_embeddings",
        type=bool,
        default=True,
        help="Resize position embeddings to the maximum sequence length of the model.",
    )
    args = parser.parse_args()

    # Sanity checks
    if (
        args.data_path is None
        and args.vocab_file is None
        and args.global_batch_size is None
        and args.max_source_length is None
    ):
        raise ValueError("Need either data_path, vocab_file, global_batch_size or max_source_length.")

    return args

class ProfilingCallback(TrainerCallback):
    def __init__(self, start_iteration=20, stop_iteration=40):
        self.start_iteration = start_iteration
        self.stop_iteration = stop_iteration
        
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == self.start_iteration:
            torch.cuda.cudart().cudaProfilerStart()
            
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.stop_iteration:
            torch.cuda.cudart().cudaProfilerStop()

@dataclass
class DummyMicroBatchSizeSetter():
    batch_size: int

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.enable_nsys_profile:
        logger.warning("Enabling nsys profiling.")

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {accelerator.process_index}, device: {accelerator.device}, n_gpu: {accelerator.num_processes}"
        + f"distributed training: {bool(accelerator.num_processes > 1)}, 16-bits training: {bool(accelerator.mixed_precision == 'fp16')}"
    )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
    )
    config.num_layers = 2
    config.num_decoder_layers = 2
    config.n_positions = max(args.max_source_length, args.max_target_length)
    config.use_cache = False

    model = AutoModelForSeq2SeqLM.from_config(config)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < args.max_source_length
    ):
        if args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {args.max_source_length}."
            )
            model.resize_position_embeddings(args.max_source_length)
        elif args.resize_position_embeddings:
            model.resize_position_embeddings(args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # load dataset
    train_dataset = get_train_ds(
        data_path=args.data_path,
        vocab_file=args.vocab_file,
        targets_data_path=args.target_data_path,
        train_epochs=int(args.num_train_epochs),
        train_iters=args.max_train_steps,
        global_batch_size=args.global_batch_size,
        encoder_seq_length=args.max_source_length,
        decoder_seq_length=args.max_target_length,
        pack_dataset=True,
    )
    tokenizer = get_tokenizer()
    # Data collator
    data_collator = DataCollatorForPackedDataset(
        expected_length=args.max_source_length,
        expected_length_target=args.max_target_length,
        pad_token_id=tokenizer.pad,
        model=model,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )
    # train_dataloader = get_train_dataloader(train_dataset,
    #                                         args.per_device_train_batch_size,
    #                                         args.max_source_length,
    #                                         accelerator.process_index,
    #                                         accelerator.num_processes,
    #                                         global_batch_size=args.global_batch_size,
    #                                         decoder_seq_length=args.max_target_length,
    #                                         num_workers=args.preprocessing_num_workers,
    #                                         tokenizer=tokenizer)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )
    lr_scheduler = MegatronLMDummyScheduler(
        optimizer=optimizer,
        total_num_steps=args.max_train_steps,
        warmup_num_steps=args.num_warmup_steps,
    )

    # Prepare everything with our `accelerator`.
    mbs_setter = DummyMicroBatchSizeSetter(args.per_device_train_batch_size)
    model, optimizer, train_dataloader, lr_scheduler, _ = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, mbs_setter
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Training
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if args.enable_nsys_profile:
                if step == 20:
                    torch.cuda.cudart().cudaProfilerStart()
                if step == 40:
                    torch.cuda.cudart().cudaProfilerStop()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            torch.cuda.synchronize()
            if completed_steps >= args.max_train_steps:
                break

    return


if __name__ == "__main__":
    main()
