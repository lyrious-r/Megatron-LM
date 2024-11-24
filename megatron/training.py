# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

from datetime import datetime
import math
import os
import sys
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_signal_handler
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.model import ModelType
from megatron.model import GPTModel
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.schedules import get_forward_backward_func
from megatron.utils import report_memory
from megatron.model.vision.knn_monitor import compute_feature_bank
from megatron.data.t5_dataset import T5UnsupervisedDataset
from megatron.utils import average_losses_across_data_parallel_group

from .pipeline_executor import get_pipeline_executor

from dynapipe.memory_opt.utils import reserve_full_memory
from dynapipe.pipe.instructions import ExecutionPlan

DEBUG_DUMP_MEMORY_STATS = os.getenv("DYNAPIPE_DEBUG_DUMP_MEMORY_STATS", 'False').lower() in ('true', '1', 't')
DEBUG_DUMP_MEMORY_PREFIX = os.environ.get('DYNAPIPE_DEBUG_DUMP_MEMORY_PREFIX', None)
if DEBUG_DUMP_MEMORY_STATS and not DEBUG_DUMP_MEMORY_PREFIX:
    raise ValueError("DYNAPIPE_DEBUG_DUMP_MEMORY_PREFIX must be set if DYNAPIPE_DEBUG_DUMP_MEMORY_STATS is set")

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()
    args = get_args()
    if DEBUG_DUMP_MEMORY_STATS and not args.dynapipe_custom_allocator:
        torch.cuda.memory._record_memory_history(True, trace_alloc_record_context=True, record_context_cpp=True)

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider, virtual_pp_rank=idx, n_virtual_pp_ranks=len(model))
            for idx in range(len(model))
        ]
        train_data_iterator = [data_iterators[0]
                               for data_iterators in all_data_iterators]
        valid_data_iterator = [data_iterators[1]
                               for data_iterators in all_data_iterators]
        test_data_iterator = [data_iterators[2]
                              for data_iterators in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)
    print_rank_0('training ...')

    if args.train_epochs is not None:
        print_rank_0('training for {} epochs.'.format(args.train_epochs))

    iteration = 0
    if args.do_train and args.train_iters > 0:
        if args.use_dynapipe:
            iteration = dynapipe_train(forward_step_func, model, optimizer,
                                   opt_param_scheduler, train_data_iterator)
        else:
            iteration = train(forward_step_func,
                            model, optimizer, opt_param_scheduler,
                            train_data_iterator, valid_data_iterator,
                            process_non_loss_data_func)
    print_datetime('after training is done')
    print("Training finished successfully.", flush=True)
    print("Taking poison pill...", flush=True)
    os.system("pkill -f 'pretrain_t5'")
    os.system("pkill -f 'pretrain_gpt'")

    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func,
                                   False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   0, process_non_loss_data_func,
                                   True)
    timers.log_all()

def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        # assert model_type != ModelType.encoder_and_decoder, \
        #     "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            if model_type == ModelType.encoder_and_decoder:
                pre_process = mpu.is_pipeline_first_stage(ignore_virtual=True) and (i == 0 or i == args.virtual_pipeline_model_parallel_size // 2)
                post_process = mpu.is_pipeline_last_stage(ignore_virtual=True) and (i == args.virtual_pipeline_model_parallel_size - 1)
                if i < args.virtual_pipeline_model_parallel_size // 2:
                    add_encoder = True
                    add_decoder = False
                else:
                    add_encoder = False
                    add_decoder = True
                this_model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    add_encoder=add_encoder,
                    add_decoder=add_decoder,
                )
            else:
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                this_model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # share embedding and positional embedding weights if using interleaved
    # schedule and T5
    if model_type == ModelType.encoder_and_decoder and \
         mpu.get_pipeline_model_parallel_world_size() > 1 and \
            args.virtual_pipeline_model_parallel_size is not None \
            and mpu.is_pipeline_first_stage(ignore_virtual=True):
        assert len(model) % 2 == 0, \
            "Number of encoder and decoder chunks should be even for interleaved schedule"
        first_encoder = model[0]
        first_decoder = model[len(model) // 2]
        assert hasattr(first_encoder.language_model, 'embedding'), \
            "First encoder layer should have embedding attribute"
        assert hasattr(first_decoder.language_model, 'embedding'), \
            "First decoder layer should have embedding attribute"
        first_decoder.language_model.embedding = first_encoder.language_model.embedding

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    args.allow_transformer_engine = all([type(m) == GPTModel for m in model])
    assert args.allow_transformer_engine or args.transformer_impl == 'local', \
        'Transformer Engine is only approved for GPT models'

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    if args.deepspeed:
        return model

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            i = torch.cuda.current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
                     for model_module in model]

        elif args.DDP_impl == 'local':
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp)
                     for model_module in model]
            # broad cast params from data parallel src rank to other data parallel ranks
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()
        else:
            raise NotImplementedError('Unknown DDP implementation specified: '
                                      '{}. Exiting.'.format(args.DDP_impl))

    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler)

    return opt_param_scheduler


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)
    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))

    optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                       scale_lr_cond, lr_mult)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.deepspeed:
        import deepspeed
        print_rank_0("DeepSpeed is enabled.")
        assert len(model) == 1, "Interleaved schedule currently do not work" \
                                "with DeepSpeed."

        # patch mpu to add get_model_parallel_world_size
        mpu.get_model_parallel_world_size = mpu.get_tensor_model_parallel_world_size
        mpu.get_model_parallel_rank = mpu.get_tensor_model_parallel_rank
        model, optimizer, _, opt_param_scheduler = deepspeed.initialize(
            model=model[0],
            optimizer=optimizer,
            args=args,
            lr_scheduler=opt_param_scheduler,
            mpu=mpu,
            dist_init_required=False
        )
        if args.use_dynapipe or mpu.get_pipeline_model_parallel_world_size() > 1:
            # we manually allreduce gradients when using dynapipe or during pp
            model.pipeline_parallelism = True
            model.enable_backward_allreduce = False
        model = [model]


    if args.load is not None:
        timers = get_timers()
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # deepspeed zeros gradients internally
    if not args.deepspeed:
        # Set grad to zero.
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            for partition in model:
                partition.zero_grad_buffer()
        optimizer.zero_grad()

    if args.dynamic_batchsize:
        # the output of data iterator is a list of microbatches,
        # we convert the list to another data iterator
        if isinstance(data_iterator, list):
            new_iterators = []
            for iterator in data_iterator:
                data = next(iterator)
                new_iterators.append(iter(data))
            microbatch_iterator = new_iterators
        else:
            data = next(data_iterator)
            microbatch_iterator = iter(data)
    else:
        microbatch_iterator = data_iterator

    # Forward pass.
    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    forward_backward_func = get_forward_backward_func()
    fwd_bwd_timers = timers if args.timing_log_level > 1 else None
    if args.deepspeed:
        model[0].set_gradient_accumulation_boundary(False)
    losses_reduced = forward_backward_func(
        forward_step_func, microbatch_iterator, model,
        optimizer, fwd_bwd_timers, forward_only=False)
    timers('forward-backward').stop()

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Reduce gradients.
    if not args.deepspeed:
        optimizer.reduce_model_grads(args, timers)

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0],
                                       (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    if not args.deepspeed:
        # Update parameters.
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
        timers('optimizer').stop()

        # Gather params.
        if update_successful:
            optimizer.gather_model_params(args, timers)

        # Vision momentum.
        if args.vision_pretraining and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0],
                                        (torchDDP, LocalDDP, Float16Module))
            unwrapped_model.update_momentum(args.curr_iteration)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1
    else:
        model[0].set_gradient_accumulation_boundary(True)
        if not model[0].enable_backward_allreduce or not optimizer.overlap_comm:
            model[0].allreduce_gradients()
        model[0].step()
        skipped_iter = 0
        grad_norm = 0
        num_zeros_in_grad = 0

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def dynapipe_train_step(data_iterator, forward_step_func,
                     model, optimizer, opt_param_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    if not args.deepspeed:
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            for partition in model:
                partition.zero_grad_buffer()
        optimizer.zero_grad()

    if DEBUG_DUMP_MEMORY_STATS:
        import pickle
        import json
        import os

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        dp_rank = mpu.get_data_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        dump_dir = os.path.join(DEBUG_DUMP_MEMORY_PREFIX, 'dr{}_pr{}_tr{}'.format(dp_rank, pp_rank, tp_rank))
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        torch.cuda.synchronize()
        if args.dynapipe_custom_allocator:
            from dynapipe.memory_opt.cuda_caching_allocator import get_allocator
            allocator = get_allocator()
            # bug, disable until fixed
            # pickled_snapshot = allocator.get_memory_snapshot()
            # snapshot = pickle.loads(pickled_snapshot)

            # get some stats
            params_size = sum([get_parameters_size(m) for m in model])
            optimizer_states_size = get_optimizer_state_size(optimizer)
            with open(os.path.join(dump_dir, f'stats_iter{args.curr_iteration}.txt'), 'w') as f:
                data= {
                    "params_size": params_size,
                    "optimizer_states_size": optimizer_states_size,
                    "peak_allocated_memory": allocator.peak_allocated_cuda_memory(),
                    "peak_reserved_memory": allocator.peak_reserved_cuda_memory(),
                    "peak_requested_memory": allocator.peak_requested_cuda_memory(),
                    "current_allocated_memory": allocator.current_allocated_cuda_memory(),
                    "current_reserved_memory": allocator.current_reserved_cuda_memory(),
                    "current_requested_memory": allocator.current_requested_cuda_memory(),
                }
                json.dump(data, f)
        else:
            snapshot = torch.cuda.memory._snapshot()

            with open(os.path.join(dump_dir, f'snapshot_iter{args.curr_iteration}.pickle'), 'wb') as f:
                pickle.dump(snapshot, f)
            # get some stats
            params_size = sum([get_parameters_size(m) for m in model])
            optimizer_states_size = get_optimizer_state_size(optimizer)
            with open(os.path.join(dump_dir, f'stats_iter{args.curr_iteration}.txt'), 'w') as f:
                data= {
                    "params_size": params_size,
                    "optimizer_states_size": optimizer_states_size,
                    "memory_stats": torch.cuda.memory_stats(),
                }
                json.dump(data, f)
        # reset peak
        if args.dynapipe_custom_allocator:
            allocator.reset_peak_stats()
            allocator.reset_accumulated_stats()
        else:
            torch.cuda.memory.reset_peak_memory_stats()
            torch.cuda.memory.reset_accumulated_memory_stats()

    # Forward pass.
    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    
    if isinstance(data_iterator, list):
        # interleaved schedule
        microbatch_iterator = []
        execution_plan = None
        for iterator in data_iterator:
            if iterator is not None:
                mb, ep = next(iterator)
                microbatch_iterator.append(iter(mb))
            else:
                microbatch_iterator.append(None)
                ep = None
            # execution plan should be the same for all iterators
            if execution_plan is None:
                execution_plan = ep
    else:
        if data_iterator is not None:
            microbatch, execution_plan = next(data_iterator)
            microbatch_iterator = iter(microbatch)
        else:
            microbatch_iterator = None
            execution_plan = None
    if args.tensor_model_parallel_size > 1:
        # broadcast execution plan across tp groups
        if mpu.get_tensor_model_parallel_rank() == 0:
            assert execution_plan is not None
            execution_plan: ExecutionPlan
            ep_bytes = execution_plan.serialize()
            size = torch.LongTensor([len(ep_bytes)]).cuda()
            torch.distributed.broadcast(size, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())
            ep_tensor = torch.frombuffer(np.fromstring(ep_bytes, dtype=np.uint8), dtype=torch.uint8).cuda()
            torch.distributed.broadcast(ep_tensor, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())
        else:
            size = torch.LongTensor([0]).cuda()
            torch.distributed.broadcast(size, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())
            ep_tensor = torch.zeros(size.cpu().item(), dtype=torch.uint8).cuda()
            torch.distributed.broadcast(ep_tensor, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())
            execution_plan = ExecutionPlan.deserialize(ep_tensor.cpu().numpy().tobytes())
    assert execution_plan is not None
    executor = get_pipeline_executor(forward_step_func, microbatch_iterator, model, optimizer)
    executor.execute(execution_plan, args.curr_iteration)
    losses_reduced = executor.forward_data_store
    timers('forward-backward').stop()

    loss_reduced = {}
    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        # Average loss across data parallel groups.
        for key in loss_reduced:
            reduced = average_losses_across_data_parallel_group([loss_reduced[key]])
            loss_reduced[key] = reduced[0]

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Reduce gradients.
    if not args.deepspeed:
        optimizer.reduce_model_grads(args, timers)

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0],
                                       (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    if not args.deepspeed:
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
        timers('optimizer').stop()

        # Gather params.
        if update_successful:
            optimizer.gather_model_params(args, timers)

        # Vision momentum.
        if args.vision_pretraining and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0],
                                        (torchDDP, LocalDDP, Float16Module))
            unwrapped_model.update_momentum(args.curr_iteration)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1
    else:
        model[0].set_gradient_accumulation_boundary(True)
        if not model[0].enable_backward_allreduce or not optimizer.overlap_comm:
            model[0].allreduce_gradients()
        model[0].step()
        skipped_iter = 0
        grad_norm = 0
        num_zeros_in_grad = 0

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad

def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'grads-all-reduce',
        'grads-reduce-scatter',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    timers('save-checkpoint').stop(barrier=True)
    timers.log(['save-checkpoint'])

def get_resident_tensors():
    import gc
    tensors = set()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensors.add(obj)
            elif (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensors.add(obj.data)
        except:
            pass
    return tensors

def get_resident_tensor_size():
    import gc
    total_size = 0
    gc.collect()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                total_size += obj.numel() * obj.element_size()
            elif (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                total_size += obj.data.numel() * obj.data.element_size()
        except:
            pass
    return total_size

def get_parameters_size(model):
    total_size = 0
    for p in model.parameters():
        total_size += p.numel() * p.element_size()
    return total_size

def _traverse_dict_or_lists(d_or_l):
    size = 0
    if torch.is_tensor(d_or_l):
        size += d_or_l.numel() * d_or_l.element_size()
    elif hasattr(d_or_l, 'data') and torch.is_tensor(d_or_l.data):
        size += d_or_l.data.numel() * d_or_l.data.element_size()
    elif isinstance(d_or_l, dict):
        for v in d_or_l.values():
            size += _traverse_dict_or_lists(v)
    elif isinstance(d_or_l, list):
        for v in d_or_l:
            size += _traverse_dict_or_lists(v)
    return size

def get_optimizer_state_size(optimizer):
    size = 0
    for params_or_state_dicts in optimizer.state_dict().values():
        size += _traverse_dict_or_lists(params_or_state_dicts)
    return size

def dynapipe_train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator):
    """Train the model function. Removed irrelavant code for testing."""
    args = get_args()
    timers = get_timers()
    from dynapipe.utils.logger import logger
    from dynapipe.pipe.data_loader import get_num_iters

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    orig_iteration = args.iteration
    iteration = args.iteration

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True if not args.dynapipe_custom_allocator else False
    rank = torch.distributed.get_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()

    # before training starts, launch an allreduce to init NCCL communicator
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    torch.distributed.all_reduce(torch.zeros(1).cuda(), group=mpu.get_model_parallel_group())

    if args.debug_dump_memory_trace:
        assert not DEBUG_DUMP_MEMORY_STATS, \
            "Cannot use both debug_dump_memory_trace and " \
            "debug_dump_memory_stats"
    while iteration < args.train_iters:
        if iteration == 1:
            if args.dynapipe_reserve_all_memory:
                # Reserve all GPU memory upfront.
                reserved_memory, memory_limit = reserve_full_memory()
                logger.info("Reserved memory: {:.2f} GB, "
                            "memory limit: {:.2f} GB".format(
                                reserved_memory / 1e9,
                                memory_limit / 1e9)
                            )
            # torch.cuda.memory._record_memory_history(True)
            # import pickle
            # def oom_observer(device, alloc, device_alloc, device_free):
            #     # snapshot right after an OOM happened
            #     print('saving allocated state during OOM')
            #     snapshot = torch.cuda.memory._snapshot()
            #     pickle.dump(snapshot, open(f'oom_snapshot_rank{rank}.pickle', 'wb'))
            # if not args.dynapipe_custom_allocator:
            #     torch._C._cuda_attach_out_of_memory_observer(oom_observer)
        if int(os.environ.get('LOCAL_RANK')) == 0:
            logger.info("Running iteration {}...".format(iteration))
        timers('iteration-time').start()
        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration
        n_iters = get_num_iters()
        if n_iters is not None and iteration >= n_iters:
            # run out of data
            break
        try:
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                dynapipe_train_step(
                        train_data_iterator,
                        forward_step_func,
                        model,
                        optimizer,
                        opt_param_scheduler)
        except StopIteration:
            # run out of data
            break
        iteration += 1
        if args.empty_unused_memory_interval > 0 and \
                iteration % args.empty_unused_memory_interval == 0:
            # Empty unused memory.
            logger.info("Emptying cuda cache...")
            torch.cuda.empty_cache()
        if args.profile_with_nsys:
            from dynapipe.utils.logger import logger
            if iteration - orig_iteration == args.nsys_profile_warmup:
                logger.warning("Cuda profiler started.")
                torch.cuda.cudart().cudaProfilerStart()
            if iteration - orig_iteration == args.nsys_profile_warmup + args.nsys_profile_steps:
                logger.warning("Cuda profiler stopped.")
                torch.cuda.cudart().cudaProfilerStop()
        if args.debug_dump_memory_trace:
            if iteration - orig_iteration == args.nsys_profile_warmup:
                logger.warning("Enabling debug_dump_memory_trace.")
                torch.cuda.memory._record_memory_history(True,
                    trace_alloc_max_entries=100000,
                    trace_alloc_record_context=True,)
            if iteration - orig_iteration == args.nsys_profile_warmup + args.nsys_profile_steps:
                import pickle
                if not os.path.exists('./memory_trace'):
                    os.makedirs('./memory_trace')
                with open(f'./memory_trace/dr{dp_rank}_pr{pp_rank}_tr{tp_rank}.pkl', 'wb') as f:
                    snapshot = torch.cuda.memory._snapshot()
                    pickle.dump(snapshot, f)
                torch.distributed.barrier()
                exit(1)
        args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()
        timers('iteration-time').stop()
        if args.per_iter_time_log_path is not None:
            with open(args.per_iter_time_log_path, 'a') as f:
                f.write(str(timers('iteration-time').elapsed()) + "\n")
        # Logging.
        if args.deepspeed:
            loss_scale = optimizer.cur_scale
        else:
            loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad)

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()
    return iteration

def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func):
    """Train the model function."""
    from dynapipe.utils.logger import logger
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    orig_iteration = args.iteration
    iteration = args.iteration
    
    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    rank = torch.distributed.get_rank()
    while iteration < args.train_iters:
        if int(os.environ.get('LOCAL_RANK')) == 0:
            logger.info("Running iteration {}...".format(iteration))
        timers('iteration-time').start()
        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler)
        iteration += 1
        if args.profile_with_nsys:
            from dynapipe.utils.logger import logger
            if iteration - orig_iteration == args.nsys_profile_warmup:
                logger.warning("Cuda profiler started.")
                torch.cuda.cudart().cudaProfilerStart()
            if iteration - orig_iteration == args.nsys_profile_warmup + args.nsys_profile_steps:
                logger.warning("Cuda profiler stopped.")
                torch.cuda.cudart().cudaProfilerStop()
        args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()
        timers('iteration-time').stop()
        if args.per_iter_time_log_path is not None:
            with open(args.per_iter_time_log_path, 'a') as f:
                f.write(str(timers('iteration-time').elapsed()) + "\n")
        # Logging.
        if args.deepspeed:
            loss_scale = optimizer.cur_scale
        else:
            loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       False)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
                print_datetime('exiting program after receiving SIGTERM.')
                sys.exit()

        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             opt_param_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()


    return iteration


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             verbose=False):
    """Evaluation."""
    args = get_args()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            forward_backward_func = get_forward_backward_func()
            loss_dicts = forward_backward_func(
                forward_step_func, data_iterator, model, optimizer=None,
                timers=None, forward_only=True)

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func, data_iterator, model, optimizer=None,
                timers=None, forward_only=True, collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict, collected_non_loss_data

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func,
                               verbose=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict, collected_non_loss_data = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider, virtual_pp_rank=0, n_virtual_pp_ranks=1):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each tensor model parallel group.
    # Data loader may also be needed on decoder first stage, so we duplicate
    # it across pipeline model ranks.
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                     args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                      eval_iters * args.global_batch_size,
                                      test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples, virtual_pp_rank=virtual_pp_rank, n_virtual_pp_ranks=n_virtual_pp_ranks, is_training=True)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples, is_training=False)
        test_dataloader = build_pretraining_data_loader(test_ds, 0, is_training=False)
        if isinstance(train_ds, T5UnsupervisedDataset):
            input_padding_eff, target_padding_eff = train_ds.get_padding_efficiency()
            print_rank_0(' > training set padding efficiency:')
            print_rank_0('    input:      {}'.format(input_padding_eff))
            print_rank_0('    target:     {}'.format(target_padding_eff))

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic', 'ordered']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type != 'cyclic' \
                            else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type != 'cyclic' \
                              else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type != 'cyclic' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
