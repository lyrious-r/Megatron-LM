# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Microbenchmark T5 layers on a single GPU"""
from functools import partial
import os
import pickle

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import (
    get_args,
    get_num_microbatches,
    get_timers,
    print_rank_0,
    update_num_microbatches,
)
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.core import mpu
from megatron.initialize import initialize_megatron, set_jit_fusion_options
from megatron.model import ModelType, T5Model
from megatron.optimizer import Float16OptimizerWithFloat16Params
from megatron.schedules import backward_step, forward_step
from megatron.training import setup_model_and_optimizer
from megatron.utils import average_losses_across_data_parallel_group

MEMORY_TRACE_DIR = "./microbench_memory_trace"
WARMUP_ITERATIONS = 20
TRACE_AT_ITER = WARMUP_ITERATIONS + 2
BENCHMARK_START_ITER = WARMUP_ITERATIONS + 5
timer_disabled = True
memory_trace_enabled = False
grad_hook_trigger_counts = {}


def start_timer(timers, name):
    if timer_disabled:
        return
    timers(name, log_level=0).start()


def stop_timer(timers, name):
    if timer_disabled:
        return
    timers(name, log_level=0).stop()


class StatRecorder:
    def __init__(self):
        self.records = {}

    def __call__(self, name):
        if name not in self.records:
            self.records[name] = []
        return self.records[name]

    def add(self, name, quantity):
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(quantity)

    def get(self, name, mean=True):
        if name in self.records:
            if mean:
                return sum(self.records[name]) / len(self.records[name])
            else:
                return self.records[name]
        else:
            return None


stat_recorder = StatRecorder()

### Hooks
# Benchmark will run multiple stages sequentially. To separate stages, we
# need to add hooks to the model to record the time and memory usage:
#              Stages                               Hooks
#                                 (start timer/memory trace for enc_embedding)
#   -> Embedding FW (enc tokens)      fw_hook("enc_embedding", "encoder")
#   -> Forward Encoder                fw_hook("encoder", "dec_embedding")
#   -> Embedding FW (dec tokens)      fw_hook("dec_embedding", "decoder")
#   -> Forward Decoder                 fw_hook("decoder", "postprocess")
#   -> Postprocess FW                    fw_hook("postprocess", None)
#                                  (start timer/memory trace for postprocess)
#   -> Postprocess BW                  grad_hook("postprocess", "decoder")
#   -> Backward Decoder               grad_hook("decoder", "encoder")
#   (there is no decoder backward embedding since grad is simply accumed)
#   -> Backward Encoder               grad_hook("encoder", "enc_embedding")
#   -> Embedding BW (enc tokens)
#                                   grad hook on embedding weights produce 
#                                   strange results. we manually stop the
#                                   timer and record the memory usage.

def get_fw_hook(stop_name, start_name):
    timers = get_timers()
    def fw_hook():
        stop_timer(timers, f"forward_{stop_name}")
        if memory_trace_enabled:
            name = get_microbenchmark_name()
            mem_trace_dir = os.path.join(MEMORY_TRACE_DIR, name)
            if not os.path.exists(mem_trace_dir):
                os.makedirs(mem_trace_dir)
            torch.cuda.synchronize()
            with open(os.path.join(mem_trace_dir, f"forward_{stop_name}.pkl"), 'wb') as f:
                snapshot = torch.cuda.memory._snapshot()
                pickle.dump(snapshot, f)
            # reset the memory history
            torch.cuda.memory._record_memory_history(True,
                trace_alloc_max_entries=100000,
                trace_alloc_record_context=True,)
        memory_after_fw = torch.cuda.memory_allocated()
        peak_memory_after_fw = torch.cuda.max_memory_allocated()
        stat_recorder.add(f"memory_after_{stop_name}", memory_after_fw / 1e6)
        stat_recorder.add(
            f"peak_memory_after_{stop_name}", peak_memory_after_fw / 1e6
        )
        if start_name is not None:
            start_timer(timers, f"forward_{start_name}")
    return fw_hook

def get_grad_hook(stop_name, start_name, n_triggers=1):
    timers = get_timers()
    def grad_hook(grad):
        key = (stop_name, start_name)
        if key not in grad_hook_trigger_counts:
            grad_hook_trigger_counts[key] = 1
        else:
            grad_hook_trigger_counts[key] += 1
        if grad_hook_trigger_counts[key] == n_triggers:
            stop_timer(timers, f"backward_{stop_name}")
            if memory_trace_enabled:
                name = get_microbenchmark_name()
                mem_trace_dir = os.path.join(MEMORY_TRACE_DIR, name)
                if not os.path.exists(mem_trace_dir):
                    os.makedirs(mem_trace_dir)
                torch.cuda.synchronize()
                with open(os.path.join(mem_trace_dir, f"backward_{stop_name}.pkl"), 'wb') as f:
                    snapshot = torch.cuda.memory._snapshot()
                    pickle.dump(snapshot, f)
                # reset the memory history
                torch.cuda.memory._record_memory_history(True,
                    trace_alloc_max_entries=100000,
                    trace_alloc_record_context=True,)
            if start_name is not None:
                start_timer(timers, f"backward_{start_name}")
        return grad
    return grad_hook

def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True
):
    """Build the model."""

    print_rank_0("building T5 model ...")
    model = T5Model(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        hooks = {
            "enc_embedding": get_fw_hook("enc_embedding", "encoder"),
            "encoder": get_fw_hook("encoder", "dec_embedding"),
            "dec_embedding": get_fw_hook("dec_embedding", "decoder"),
            "decoder": get_fw_hook("decoder", "postprocess"),
            # "postprocess": get_fw_hook("postprocess", None),
            "postprocess_grad": get_grad_hook("postprocess", "decoder"),
            "decoder_grad": get_grad_hook("decoder", "encoder", n_triggers=2),  # encoder output and decoder input
            "encoder_grad": get_grad_hook("encoder", "enc_embedding"),
        },
    )
    return model


def get_batch(data_iterator):
    """Build the batch."""
    assert data_iterator is not None
    microbatch_size, enc_sequence_length, dec_sequence_length = next(
        data_iterator
    )
    datatype = torch.int64
    # generate random data
    data_b = {
        "text_enc": torch.randint(
            0, 32000, (microbatch_size, enc_sequence_length), dtype=datatype
        ),
        "text_dec": torch.randint(
            0, 32000, (microbatch_size, dec_sequence_length), dtype=datatype
        ),
        "labels": torch.randint(
            0, 32000, (microbatch_size, dec_sequence_length), dtype=datatype
        ),
        "loss_mask": torch.ones(
            (microbatch_size, dec_sequence_length), dtype=datatype
        ),
        "enc_mask": torch.ones(
            (microbatch_size, enc_sequence_length, enc_sequence_length),
            dtype=datatype,
        ),
        "dec_mask": torch.ones(
            (microbatch_size, dec_sequence_length, dec_sequence_length),
            dtype=datatype,
        ),
        "enc_dec_mask": torch.ones(
            (microbatch_size, dec_sequence_length, enc_sequence_length),
            dtype=datatype,
        ),
    }
    for k, v in data_b.items():
        data_b[k] = v.cuda()

    # Unpack.
    tokens_enc = data_b["text_enc"].long()
    tokens_dec = data_b["text_dec"].long()
    labels = data_b["labels"].long()
    loss_mask = data_b["loss_mask"].float()

    enc_mask = data_b["enc_mask"] < 0.5
    dec_mask = data_b["dec_mask"] < 0.5
    enc_dec_mask = data_b["enc_dec_mask"] < 0.5

    return (
        tokens_enc,
        tokens_dec,
        loss_mask,
        labels,
        enc_mask,
        dec_mask,
        enc_dec_mask,
    )


def loss_func(loss_mask, output_tensor):
    lm_loss_ = output_tensor.float()
    lm_loss = (
        torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
    )

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])

    return loss, {"lm loss": averaged_losses[0]}


def forward_step_func(data_iterator, model):
    """Forward step."""
    # here data_iterator contains sequences of (microbatch_size, enc_seqlen, dec_seqlen)

    # Get the batch.
    with torch.cuda.nvtx.range("batch_generator"):
        (
            tokens_enc,
            tokens_dec,
            loss_mask,
            lm_labels,
            enc_mask,
            dec_mask,
            enc_dec_mask,
        ) = get_batch(data_iterator)

    # Forward model lm_labels
    output_tensor = model(
        tokens_enc,
        tokens_dec,
        enc_mask,
        dec_mask,
        enc_dec_mask,
        tokentype_ids=None,
        lm_labels=lm_labels,
    )

    return output_tensor, partial(loss_func, loss_mask)


def train_shape_provider():
    args = get_args()

    def train_shape_iterator():
        while True:
            yield args.micro_batch_size, args.encoder_seq_length, args.decoder_seq_length

    return train_shape_iterator()


def benchmark_forward_backward_no_pipelining(
    iteration,
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    timers,
    forward_only,
    collect_non_loss_data=False,
    **kwargs,
):

    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses."""
    assert len(model) == 1
    model = model[0]

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None

    memory_before_forward = torch.cuda.memory_allocated()
    peak_memory_before_forward = torch.cuda.max_memory_allocated()
    stat_recorder.add("memory_before_forward", memory_before_forward / 1e6)
    stat_recorder.add(
        "peak_memory_before_forward", peak_memory_before_forward / 1e6
    )
    start_timer(timers, "forward_total")
    start_timer(timers, "forward_enc_embedding")
    global memory_trace_enabled
    if iteration == TRACE_AT_ITER:
        memory_trace_enabled = True
        torch.cuda.memory._record_memory_history(True,
            trace_alloc_max_entries=100000,
            trace_alloc_record_context=True,)
    else:
        memory_trace_enabled = False
    output_tensor = forward_step(
        forward_step_func,
        data_iterator,
        model,
        input_tensor,
        forward_data_store,
        timers,
        collect_non_loss_data,
    )
    get_fw_hook("postprocess", None)()
    stop_timer(timers, "forward_total")
    memory_after_forward = torch.cuda.memory_allocated()
    peak_memory_after_forward = torch.cuda.max_memory_allocated()
    stat_recorder.add("memory_after_dec_forward", memory_after_forward / 1e6)
    stat_recorder.add(
        "peak_memory_after_dec_forward", peak_memory_after_forward / 1e6
    )
    if not forward_only:
        start_timer(timers, "backward_total")
        start_timer(timers, "backward_postprocess")
        if iteration == TRACE_AT_ITER:
            torch.cuda.memory._record_memory_history(True,
                trace_alloc_max_entries=100000,
                trace_alloc_record_context=True,)
        # backward_decoder stop is called in the gradient hook
        backward_step(
            optimizer, input_tensor, output_tensor, output_tensor_grad, timers
        )
        if memory_trace_enabled:
            name = get_microbenchmark_name()
            mem_trace_dir = os.path.join(MEMORY_TRACE_DIR, name)
            if not os.path.exists(mem_trace_dir):
                os.makedirs(mem_trace_dir)
            torch.cuda.synchronize()
            with open(os.path.join(mem_trace_dir, f"backward_enc_embedding.pkl"), 'wb') as f:
                snapshot = torch.cuda.memory._snapshot()
                pickle.dump(snapshot, f)
            # reset
            torch.cuda.memory._record_memory_history(True,
                    trace_alloc_max_entries=100000,
                    trace_alloc_record_context=True,)
        stop_timer(timers, f"backward_enc_embedding")
        stop_timer(timers, "backward_total")
        memory_after_backward = torch.cuda.memory_allocated()
        peak_memory_after_backward = torch.cuda.max_memory_allocated()
        stat_recorder.add("memory_after_backward", memory_after_backward / 1e6)
        stat_recorder.add(
            "peak_memory_after_backward", peak_memory_after_backward / 1e6
        )
        if memory_trace_enabled:
            torch.cuda.memory._record_memory_history(False)
            memory_trace_enabled = False

    return forward_data_store


def benchmark_train_step(
    forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, iteration
):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    if args.DDP_impl == "local" and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    # Forward and Backward pass.
    losses_reduced = benchmark_forward_backward_no_pipelining(
        iteration,
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        timers,
        forward_only=False,
    )

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Reduce gradients.
    optimizer.reduce_model_grads(args, timers)

    # Update parameters.
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(
        args, timers
    )

    # Gather params.
    if update_successful:
        timers("backward-gather-model-params").start()
        optimizer.gather_model_params(args, timers)
        timers("backward-gather-model-params").stop()

    # Update learning rate.
    if update_successful:
        increment = (
            get_num_microbatches()
            * args.micro_batch_size
            * args.data_parallel_size
        )
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(
                losses_reduced_for_key
            )
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def benchmark_train(
    forward_step_func,
    model,
    optimizer,
    opt_param_scheduler,
    benchmark_shape_iterator,
):
    """Train the model function."""
    global timer_disabled
    global grad_hook_trigger_counts
    args = get_args()
    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Iterations.
    iteration = args.iteration
    assert (
        args.train_iters >= WARMUP_ITERATIONS
    ), "train_iters must be greater than or equal to {} for benchmarking".format(
        WARMUP_ITERATIONS
    )

    while iteration < args.train_iters:
        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration
        # reset memory counter so we capture peak memory per iter
        torch.cuda.reset_peak_memory_stats()
        (
            loss_dict,
            skipped_iter,
            grad_norm,
            num_zeros_in_grad,
        ) = benchmark_train_step(
            forward_step_func,
            benchmark_shape_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            iteration,
        )
        iteration += 1
        args.consumed_train_samples += (
            mpu.get_data_parallel_world_size()
            * args.micro_batch_size
            * get_num_microbatches()
        )
        if iteration >= WARMUP_ITERATIONS:
            timer_disabled = False
        grad_hook_trigger_counts = {}

    return iteration


def get_optimizer_state_size(optimizer):
    """Get the size of the stored optimizer states."""
    state_size = 0
    for per_tensor_states in optimizer.state_dict()["optimizer"][
        "state"
    ].values():
        for state_val in per_tensor_states.values():
            if isinstance(state_val, torch.Tensor):
                state_size += state_val.numel() * state_val.element_size()
    # we should also count the additional copy of model parameters in FP32
    if isinstance(optimizer, Float16OptimizerWithFloat16Params):
        for param_group in optimizer.fp32_from_float16_groups:
            for p in param_group:
                state_size += p.numel() * p.element_size()
    return state_size


def get_microbenchmark_name():
    args = get_args()
    name = "hs{}_ah{}_kv{}_ffhs{}_encsl{}_decsl{}_mbs{}".format(
        args.hidden_size,
        args.num_attention_heads,
        args.kv_channels,
        args.ffn_hidden_size,
        args.encoder_seq_length,
        args.decoder_seq_length,
        args.micro_batch_size,
    )
    # add recomputation settings if exist
    if args.recompute_granularity:
        name += "_rc_{}".format(args.recompute_granularity)
        if args.recompute_granularity == "full":
            name += "_{}".format(args.recompute_method)
    return name


def generate_report(n_iters, save_path=None):
    timers = get_timers()

    f = None
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        f = open(save_path, "w")
        # write basic data of the model into the file
        f.write("# " + get_microbenchmark_name() + "\n")

    def _rprint(s, padding="="):
        print((s + " ").ljust(80, padding))
        if f is not None:
            f.write(s.strip() + "\n")

    def _cprint(s, filler="="):
        format_str = "{{:{}^80}}".format(filler)
        print(format_str.format(" " + s + " "))

    _cprint("")
    # memory summary
    _cprint("Memory Summary")
    _cprint("Model States", "-")

    def _get_stats_and_print(attr_name):
        val = stat_recorder.get(attr_name)
        if val:
            _rprint("    {}: {:.2f} MB".format(attr_name, val), " ")

    def _get_stats_and_print_difference(attr_name1, attr_name2, new_name):
        val = stat_recorder.get(attr_name1) - stat_recorder.get(attr_name2)
        if val:
            _rprint("    {}: {:.2f} MB".format(new_name, val), " ")

    def _get_time_and_print(attr_name):
        val = timers(attr_name).elapsed(reset=False)
        if val:
            _rprint(
                "    {}: {:.2f} ms".format(attr_name, val / n_iters * 1000),
                " ",
            )

    def _get_time_and_print_difference(attr_name1, attr_name2, new_name):
        val = timers(attr_name1).elapsed(reset=False) - timers(
            attr_name2
        ).elapsed(reset=False)
        if val:
            _rprint(
                "    {}: {:.2f} ms".format(new_name, val / n_iters * 1000), " "
            )

    _get_stats_and_print("model_embedding_param_size")
    _get_stats_and_print("model_encoder_param_size")
    _get_stats_and_print("model_decoder_param_size")
    _get_stats_and_print("model_pooler_param_size")
    _get_stats_and_print("optimizer_state_size")
    _cprint("Activations ", "-")
    _get_stats_and_print("memory_before_forward")
    # _get_stats_and_print("memory_after_enc_forward")
    # _get_stats_and_print("memory_after_dec_forward")
    _get_stats_and_print("memory_after_backward")
    _get_stats_and_print_difference(
        "memory_after_enc_embedding", "memory_before_forward", "enc_embedding_activation"
    )
    _get_stats_and_print_difference(
        "memory_after_encoder", "memory_after_enc_embedding", "encoder_activation"
    )
    _get_stats_and_print_difference(
        "memory_after_dec_embedding", "memory_after_encoder", "dec_embedding_encoder_activation"
    )
    _get_stats_and_print_difference(
        "memory_after_decoder", "memory_after_dec_embedding", "decoder_activation"
    )
    _get_stats_and_print_difference(
        "memory_after_postprocess", "memory_after_decoder", "postprocess_activation"
    )
    _get_stats_and_print("peak_memory_before_forward")
    _get_stats_and_print("peak_memory_after_backward")
    _cprint("")
    # execution time summary
    _cprint("Execution Time Summary")
    _get_time_and_print("forward_total")
    _get_time_and_print("forward_enc_embedding")
    _get_time_and_print("forward_encoder")
    _get_time_and_print("forward_dec_embedding")
    _get_time_and_print("forward_decoder")
    _get_time_and_print("forward_postprocess")
    _get_time_and_print("backward_total")
    _get_time_and_print("backward_postprocess")
    _get_time_and_print("backward_decoder")
    _get_time_and_print("backward_encoder")
    _get_time_and_print("backward_enc_embedding")
    if f is not None:
        f.close()


def microbenchmark(
    benchmark_shape_provider,
    model_provider,
    model_type,
    forward_step_func,
    extra_args_provider=None,
    args_defaults={},
):
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
    initialize_megatron(
        extra_args_provider=extra_args_provider, args_defaults=args_defaults
    )
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    args = get_args()

    assert args.microbenchmark_save_dir is not None, (
        "Please specify a directory to save microbenchmark results"
    )

    # Model, optimizer, and learning rate.
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type
    )
    if isinstance(model, list):
        assert len(model) == 1
        unwrapped_model = unwrap_model(model[0], (torchDDP, LocalDDP, Float16Module))
    else:
        unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    if isinstance(unwrapped_model, T5Model):
        unwrapped_model = unwrapped_model.language_model

    # print model parameters and optimizer states in MB
    def _get_param_size(params):
        return sum(p.numel() * p.element_size() for p in params) / 1e6

    if hasattr(unwrapped_model, "embedding"):
        model_embedding_param_size = _get_param_size(
            unwrapped_model.embedding.parameters()
        )
        stat_recorder.add(
            "model_embedding_param_size", model_embedding_param_size
        )
    if hasattr(unwrapped_model, "encoder"):
        model_encoder_param_size = _get_param_size(unwrapped_model.encoder.parameters())
        stat_recorder.add("model_encoder_param_size", model_encoder_param_size)
    if hasattr(unwrapped_model, "decoder"):
        model_decoder_param_size = _get_param_size(unwrapped_model.decoder.parameters())
        stat_recorder.add("model_decoder_param_size", model_decoder_param_size)
    if hasattr(unwrapped_model, "pooler"):
        model_pooler_param_size = _get_param_size(unwrapped_model.pooler.parameters())
        stat_recorder.add("model_pooler_param_size", model_pooler_param_size)

    iteration = 0
    iteration = benchmark_train(
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        benchmark_shape_provider(),
    )
    # optimizer state only exists after the first iteration
    # its quite troublesome to get parameter state for each param
    # so we just get the total size of the optimizer state and
    # validate it against formula
    optimizer_state_size = get_optimizer_state_size(optimizer) / 1e6
    stat_recorder.add("optimizer_state_size", optimizer_state_size)

    # generate report
    microbenchmark_save_path = (
        os.path.join(args.microbenchmark_save_dir, f"microbench_{get_microbenchmark_name()}.txt")
    )
    generate_report(iteration, microbenchmark_save_path)


if __name__ == "__main__":
    microbenchmark(
        train_shape_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step_func,
        args_defaults={"tokenizer_type": "BertWordPieceLowerCase"},
    )
