# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
import torch
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_num_microbatches
from megatron import get_timers
from megatron import p2p_communication
from megatron.core import mpu
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model import ModelType

DEBUG_PRINT = False

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_compute(*args, **kwargs):
    if DEBUG_PRINT:
        print(bcolors.OKBLUE + ' '.join([str(a) for a in args]) + bcolors.ENDC, **kwargs)

def print_comm_start(*args, **kwargs):
    if DEBUG_PRINT:
        print(bcolors.OKCYAN + ' '.join([str(a) for a in args]) + bcolors.ENDC, **kwargs)

def print_comm_end(*args, **kwargs):
    if DEBUG_PRINT:
        print(bcolors.OKGREEN + ' '.join([str(a) for a in args]) + bcolors.ENDC, **kwargs)

def get_forward_backward_func():
    args = get_args()
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            assert get_num_microbatches() % \
                args.pipeline_model_parallel_size == 0, \
                'number of microbatches (%d) is not divisible by pipeline-' \
                'model-parallel-size (%d) when using interleaved schedule' % (
                    get_num_microbatches(),
                    args.pipeline_model_parallel_size,
                )
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func

def deallocate_output_tensor(out):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if out is None:
        return
    assert isinstance(out, torch.Tensor), \
        "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, \
        "counter-productive to free a view of another tensor."
    if not out.numel() == 1:
        out.data = torch.empty(
            (1,),
            device = out.device,
            dtype = out.dtype,
        )
        
def custom_backward(output, grad_output, deepspeed_model=None):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    if deepspeed_model is not None:
        deepspeed_model.backward(output, grad_output)
        return

    assert output.numel() == 1, \
        "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), \
        "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), \
        "grad_output == '%s'." % type(grad_output).__name__

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format = torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors = (output,),
        grad_tensors = (grad_output,),
        keep_graph = False,
        create_graph = False,
        inputs = tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )
        

def forward_step(forward_step_func,
                 data_iterator,
                 model,
                 input_tensor,
                 forward_data_store,
                 timers,
                 collect_non_loss_data=False,
                 recompute_policy=None):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    args = get_args()

    if timers is not None:
        timers('forward-compute', log_level=2).start()
    if args.deepspeed:
        from deepspeed.runtime.engine import DeepSpeedEngine
        unwrap_classes = (torchDDP, LocalDDP, Float16Module, DeepSpeedEngine)
    else:
        unwrap_classes = (torchDDP, LocalDDP, Float16Module)
    unwrapped_model = unwrap_model(
        model, unwrap_classes)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor, loss_func = forward_step_func(data_iterator, model,recompute_policy=recompute_policy)
    if mpu.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / get_num_microbatches()
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if timers is not None:
        timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    if args.model_type == ModelType.encoder_and_decoder:
        if args.virtual_pipeline_model_parallel_size is not None:
            if mpu.get_virtual_pipeline_model_parallel_rank() >= \
                    args.virtual_pipeline_model_parallel_size // 2:
                return [output_tensor, input_tensor[-1]]
        else:
            if mpu.is_pipeline_stage_after_split():
                return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(optimizer, input_tensor, output_tensor,
                  output_tensor_grad, timers, ds_model=None):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    args = get_args()

    if timers is not None:
        timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None:
        # last stage, output_tensor is loss
        if args.deepspeed:
            assert ds_model is not None
            if not ds_model.pipeline_parallelism and ds_model.enable_backward_allreduce:
                # use deepspeed backward
                ds_model.backward(output_tensor[0])
            else:
                # manually scale loss and use custom backward
                output_tensor = [optimizer.loss_scaler.loss_scale * output_tensor[0]]
                custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            output_tensor = optimizer.scale_loss(output_tensor[0])
            custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
            args.model_type == ModelType.encoder_and_decoder:
        if args.virtual_pipeline_model_parallel_size is not None:
            if mpu.get_virtual_pipeline_model_parallel_rank() >= \
                    args.virtual_pipeline_model_parallel_size // 2:
                if output_tensor_grad[1] is not None:
                    input_tensor_grad[-1].add_(output_tensor_grad[1])
        else:
            if mpu.is_pipeline_stage_after_split():
                if output_tensor_grad[1] is not None:
                    input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if timers is not None:
        timers('backward-compute').stop()

    return input_tensor_grad


@contextmanager
def dummy_handler():
    try:
        yield
    finally:
        pass


def forward_backward_no_pipelining(forward_step_func,
                                   data_iterator, model,
                                   optimizer,
                                   timers,
                                   forward_only,
                                   collect_non_loss_data=False):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses."""
    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            with torch.cuda.nvtx.range("Forward Microbatch {}".format(i)):
                output_tensor = forward_step(forward_step_func, data_iterator,
                                            model, input_tensor, forward_data_store,
                                            timers, collect_non_loss_data)
            if not forward_only:
                with torch.cuda.nvtx.range("Backward Microbatch {}".format(i)):
                    backward_step(optimizer, input_tensor, output_tensor,
                                output_tensor_grad, timers, ds_model=model)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    with torch.cuda.nvtx.range("Forward Microbatch {}".format(get_num_microbatches() - 1)):
        output_tensor = forward_step(forward_step_func, data_iterator,
                                    model, input_tensor, forward_data_store,
                                    timers, collect_non_loss_data)
    if not forward_only:
        with torch.cuda.nvtx.range("Backward Microbatch {}".format(get_num_microbatches() - 1)):
            backward_step(optimizer, input_tensor, output_tensor,
                        output_tensor_grad, timers, ds_model=model)

    return forward_data_store


def get_recv_shapes_interleaved(chunk_id, model_type, is_prev):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    args = get_args()
    tensor_shapes = []

    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length

    if model_type == ModelType.encoder_and_decoder:
        if args.sequence_parallel:
            decoder_seq_length = args.decoder_seq_length // mpu.get_tensor_model_parallel_world_size()
        else:
            decoder_seq_length = args.decoder_seq_length

        rank = mpu.get_pipeline_model_parallel_rank()
        if chunk_id < args.virtual_pipeline_model_parallel_size // 2:
            tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
        elif chunk_id == args.virtual_pipeline_model_parallel_size // 2 and rank == 0 and is_prev:
            # first decoder layer, only receives encoder output
            tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, args.micro_batch_size, args.hidden_size))
            tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
    else:
        tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
    return tensor_shapes

def recv_forward(tensor_shapes, timers):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape,
                                                                timers=timers))
    return input_tensors


def recv_backward(tensor_shapes, timers):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape,
                                                                       timers=timers))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, tensor_shape, timers=timers)


def send_backward(input_tensor_grads, tensor_shapes, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, tensor_shape, timers=timers)


def send_forward_recv_backward(output_tensors, tensor_shapes, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, tensor_shape, timers=timers)
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
                input_tensor_grad, tensor_shape, timers=timers)
        input_tensors.append(input_tensor)
    return input_tensors

def _tensor_shape_to_str(tensor):
    if not isinstance(tensor, list):
        tensor = [tensor]
    shapes = []
    for t in tensor:
        if t is None:
            shapes.append('None')
        else:
            shapes.append(tuple(t.shape))
    return shapes

def forward_backward_pipelining_with_interleaving(forward_step_func,
                                                  data_iterator, model,
                                                  optimizer,
                                                  timers,
                                                  forward_only, 
                                                  collect_non_loss_data=False):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    args = get_args()

    if args.deepspeed:
        from deepspeed.runtime.engine import DeepSpeedEngine
        unwrap_classes = (torchDDP, LocalDDP, Float16Module, DeepSpeedEngine)
    else:
        unwrap_classes = (torchDDP, LocalDDP, Float16Module)
    unwrapped_model = unwrap_model(
        model[0], unwrap_classes)
    model_type = unwrapped_model.model_type

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = mpu.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = mpu.get_pipeline_model_parallel_rank()

    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    tensor_shape = (seq_length, args.micro_batch_size, args.hidden_size)
    
    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if get_num_microbatches() == pipeline_parallel_size:
            num_warmup_microbatches = num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = \
                (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (
                num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches,
                                          num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    rank = mpu.get_pipeline_model_parallel_rank()

    def _annotate_microbatch(microbatch_id, chunk_id, is_forward=True):
        if is_forward:
            return torch.cuda.nvtx.range("Forward Microbatch {} Chunk {}".format(microbatch_id, chunk_id))
        else:
            return torch.cuda.nvtx.range("Backward Microbatch {} Chunk {}".format(microbatch_id, chunk_id))

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # forward step
        if mpu.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        print_compute("Rank {}: FW - running microbatch {} on model chunk {}.".format(rank, microbatch_id, model_chunk_id))
        with _annotate_microbatch(microbatch_id, model_chunk_id, is_forward=True):
            output_tensor = forward_step(forward_step_func,
                                        data_iterator[model_chunk_id],
                                        model[model_chunk_id],
                                        input_tensor, 
                                        forward_data_store,
                                        timers,
                                        collect_non_loss_data)
        if not isinstance(output_tensor, list):
            output_tensor = [output_tensor]
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if mpu.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                if model_chunk_id >= args.virtual_pipeline_model_parallel_size // 2:
                    print_compute("Rank {}, appending [None, None] to output_tensor_grads for model chunk {}.".format(rank, model_chunk_id))
                    output_tensor_grads[model_chunk_id].append([None, None])
                else:
                    print_compute("Rank {}, appending None to output_tensor_grads for model chunk {}.".format(rank, model_chunk_id))
                    output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        print_compute("Rank {}: BW - running microbatch {} on model chunk {}, out tensor shape: {}, out grad shape: {}.".format(rank, microbatch_id, model_chunk_id,
                                                                                                                                _tensor_shape_to_str(output_tensor),
                                                                                                                                _tensor_shape_to_str(output_tensor_grad)))
        with _annotate_microbatch(microbatch_id, model_chunk_id, is_forward=False):
            input_tensor_grad = \
                backward_step(optimizer,
                            input_tensor,
                            output_tensor,
                            output_tensor_grad,
                            timers,
                            ds_model=model[model_chunk_id])
        if not isinstance(input_tensor_grad, list):
            input_tensor_grad = [input_tensor_grad]
        return input_tensor_grad

    # Run warmup forward passes.
    mpu.set_virtual_pipeline_model_parallel_rank(0)
    tensor_shapes = get_recv_shapes_interleaved(0, model_type, True)
    print_comm_start("Rank {}: Before warmup, recv forward: {} started".format(rank, tensor_shapes), flush=True)
    input_tensors[0].append(recv_forward(tensor_shapes, timers=timers))
    print_comm_end("Rank {}: Before warmup, recv forward: {} ended".format(rank, tensor_shapes), flush=True)
    for k in range(num_warmup_microbatches):
        output_tensor = forward_step_helper(k)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False
        recv_prev_shapes = get_recv_shapes_interleaved(next_forward_model_chunk_id, model_type, True)
        # Don't send tensor downstream if on last stage.
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if k == (num_warmup_microbatches - 1) and not forward_only and \
                not all_warmup_microbatches:
            input_tensor_grad = None
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            next_chunk_id = get_model_chunk_id(0, forward=False)
            recv_next_shapes = get_recv_shapes_interleaved(next_chunk_id, model_type, False)
            print_comm_start("Rank {}: Warm up - Microbatch {}, Send next: {}, Send prev: {}, Recv prev: {}, Recv next: {} started.".format(
                rank,
                k,
                _tensor_shape_to_str(output_tensor),
                _tensor_shape_to_str(input_tensor_grad),
                recv_prev_shapes if recv_prev else None,
                recv_next_shapes if recv_next else None,
                ), flush=True)
            input_tensor, output_tensor_grad = \
                p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor, input_tensor_grad,
                        recv_prev=recv_prev, recv_next=recv_next,
                        receive_prev_shape=recv_prev_shapes,
                        receive_next_shape=recv_next_shapes,
                        timers=timers)
            print_comm_end("Rank {}: Warm up - Microbatch {}, Send next: {}, Send prev: {}, Recv prev: {}, Recv next: {} ended.".format(
                rank,
                k,
                _tensor_shape_to_str(output_tensor),
                _tensor_shape_to_str(input_tensor_grad),
                recv_prev_shapes if recv_prev else None,
                recv_next_shapes if recv_next else None,
                ), flush=True)
            output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
        else:
            print_comm_start("Rank {}: Warm up - Microbatch {}, Send next: {}, Send prev: {}, Recv prev: {}, Recv next: {} started.".format(
                rank,
                k,
                _tensor_shape_to_str(output_tensor),
                None,
                recv_prev_shapes,
                None,
                ), flush=True)
            input_tensor = \
                p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev,
                    tensor_shape=recv_prev_shapes,
                    timers=timers)
            print_comm_end("Rank {}: Warm up - Microbatch {}, Send next: {}, Send prev: {}, Recv prev: {}, Recv next: {} ended.".format(
                rank,
                k,
                _tensor_shape_to_str(output_tensor),
                None,
                recv_prev_shapes,
                None,
                ), flush=True)
        input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if output_tensor is not None:
            deallocate_output_tensor(output_tensor[0])

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        output_tensor = forward_step_helper(forward_k)

        # Backward pass.
        backward_k = k
        input_tensor_grad = backward_step_helper(backward_k)

        # Send output_tensor and input_tensor_grad, receive input_tensor
        # and output_tensor_grad.

        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        if mpu.is_pipeline_first_stage():
            input_tensor_grad = None

        # Determine if peers are sending, and where in data structure to put
        # received tensors.
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            # First stage is ahead of last stage by (pipeline_parallel_size - 1).
            next_forward_model_chunk_id = get_model_chunk_id(
                forward_k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                recv_prev = False
            next_forward_model_chunk_id += 1
        else:
            next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                             forward=True)
        recv_prev_shapes = get_recv_shapes_interleaved(next_forward_model_chunk_id, model_type, True)

        recv_next = True
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                recv_next = False
            next_backward_model_chunk_id -= 1
        else:
            next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1,
                                                              forward=False)
        recv_next_shapes = get_recv_shapes_interleaved(next_backward_model_chunk_id, model_type, False)

        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False
        print_comm_start("Rank {}: 1F1B - forward k: {}, backward k: {}, Send next: {}, Send prev: {}, Recv prev: {}, Recv next: {}, started.".format(
            rank,
            forward_k,
            backward_k,
            _tensor_shape_to_str(output_tensor),
            _tensor_shape_to_str(input_tensor_grad),
            recv_prev_shapes if recv_prev else None,
            recv_next_shapes if recv_next else None,
            ), flush=True)
        # Communicate tensors.
        input_tensor, output_tensor_grad = \
            p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor, input_tensor_grad,
                    recv_prev=recv_prev, recv_next=recv_next,
                    receive_prev_shape=recv_prev_shapes,
                    receive_next_shape=recv_next_shapes,
                    timers=timers)
        print_comm_end("Rank {}: 1F1B - forward k: {}, backward k: {}, Send next: {}, Send prev: {}, Recv prev: {}, Recv next: {}, ended.".format(
            rank,
            forward_k,
            backward_k,
            _tensor_shape_to_str(output_tensor),
            _tensor_shape_to_str(input_tensor_grad),
            recv_prev_shapes if recv_prev else None,
            recv_next_shapes if recv_next else None,
            ), flush=True)
        if output_tensor is not None:
            deallocate_output_tensor(output_tensor[0])

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(
                output_tensor_grad)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if all_warmup_microbatches:
            recv_bw_chunk_id = get_model_chunk_id(num_microbatches_remaining, forward=False)
            recv_shapes = get_recv_shapes_interleaved(recv_bw_chunk_id, model_type, True)
            print_comm_start("Rank {}: recv backward: {} started.".format(
                rank,
                recv_shapes
                ), flush=True)
            output_tensor_grads[num_model_chunks-1].append(
                p2p_communication.recv_backward(recv_shapes, timers=timers))
            print_comm_end("Rank {}: recv backward: {} ended.".format(
                rank,
                recv_shapes
                ), flush=True)
        for k in range(num_microbatches_remaining, num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (num_microbatches - 1):
                recv_next = False
            if recv_next:
                recv_next_shapes = get_recv_shapes_interleaved(next_backward_model_chunk_id, model_type, False)
            else:
                recv_next_shapes = None
            print_comm_start("Rank {}: Send BW: {}, Recv BW: {} started.".format(
                rank,
                _tensor_shape_to_str(input_tensor_grad),
                recv_next_shapes
                ), flush=True)
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next,
                    tensor_shape=recv_next_shapes,
                    timers=timers))
            print_comm_end("Rank {}: Send BW: {}, Recv BW: {} ended.".format(
                rank,
                _tensor_shape_to_str(input_tensor_grad),
                recv_next_shapes
                ), flush=True)

    return forward_data_store


def get_tensor_shapes(rank, model_type):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    args = get_args()
    tensor_shapes = []

    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length

    if model_type == ModelType.encoder_and_decoder:
        if args.sequence_parallel:
            decoder_seq_length = args.decoder_seq_length // mpu.get_tensor_model_parallel_world_size()
        else:
            decoder_seq_length = args.decoder_seq_length

        if mpu.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, args.micro_batch_size, args.hidden_size))
            tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
    else:
        tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
    return tensor_shapes



def forward_backward_pipelining_without_interleaving(forward_step_func,
                                                     data_iterator,
                                                     model,
                                                     optimizer,
                                                     timers,
                                                     forward_only,
                                                     collect_non_loss_data=False):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    
    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    if args.deepspeed:
        from deepspeed.runtime.engine import DeepSpeedEngine
        unwrap_classes = (torchDDP, LocalDDP, Float16Module, DeepSpeedEngine)
    else:
        unwrap_classes = (torchDDP, LocalDDP, Float16Module)
    unwrapped_model = unwrap_model(
        model, unwrap_classes)
    model_type = unwrapped_model.model_type
    rank = mpu.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(rank-1, model_type)
    send_tensor_shapes = get_tensor_shapes(rank, model_type)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    current_fw_mb = 0
    current_bw_mb = 0
    def _annotate_microbatch(stage, is_forward=True):
        if is_forward:
            return torch.cuda.nvtx.range("Forward Microbatch {} {}".format(current_fw_mb, stage))
        else:
            return torch.cuda.nvtx.range("Backward Microbatch {} {}".format(current_bw_mb, stage))
    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        with _annotate_microbatch("recv"):
            input_tensor = recv_forward(recv_tensor_shapes, timers=timers)
        with _annotate_microbatch("compute"):
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, forward_data_store,
                                        timers, collect_non_loss_data)
        with _annotate_microbatch("send"):
            send_forward(output_tensor, send_tensor_shapes, timers=timers)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0])

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        with _annotate_microbatch("recv"):
            input_tensor = recv_forward(recv_tensor_shapes, timers=timers)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))

        with _annotate_microbatch("compute"):
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, forward_data_store,
                                        timers, collect_non_loss_data)
        if forward_only:
            with _annotate_microbatch("send"):
                send_forward(output_tensor, send_tensor_shapes, timers=timers)
                current_fw_mb += 1
            if not last_iteration:
                with _annotate_microbatch("recv"):
                    input_tensor = recv_forward(recv_tensor_shapes, timers=timers)

        else:
            with _annotate_microbatch("send_fw_recv_bw"):
                output_tensor_grad = \
                    send_forward_recv_backward(output_tensor,
                                            send_tensor_shapes,
                                            timers=timers)
            current_fw_mb += 1
            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0])

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            with _annotate_microbatch("compute", is_forward=False):
                input_tensor_grad = \
                    backward_step(optimizer, input_tensor, output_tensor,
                                output_tensor_grad, timers, ds_model=model)

            if last_iteration:
                input_tensor = None
                with _annotate_microbatch("send", is_forward=False):
                    send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)
            else:
                with _annotate_microbatch("send_bw_recv_fw", is_forward=False):
                    input_tensor = \
                        send_backward_recv_forward(
                            input_tensor_grad, recv_tensor_shapes, timers=timers)
            current_bw_mb += 1

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            with _annotate_microbatch("recv", is_forward=False):
                output_tensor_grad = recv_backward(send_tensor_shapes, timers=timers)

            with _annotate_microbatch("compute", is_forward=False):
                input_tensor_grad = \
                    backward_step(optimizer, input_tensor, output_tensor,
                                output_tensor_grad, timers, ds_model=model)

            with _annotate_microbatch("send", is_forward=False):
                send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)

    return forward_data_store
