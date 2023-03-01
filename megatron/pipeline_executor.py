from functools import partial

from plopt.pipe.instructions import * # noqa: F403
from plopt.pipe.executor import PipelineExecutor

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_timers, get_args
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module, ModelType
from megatron.core import mpu
from megatron.schedules import forward_step, backward_step, deallocate_output_tensor

def recompute_level_to_flag(recompute_lvl: RecomputeMethod):
    if recompute_lvl == RecomputeMethod.NONE:
        return None
    elif recompute_lvl == RecomputeMethod.FULL:
        return "full"
    elif recompute_lvl == RecomputeMethod.SELECTIVE:
        return "selective"
    else:
        raise ValueError("Unknown recompute level: {}".format(recompute_lvl))

def with_nvtx_stage_name(stage_name):
    def decorator(func):
        def wrapper(exec: PipelineExecutor, instr: PipeInstruction):
            with torch.cuda.nvtx.range("{}_m{}_s{}".format(stage_name, instr.microbatch, instr.stage)):
                return func(exec, instr)
        return wrapper
    return decorator

def _handle_load_input(exec: PipelineExecutor, instr: LoadInput):
    # just set buffers to none, since actual loading is done 
    # in the forward pass
    for buffer_id in instr.buffer_ids:
        exec.buffer_slots[buffer_id] = None

# def _create_load_input_handler(data_iterator, get_batch_fn):
#     # get_batch_fn should behave differently on encoders and decoders
#     def _handle_load_input(exec: PipelineExecutor, instr: LoadInput):
#         # load input fetches data from the dataloader and puts it in the buffer
#         # Get the batch.
#         timers = get_timers()
#         timers('batch generator', log_level=2).start()
#         with torch.cuda.nvtx.range("batch_generator"):
#             # here batch can be anything that will be used by the model
#             # we don't check the tensors' shapes despite the instr
#             # has a shape attribute
#             # TODO: somehow unify this
#             batch_input = get_batch_fn(data_iterator)
#         timers('batch generator').stop()
#         buffer_ids = instr.buffer_ids
#         if len(buffer_ids) != len(batch_input):
#             raise ValueError("Number of buffers and number of tensors in batch"
#                              " do not match: expected {}, got {}".format(
#                                     len(buffer_ids), len(batch_input)))
#         for buffer_id, tensor in zip(buffer_ids, batch_input):
#             exec.buffer_slots[buffer_id] = tensor 
#     return _handle_load_input

def _create_forward_handler(forward_step_func, data_iterators, models):
    args = get_args()
    timers = get_timers()
    fwd_bwd_timers = timers if args.timing_log_level > 1 else None

    @with_nvtx_stage_name("forward")
    def _handle_forward(exec: PipelineExecutor, instr: ForwardPass):
        # set recompute flag
        flag = recompute_level_to_flag(exec.execution_plan.recompute_method)
        mpu.set_recomputation_level(flag)
        exec.logger.debug("Recompute level: {}".format(flag))
        if args.virtual_pipeline_model_parallel_size is not None:
            # interleaved scheduling
            # needs to call set_virtual_pipeline_model_parallel_rank before
            # forward and backward
            # first get the current model chunk id from the instruction
            if not hasattr(exec, "_rev_chunk_index"):
                exec._rev_chunk_index = {}
                n_assigned_chunks = len(exec.execution_plan.assigned_stages)
                for i, chunk in enumerate(exec.execution_plan.assigned_stages):
                    if i >= n_assigned_chunks // 2:
                        # backward chunk
                        i = n_assigned_chunks - i - 1
                    exec._rev_chunk_index[chunk] = i
            chunk_id = exec._rev_chunk_index[instr.stage]
            model = models[chunk_id]
            data_iterator = data_iterators[chunk_id]
            mpu.set_virtual_pipeline_model_parallel_rank(chunk_id)
        else:
            model = models[0]
            if isinstance(data_iterators, list):
                data_iterator = data_iterators[0]
            else:
                data_iterator = data_iterators
        buffer_ids = instr.buffer_ids
        # in decoder stage, there should be two input tensors
        # first one is received encoder activation, second is last decoder
        # layer's output (or data loaded from data loader)
        # Megatron-LM expects the first one to be decoder input, so we
        # need to swap them
        input_tensor = [exec.buffer_slots[buffer_id] for buffer_id in buffer_ids if exec.buffer_slots[buffer_id] is not None]
        if len(input_tensor) == 2:
            new_input_tensor = [input_tensor[1], input_tensor[0]]
            input_tensor = new_input_tensor
        if len(input_tensor) == 0:
            # no input tensor, load from dataloader
            input_tensor = None
        # create a bunch of local stores
        if not hasattr(exec, "input_tensors"):
            exec.input_tensors = {}
        if not hasattr(exec, "output_tensors"):
            exec.output_tensors = {}
        key = (instr.microbatch, instr.stage)
        exec.input_tensors[key] = input_tensor
        if not hasattr(exec, "forward_data_store"):
            exec.forward_data_store = []
        outputs = forward_step(forward_step_func, data_iterator, model, input_tensor, exec.forward_data_store, fwd_bwd_timers, collect_non_loss_data=False)
        # output_tensors saves the output tensor and a flag indicating
        # whether the tensor should be freed after communication
        # the order of output_tensors follows Megatron-LM
        if isinstance(outputs, list):
            exec.output_tensors[key] = list(zip(outputs, [True, False] if len(outputs) == 2 else [True]))
        else:
            exec.output_tensors[key] = [(outputs, True)]
        if len(outputs) == 2:
            # decoder stage, first output is decoder output, second is
            # encoder activation. We need to swap them to match plopt's
            # order
            new_outputs = [outputs[1], outputs[0]]
            outputs = new_outputs

        if not isinstance(outputs, list):
            outputs = [outputs]
        assert len(outputs) <= len(buffer_ids), "On rank {}, number of outputs is greater than number of buffers ({} v.s. {}) when executing instruction: {}" \
            .format(exec.execution_plan.rank, len(outputs), len(buffer_ids), instr)
        # output may not use all the buffers
        # we only fill the first len(outputs) buffers
        for buffer_id, output in zip(buffer_ids[:len(outputs)], outputs):
            exec.buffer_slots[buffer_id] = output
    return _handle_forward

def _create_backward_handler(optimizer):
    args = get_args()
    timers = get_timers()
    fwd_bwd_timers = timers if args.timing_log_level > 1 else None

    @with_nvtx_stage_name("backward")
    def _handle_backward(exec: PipelineExecutor, instr: BackwardPass):
        if args.virtual_pipeline_model_parallel_size is not None:
            assert hasattr(exec, "_rev_chunk_index")
            chunk_id = exec._rev_chunk_index[instr.stage]
            mpu.set_virtual_pipeline_model_parallel_rank(chunk_id)
        # interleaved scheduling is more involved, supporting it later
        assert hasattr(exec, "input_tensors")
        assert hasattr(exec, "output_tensors")
        buffer_ids = instr.buffer_ids
        key = (instr.microbatch, exec.execution_plan.nstages - instr.stage)
        input_tensor = exec.input_tensors[key]
        exec.input_tensors[key] = None
        output_tensor, _ = zip(*exec.output_tensors[key])
        output_tensor = list(output_tensor)
        exec.output_tensors[key] = None
        output_tensor_grad = [exec.buffer_slots[buffer_id] for buffer_id in buffer_ids if exec.buffer_slots[buffer_id] is not None]
        # same here, if there are two output tensor grads, we need to swap them
        # to match Megatron-LM's order (decoder output, encoder activation)
        if len(output_tensor_grad) == 2:
            new_output_tensor_grad = [output_tensor_grad[1], output_tensor_grad[0]]
            output_tensor_grad = new_output_tensor_grad
        input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                            output_tensor_grad, fwd_bwd_timers)
        if not isinstance(input_tensor_grad, list):
            input_tensor_grad = [input_tensor_grad]
        # swap them back
        if len(input_tensor_grad) == 2:
            new_input_tensor_grad = [input_tensor_grad[1], input_tensor_grad[0]]
            input_tensor_grad = new_input_tensor_grad
        # output may not use all the buffers
        # we only fill the first len(outputs) buffers
        for buffer_id, output in zip(buffer_ids[:len(input_tensor_grad)], input_tensor_grad):
            exec.buffer_slots[buffer_id] = output
    return _handle_backward

_comm_instr_key_map = {
    SendActivationStart: "act",
    SendGradStart: "grad",
    RecvActivationStart: "act",
    RecvGradStart: "grad",
    SendActivationFinish: "act",
    SendGradFinish: "grad",
    RecvActivationFinish: "act",
    RecvGradFinish: "grad",
}

def _transpose_tensor_shape(tensor_shape):
    # Megatron-LM expect communicated tensors to be
    # (sequence length, microbatch size, hidden size)
    # while plopt expects them to be
    # (microbatch size, sequence length, hidden size)
    return (tensor_shape[1], tensor_shape[0], tensor_shape[2])

def _handle_send_start(exec: PipelineExecutor, instr: CommunicationStartInstruction):
    output_tensors = [exec.buffer_slots[buffer_id] for buffer_id in instr.buffer_ids if exec.buffer_slots[buffer_id] is not None]
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    tensor_shapes = instr.buffer_shapes
    # transpose tensor shapes
    tensor_shapes = [_transpose_tensor_shape(s) for s in tensor_shapes]
    assert len(output_tensors) >= len(tensor_shapes), (
        "Number of output tensors and number of tensor shapes do not match."
        " Expected {}, got {}".format(len(output_tensors), len(tensor_shapes)))
    output_tensors = output_tensors[:len(tensor_shapes)]
    pending_ops = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        op = dist.isend(output_tensor, instr.peer)
        pending_ops.append(op)
    key = (instr.microbatch, instr.stage, _comm_instr_key_map[instr.__class__])
    if not hasattr(exec, "pending_send_ops"):
        exec.pending_send_ops = {}
    exec.pending_send_ops[key] = pending_ops

def _handle_send_finish(exec: PipelineExecutor, instr: CommunicationFinishInsturction):
    key = (instr.microbatch, instr.stage, _comm_instr_key_map[instr.__class__])
    pending_ops = exec.pending_send_ops[key]
    exec.pending_send_ops[key] = None
    for op in pending_ops:
        op.wait()

@with_nvtx_stage_name("send_forward_finish")
def _handle_send_forward_finish(exec: PipelineExecutor, instr: CommunicationFinishInsturction):
    # wait
    _handle_send_finish(exec, instr)
    # free output tensor if needed
    output_key = (instr.microbatch, instr.stage)
    output_tensors = exec.output_tensors[output_key]
    for (output_tensor, free) in output_tensors:
        if free:
            deallocate_output_tensor(output_tensor)

def _handle_recv_start(exec: PipelineExecutor, instr: CommunicationStartInstruction):
    args=get_args()
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float
    requires_grad = True
    tensor_shapes = instr.buffer_shapes
    # transpose tensor shapes
    tensor_shapes = [_transpose_tensor_shape(s) for s in tensor_shapes]
    input_tensors = [torch.empty(tensor_shape, dtype=dtype, requires_grad=requires_grad, device=torch.cuda.current_device()) for tensor_shape in tensor_shapes]
    pending_ops = []
    for (input_tensor, tensor_shape) in zip(input_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        op = dist.irecv(input_tensor, instr.peer)
        pending_ops.append(op)
    key = (instr.microbatch, instr.stage, _comm_instr_key_map[instr.__class__])
    if not hasattr(exec, "pending_recv_ops"):
        exec.pending_recv_ops = {}
    exec.pending_recv_ops[key] = pending_ops
    # add the input tensors to the buffer slots
    for buffer_id, input_tensor in zip(instr.buffer_ids, input_tensors):
        exec.buffer_slots[buffer_id] = input_tensor

def _handle_recv_finish(exec: PipelineExecutor, instr: CommunicationFinishInsturction):
    key = (instr.microbatch, instr.stage, _comm_instr_key_map[instr.__class__])
    pending_ops = exec.pending_recv_ops[key]
    exec.pending_recv_ops[key] = None
    for op in pending_ops:
        op.wait()

def get_pipeline_executor(forward_step_func, data_iterator, model, optimizer):
    executor = PipelineExecutor(rank=mpu.get_pipeline_model_parallel_rank())
    # register handlers
    executor.register_handler(LoadInput, _handle_load_input)
    executor.register_handler(ForwardPass, _create_forward_handler(forward_step_func, data_iterator, model))
    executor.register_handler(BackwardPass, _create_backward_handler(optimizer))
    # comm handlers
    executor.register_handler(SendActivationStart, with_nvtx_stage_name("send_forward_start")(_handle_send_start))
    executor.register_handler(SendGradStart, with_nvtx_stage_name("send_backward_start")(_handle_send_start))
    executor.register_handler(RecvActivationStart, with_nvtx_stage_name("recv_forward_start")(_handle_recv_start))
    executor.register_handler(RecvGradStart, with_nvtx_stage_name("recv_backward_start")(_handle_recv_start))
    executor.register_handler(SendActivationFinish, _handle_send_forward_finish)
    executor.register_handler(SendGradFinish, with_nvtx_stage_name("send_backward_finish")(_handle_send_finish))
    executor.register_handler(RecvActivationFinish, with_nvtx_stage_name("recv_forward_finish")(_handle_recv_finish))
    executor.register_handler(RecvGradFinish, with_nvtx_stage_name("recv_backward_finish")(_handle_recv_finish))
    executor.check_all_handlers_registered()
    return executor







