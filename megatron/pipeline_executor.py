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

def _create_forward_handler(forward_step_func, data_iterator, model):
    args = get_args()
    timers = get_timers()
    fwd_bwd_timers = timers if args.timing_log_level > 1 else None
    def _handle_forward(exec: PipelineExecutor, instr: ForwardPass):
        with torch.cuda.nvtx.range("forward_{}_{}".format(instr.microbatch, instr.stage)):
            buffer_ids = instr.buffer_ids
            input_tensor = [exec.buffer_slots[buffer_id] for buffer_id in buffer_ids if exec.buffer_slots[buffer_id] is not None]
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
            exec.output_tensors[key] = outputs
            if not isinstance(outputs, list):
                outputs = [outputs]
            deallocate_output_tensor(outputs[0])
            assert len(outputs) <= len(buffer_ids), "Number of outputs is greater than number of buffers"
            # output may not use all the buffers
            # we only fill the first len(outputs) buffers
            for buffer_id, output in zip(buffer_ids[:len(outputs)], outputs):
                exec.buffer_slots[buffer_id] = output
    return _handle_forward

def _create_backward_handler(optimizer):
    args = get_args()
    timers = get_timers()
    fwd_bwd_timers = timers if args.timing_log_level > 1 else None
    def _handle_backward(exec: PipelineExecutor, instr: BackwardPass):
        with torch.cuda.nvtx.range("backward_{}_{}".format(instr.microbatch, instr.stage)):
            # interleaved scheduling is more involved, supporting it later
            assert hasattr(exec, "input_tensors")
            assert hasattr(exec, "output_tensors")
            buffer_ids = instr.buffer_ids
            key = (instr.microbatch, instr.stage // 2)
            input_tensor = exec.input_tensors[key]
            exec.input_tensors[key] = None
            output_tensor = exec.output_tensors[key]
            exec.output_tensors[key] = None
            output_tensor_grad = [exec.buffer_slots[buffer_id] for buffer_id in buffer_ids if exec.buffer_slots[buffer_id] is not None]
            input_tensor_grad = \
                    backward_step(optimizer, input_tensor, output_tensor,
                                output_tensor_grad, fwd_bwd_timers)
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

def _handle_send_start(exec: PipelineExecutor, instr: CommunicationStartInstruction):
    output_tensors = [exec.buffer_slots[buffer_id] for buffer_id in instr.buffer_ids if exec.buffer_slots[buffer_id] is not None]
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    tensor_shapes = instr.buffer_shapes
    assert len(output_tensors) == len(tensor_shapes), (
        "Number of output tensors and number of tensor shapes do not match."
        " Expected {}, got {}".format(len(output_tensors), len(tensor_shapes)))
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

def _handle_recv_start(exec: PipelineExecutor, instr: CommunicationStartInstruction):
    args=get_args()
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float
    requires_grad = True
    tensor_shapes = instr.buffer_shapes
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
    executor = PipelineExecutor()
    # register handlers
    executor.register_handler(LoadInput, _handle_load_input)
    executor.register_handler(ForwardPass, _create_forward_handler(forward_step_func, data_iterator, model))
    executor.register_handler(BackwardPass, _create_backward_handler(optimizer))
    # comm handlers
    executor.register_handler(SendActivationStart, _handle_send_start)
    executor.register_handler(SendGradStart, _handle_send_start)
    executor.register_handler(RecvActivationStart, _handle_recv_start)
    executor.register_handler(RecvGradStart, _handle_recv_start)
    executor.register_handler(SendActivationFinish, _handle_send_finish)
    executor.register_handler(SendGradFinish, _handle_send_finish)
    executor.register_handler(RecvActivationFinish, _handle_recv_finish)
    executor.register_handler(RecvGradFinish, _handle_recv_finish)
    executor.check_all_handlers_registered()
    return executor







