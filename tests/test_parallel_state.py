import torch
import megatron.core.parallel_state as ps
import pytest
from tests.test_utilities import Utils
import os 

rank = Utils.rank
world_size = Utils.world_size

def test_initialize__and_destroy_model_parallel():
    with pytest.raises(AssertionError):
        assert(ps.initialize_model_parallel())
    Utils.initialize_distributed()
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(tensor_model_parallel_size=2*world_size))
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(pipeline_model_parallel_size=2*world_size))
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(pipeline_model_parallel_size=world_size, tensor_model_parallel_size=world_size))
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(virtual_pipeline_model_parallel_size=2))
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    assert(ps.model_parallel_is_initialized())
    assert(ps.get_model_parallel_group() is not None)
    assert(ps.get_tensor_model_parallel_group() is not None)
    assert(ps.get_pipeline_model_parallel_group() is not None)
    assert(ps.get_data_parallel_group() is not None)  
    Utils.destroy_model_parallel()
    assert(ps._MODEL_PARALLEL_GROUP is None)

def test_pipeline_parallel_initializations():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    assert(ps.get_pipeline_model_parallel_first_rank() == rank % 2 )
    assert(ps.get_data_parallel_src_rank() == rank)
    assert(ps.get_pipeline_model_parallel_next_rank() == ((rank + 2) % world_size))
    assert(ps.get_pipeline_model_parallel_prev_rank() == ((rank - 2) % world_size))
    Utils.destroy_model_parallel()

def test_data_parallel_initializations():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.get_data_parallel_src_rank() == rank)
    assert(ps.get_data_parallel_world_size() == 1)
    assert(ps.get_data_parallel_rank() == 0)
    Utils.destroy_model_parallel()
    

def test_tensor_model_parellel_world_size():
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    assert(ps.get_tensor_model_parallel_world_size() == world_size)
    ps.set_tensor_model_parallel_world_size(None)
    assert(ps.get_tensor_model_parallel_world_size() == world_size)
    Utils.destroy_model_parallel()
    

def test_pipeline_model_parallel_world_size():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.get_pipeline_model_parallel_world_size() == world_size)
    ps.set_pipeline_model_parallel_world_size(None)
    assert(ps.get_pipeline_model_parallel_world_size() == world_size)
    Utils.destroy_model_parallel()    
    

def test_tensor_model_parallel_rank():
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    assert(ps.get_tensor_model_parallel_rank() == rank)
    ps.set_tensor_model_parallel_rank(None)
    assert(ps.get_tensor_model_parallel_rank() == rank)    
    Utils.destroy_model_parallel()    
    

def test_pipeline_model_parallel_rank():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.get_pipeline_model_parallel_rank() == rank)
    ps.set_pipeline_model_parallel_rank(None)
    assert(ps.get_pipeline_model_parallel_rank() == rank)
    Utils.destroy_model_parallel()
    

def test_is_pipeline_first_stage():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.is_pipeline_first_stage(ignore_virtual=True) == (rank == 0))
    assert(ps.is_pipeline_first_stage() == (rank == 0))
    Utils.destroy_model_parallel()
    

def test_is_pipeline_last_stage():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.is_pipeline_last_stage(ignore_virtual=True) == (rank == world_size-1))
    assert(ps.is_pipeline_last_stage() == (rank == world_size-1))
    Utils.destroy_model_parallel()
    

def test_virtual_pipeline_model_parallel_rank():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    ps.set_virtual_pipeline_model_parallel_rank(rank)
    assert(ps.get_virtual_pipeline_model_parallel_rank() == rank)
    Utils.destroy_model_parallel()
    

def test_get_tensor_model_parallel_src_rank():
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    assert(ps.get_tensor_model_parallel_src_rank() == ((rank // world_size) * world_size))
    Utils.destroy_model_parallel()

def test_get_num_layers_basic(monkeypatch):
    """测试 _get_num_layers 的基本分支（单卡/多卡/encoder-decoder/standalone embedding等）"""
    from types import SimpleNamespace
    import megatron.model.transformer as transformer

    # mock mpu
    class MockMPU:
        def __init__(self, rank=0, world_size=1, before_split=True):
            self._rank = rank
            self._world_size = world_size
            self._before_split = before_split
        def get_pipeline_model_parallel_rank(self):
            return self._rank
        def get_pipeline_model_parallel_world_size(self):
            return self._world_size
        def is_pipeline_stage_before_split(self):
            return self._before_split
    
    # 单卡 encoder-only
    args = SimpleNamespace(
        adalayer=False,
        use_dynapipe=False,
        num_layers=12,
        encoder_num_layers=12,
        decoder_num_layers=12,
        transformer_pipeline_model_parallel_size=1,
        standalone_embedding_stage=False,
        virtual_pipeline_model_parallel_size=None,
        model_type=None,
        pipeline_model_parallel_split_rank=None
    )
    monkeypatch.setattr(transformer, "mpu", MockMPU(rank=0, world_size=1))
    assert transformer._get_num_layers(args, is_encoder_and_decoder_model=False) == 12

    # 多卡 encoder-only
    args.transformer_pipeline_model_parallel_size = 2
    monkeypatch.setattr(transformer, "mpu", MockMPU(rank=0, world_size=2))
    assert transformer._get_num_layers(args, is_encoder_and_decoder_model=False) == 6
    monkeypatch.setattr(transformer, "mpu", MockMPU(rank=1, world_size=2))
    assert transformer._get_num_layers(args, is_encoder_and_decoder_model=False) == 6

    # encoder-decoder, split, 非standalone embedding
    args.num_layers = 12
    args.encoder_num_layers = 8
    args.decoder_num_layers = 4
    args.pipeline_model_parallel_split_rank = 2
    args.transformer_pipeline_model_parallel_size = 4
    args.standalone_embedding_stage = False
    monkeypatch.setattr(transformer, "mpu", MockMPU(rank=0, world_size=4, before_split=True))
    assert transformer._get_num_layers(args, is_encoder_and_decoder_model=True) == 4
    monkeypatch.setattr(transformer, "mpu", MockMPU(rank=2, world_size=4, before_split=False))
    assert transformer._get_num_layers(args, is_encoder_and_decoder_model=True) == 2

    # encoder-decoder, split, standalone embedding
    args.standalone_embedding_stage = True
    monkeypatch.setattr(transformer, "mpu", MockMPU(rank=0, world_size=4, before_split=True))
    assert transformer._get_num_layers(args, is_encoder_and_decoder_model=True) == 0
    monkeypatch.setattr(transformer, "mpu", MockMPU(rank=1, world_size=4, before_split=True))
    assert transformer._get_num_layers(args, is_encoder_and_decoder_model=True) == 4

    # adalayer + dynapipe
    args.adalayer = True
    args.use_dynapipe = True
    args.dynapipe_layer_to_device = [0,1,1,2,2,2]
    monkeypatch.setattr(transformer, "mpu", MockMPU(rank=2, world_size=3))
    assert transformer._get_num_layers(args, is_encoder_and_decoder_model=False) == 3 