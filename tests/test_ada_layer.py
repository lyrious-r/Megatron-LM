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
    # args.standalone_embedding_stage = True
    # args.transformer_pipeline_model_parallel_size = 5
    # monkeypatch.setattr(transformer, "mpu", MockMPU(rank=0, world_size=4, before_split=True))
    # assert transformer._get_num_layers(args, is_encoder_and_decoder_model=True) == 0
    # monkeypatch.setattr(transformer, "mpu", MockMPU(rank=1, world_size=4, before_split=True))
    # assert transformer._get_num_layers(args, is_encoder_and_decoder_model=True) == 4

    # adalayer + dynapipe
    args.adalayer = True
    args.use_dynapipe = True
    args.dynapipe_layer_to_device = [0,1,1,2,2,2]
    monkeypatch.setattr(transformer, "mpu", MockMPU(rank=0, world_size=3))
    assert transformer._get_num_layers(args, is_encoder_and_decoder_model=False) == 1