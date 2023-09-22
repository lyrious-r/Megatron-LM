# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron import get_args
from megatron.core import mpu
from megatron import get_num_microbatches
from megatron.utils import print_rank_0
from megatron.data.t5_dataset import T5SupervisedDataset, T5UnsupervisedDataset
from typing import Union


from dynapipe.model import DynaPipeCluster, TransformerModelSpec
from dynapipe.pipe.data_loader import DynaPipeDataLoader, TrainingSpec

def build_pretraining_data_loader(dataset, consumed_samples, virtual_pp_rank=0, n_virtual_pp_ranks=1, is_training=False):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "ordered":
        batch_sampler = MegatronPretrainingOrderedSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
            dynamic_batchsize=args.dynamic_batchsize,
            tokens_per_global_batch=args.tokens_per_global_batch,
            skip_iters=args.skip_iters,
            use_dynapipe=args.use_dynapipe,
            is_training=is_training,
        )
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    if args.use_dynapipe and is_training:
        assert isinstance(dataset, T5SupervisedDataset)
        dataset: T5SupervisedDataset
        cluster_spec = DynaPipeCluster(
            args.dynapipe_device_to_node,
            [args.dynapipe_device_memory_limit] * len(args.dynapipe_device_to_node),
            args.dynapipe_intra_node_bw,
            args.dynapipe_inter_node_bw,
            args.dynapipe_intra_node_lat,
            args.dynapipe_inter_node_lat,
        )
        buffer_size = args.dynapipe_prefetch_planner_num_workers
        listener_workers = args.dynapipe_prefetch_listener_num_workers
        dp_size = torch.distributed.get_world_size() // \
                    (args.tensor_model_parallel_size * 
                     args.pipeline_model_parallel_size)
        if dataset.inputs_only:
            n_encoder_layers = args.num_layers
            n_decoder_layers = 0
        else:
            n_encoder_layers = args.encoder_num_layers
            n_decoder_layers = args.decoder_num_layers
        training_spec = TrainingSpec(
            args.dynapipe_cost_model,
            cluster_spec,
            TransformerModelSpec(n_encoder_layers, n_decoder_layers,
                                args.hidden_size, args.num_attention_heads,
                                args.ffn_hidden_size, args.kv_channels),
            dp_size,
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            args.dynapipe_zero_stage,
            args.dynapipe_layer_to_device,
            args.dynapipe_device_memory_limit,
            args.dynapipe_partition_algo,
            args.dynapipe_token_based_partition_mbs,
            schedule_method=args.dynapipe_schedule_method,
            disable_mb_permutation=args.dynapipe_disable_mb_permutation,
            disable_scheduler_memory_limit=args.dynapipe_disable_scheduler_memory_limit,
            enable_packing=args.dynapipe_enable_packing,
            per_mb_memory_fraction=args.dynapipe_per_mb_mem_fraction,
            round_seqlen_multiple=args.dynapipe_round_seqlen_multiple,
            seqlen_offset=args.dynapipe_seqlen_offset,
            limit_rc_type=args.dynapipe_limit_rc_type,
            model_type="gpt" if dataset.inputs_only else "t5",
        )
        node_rank = torch.distributed.get_rank() // int(os.environ["LOCAL_WORLD_SIZE"])
        node_size = torch.distributed.get_world_size() // int(os.environ["LOCAL_WORLD_SIZE"])
        encoder_key = "text_enc" if not dataset.inputs_only else "text"
        decoder_key = "text_dec" if not dataset.inputs_only else None
        joint_dataloader = DynaPipeDataLoader(training_spec,
                                        dataset,
                                        dataset.pack_fn,
                                        dataset.constructor_fn,
                                        is_kv_host=torch.distributed.get_rank() == 0,
                                        node_rank=node_rank,
                                        node_local_rank=int(os.environ['LOCAL_RANK']),
                                        node_size = node_size,
                                        dp_rank = mpu.get_data_parallel_rank(),
                                        pp_rank=mpu.get_pipeline_model_parallel_rank(),
                                        virtual_pp_rank=virtual_pp_rank,
                                        batch_sampler=batch_sampler,
                                        num_workers=listener_workers,
                                        num_preprocess_workers=buffer_size,
                                        pin_memory=True,
                                        encoder_key=encoder_key,
                                        decoder_key=decoder_key,)
        return joint_dataloader

    # Torch dataloader.
    if isinstance(dataset, T5SupervisedDataset):
        # dynamic microbatching
        collate_fn = dataset.non_dynapipe_collate_fn
    else:
        collate_fn = None
    return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )


class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

class MegatronPretrainingOrderedSampler(MegatronPretrainingRandomSampler):
    # this sampler assumes that the dataset is already shuffled
    # supports dynamic batching
    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
        micro_batch_size=None,
        dynamic_batchsize=False,
        tokens_per_global_batch=None,
        skip_iters=0,
        use_dynapipe=False,
        is_training=True,
    ):
        super().__init__(
            dataset,
            total_samples,
            consumed_samples,
            micro_batch_size,
            data_parallel_rank,
            data_parallel_size,
            data_sharding,
        )
        if not isinstance(dataset, (T5UnsupervisedDataset, T5SupervisedDataset)):
            raise NotImplementedError(
                "Only T5Dataset is supported for ordered sampler for now."
            )
        self.dataset: Union[T5UnsupervisedDataset, T5SupervisedDataset]
        if not hasattr(dataset, "ordered") and dataset.ordered:
            raise ValueError(
                "Dataset should be ordered for ordered sampler."
            )
        if not (dynamic_batchsize or micro_batch_size):
            raise ValueError(
                "If dynamic_batchsize is False, micro_batch_size should be provided."
            )
        if micro_batch_size:
            self._global_batch_size_per_rank = (
                get_num_microbatches() * self.micro_batch_size
            )
            self._global_batch_size = (
                self._global_batch_size_per_rank * self.data_parallel_size
            )
        self._dynamic_batchsize = dynamic_batchsize
        if dynamic_batchsize:
            if not tokens_per_global_batch:
                raise ValueError(
                    "tokens_per_global_batch should be provided for dynamic batching"
                )
            self._tokens_per_global_batch = tokens_per_global_batch
        self._is_supervised_dataset = isinstance(dataset, T5SupervisedDataset)
        self.use_dynapipe = use_dynapipe
        # handle skip iters
        self.skip_iters = skip_iters
        if self.skip_iters > 0 and is_training:
            current_epoch_samples, _ = self._calc_sample_offsets()
            start_idx = current_epoch_samples
            for _ in range(self.skip_iters):
                if self._dynamic_batchsize:
                    # assume dataset is pre_divided among data parallel groups
                    next_batch_end_idx = self._get_next_batch(start_idx, self.total_samples)
                    assert next_batch_end_idx > start_idx
                    batch = list(range(start_idx, next_batch_end_idx))
                    self.consumed_samples += len(batch)
                    start_idx = next_batch_end_idx

    def _get_next_batch(self, start_idx, end_idx):
        assert self._dynamic_batchsize
        current_batch_tokens = 0
        current_batch_end_idx = None

        for idx in range(start_idx, end_idx):
            input_seq_len = min(
                self.dataset.get_seq_len(idx), self.dataset.max_seq_length
            )
            target_seq_len = min(
                self.dataset.get_dec_seq_len(idx),
                self.dataset.max_seq_length_dec,
            )
            # here we count both input and target tokens
            # another option is to count only input tokens
            tokens_if_added = current_batch_tokens + input_seq_len + target_seq_len
            # if not using dynapipe, each microbatch should have fixed shape
            # so the number pf samples in a minibatch should be a multiple of
            # microbatch size
            is_microbatch_boundary = (
                (idx - start_idx) % self.micro_batch_size == 0 if not self.use_dynapipe else True
            )
            if (
                # current batch is not empty
                current_batch_tokens
                # contains roughly _tokens_per_global_batch tokens
                and tokens_if_added > self._tokens_per_global_batch
                and is_microbatch_boundary
            ):
                # stop adding samples to current batch
                current_batch_end_idx = idx
                break
            current_batch_tokens = tokens_if_added
        if current_batch_end_idx is None:
            current_batch_end_idx = end_idx
        return current_batch_end_idx

    def _calc_sample_offsets(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)
        current_epoch_samples = self.consumed_samples % active_total_samples
        epoch = self.epoch
        if not self._dynamic_batchsize:
            assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0
        if not self.data_sharding:
            print_rank_0(
                "WARNING: data sharding must be enabled for sorted sampler. Ignoring setting."
            )
        return current_epoch_samples, epoch

    def __iter__(self):
        current_epoch_samples, epoch = self._calc_sample_offsets()
        g = torch.Generator()
        g.manual_seed(epoch)
        if self._dynamic_batchsize:
            assert self.use_dynapipe, "Please use precalculated batch size for packed training."
            start_idx = current_epoch_samples
            while start_idx < self.total_samples:
                next_batch_end_idx = self._get_next_batch(start_idx, self.total_samples)
                assert next_batch_end_idx > start_idx
                batch = list(range(start_idx, next_batch_end_idx))
                self.consumed_samples += len(batch)
                start_idx = next_batch_end_idx
                yield batch
        else:
            # since we are using sorted dataset, data access is strided for each rank
            stride_size_per_batch = self.micro_batch_times_data_parallel_size
            n_batches = self.total_samples // stride_size_per_batch
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_offset = self.data_parallel_rank

            per_batch_start_idx = torch.randperm(n_batches, generator=g).tolist()
            batch_idx_range = [
                start_offset + x * stride_size_per_batch
                for x in per_batch_start_idx[bucket_offset:]
            ]

            batch = []
            # Last batch if not complete will be dropped.
            for idx in batch_idx_range:
                for i in range(self.micro_batch_size):
                    batch.append(idx + i * self.data_parallel_size)
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
