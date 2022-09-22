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

"""Dataloaders."""


import random
import bisect
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron import get_args
from megatron import mpu
from megatron import get_num_microbatches
from megatron.utils import print_rank_0
from megatron.data.t5_dataset import T5SupervisedDataset, T5UnsupervisedDataset

from plopt.memory_utils import InvTransformerMemoryModel


def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == "single":
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )
        shape_iterator = None
        n_iters_per_epoch = len(dataset) // args.global_batch_size
    elif args.dataloader_type == "cyclic":
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
        )
        shape_iterator = None
        n_iters_per_epoch = len(dataset) // args.global_batch_size
    elif args.dataloader_type == "sorted":
        batch_sampler = MegatronPretrainingSortedSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
            dynamic_batchsize=args.dynamic_batchsize,
            dynamic_batch_level=args.dynamic_batch_level,
            seq_len_buckets=args.seq_len_buckets,
            max_truncation_factor=args.max_truncation_factor,
            min_truncation_seq_len=args.min_truncation_seq_len,
        )
        shape_iterator = batch_sampler.get_shape_iterator()
        n_iters_per_epoch = batch_sampler.n_iters_per_epoch()
    else:
        raise Exception(
            "{} dataloader type is not supported.".format(args.dataloader_type)
        )

    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    ), shape_iterator, n_iters_per_epoch


class MegatronPretrainingSampler:
    def __init__(
        self,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last=True,
    ):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = (
            self.micro_batch_size * data_parallel_size
        )
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, "no sample to consume: {}".format(
            self.total_samples
        )
        assert (
            self.consumed_samples < self.total_samples
        ), "no samples left to consume: {}, {}".format(
            self.consumed_samples, self.total_samples
        )
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), "data_parallel_rank should be smaller than data size: {}, " "{}".format(
            self.data_parallel_rank, data_parallel_size
        )

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
    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = (
            self.micro_batch_size * data_parallel_size
        )
        self.last_batch_size = (
            self.total_samples % self.micro_batch_times_data_parallel_size
        )

        # Sanity checks.
        assert self.total_samples > 0, "no sample to consume: {}".format(
            self.total_samples
        )
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), "data_parallel_rank should be smaller than data size: {}, " "{}".format(
            self.data_parallel_rank, data_parallel_size
        )

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
            bucket_size = (
                self.total_samples // self.micro_batch_times_data_parallel_size
            ) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (
                self.total_samples // self.micro_batch_size
            ) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[
                self.data_parallel_rank :: self.data_parallel_size
            ]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

class ShapeIterator():
    def __init__(self, shape_generator):
        self.shape_generator = shape_generator

    def __iter__(self):
        return self.shape_generator()

class MegatronPretrainingSortedSampler(MegatronPretrainingRandomSampler):
    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
        dynamic_batchsize=False,
        dynamic_batch_level='batch',
        seq_len_buckets=None,
        dec_seq_len_buckets=None,
        microbatch_size_buckets=None,
        max_truncation_factor=0.05,
        min_truncation_seq_len=512,
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
        assert dynamic_batch_level == 'batch', 'Only batch level dynamic batching is supported'
        if not isinstance(dataset, (T5UnsupervisedDataset, T5SupervisedDataset)):
            raise NotImplementedError(
                "Only T5Dataset is supported for sorted sampler for now."
            )
        assert (
            hasattr(dataset, "sorted") and dataset.sorted
        ), "Dataset should be sorted for sorted sampler."
        self._global_batch_size_per_rank = get_num_microbatches() * self.micro_batch_size
        self._global_batch_size = self._global_batch_size_per_rank * self.data_parallel_size
        self._dynamic_batchsize = dynamic_batchsize
        self._seq_len_buckets = seq_len_buckets
        self._dec_seq_len_buckets = dec_seq_len_buckets
        self._microbatch_size_buckets = microbatch_size_buckets
        self._max_truncation_factor = max_truncation_factor
        self._min_truncation_seq_len = min_truncation_seq_len
        self._shape_consumed_samples = consumed_samples
        self._is_supervised_dataset = isinstance(dataset, T5SupervisedDataset)
        if dynamic_batchsize:
            self._precalc_batches()
        else:
            self.dataset.set_per_sample_seq_len_func(lambda _: (self.dataset.max_seq_length, self.dataset.max_seq_length_dec))
    
    def _round_mbs(self, target_mbs):
        mbs_bucket_idx = bisect.bisect_right(self._microbatch_size_buckets, target_mbs) - 1
        if mbs_bucket_idx < 1:
            mbs_bucket_idx = 1
        rounded_mbs = self._microbatch_size_buckets[mbs_bucket_idx]
        return rounded_mbs
    
    def _get_seq_len_bucket_idx(self, seq_len, decoder=False):
        buckets = self._seq_len_buckets if not decoder else self._dec_seq_len_buckets
        bucket_idx = bisect.bisect_left(buckets, seq_len)
        if bucket_idx == len(buckets):
            bucket_idx -= 1
        if not decoder:
            # see if seq_len is within truncation factor of the previous bucket
            if bucket_idx > 1 and seq_len > self._min_truncation_seq_len:
                prev_bucket_seq_len = buckets[bucket_idx - 1]
                if seq_len <= prev_bucket_seq_len * (1 + self._max_truncation_factor):
                    bucket_idx -= 1
        return bucket_idx

    def _n_microbatches_per_global_batch(self, mbs):
        return self._global_batch_size_per_rank // mbs

    def n_iters_per_epoch(self):
        if self._dynamic_batchsize:
            return len(self._batches)
        else:
            return len(self.dataset) // self._global_batch_size

    def _precalc_batches(self):
        assert self._dynamic_batchsize

        # calculate buckets if not provided
        if not self._seq_len_buckets:
            # from 32 to max seq len
            seq_len_buckets = [
                2**i
                for i in range(
                    5, int(np.ceil(np.log2(self.dataset.get_max_seq_len_from_data()))) + 1
                )
            ]
            self._seq_len_buckets = seq_len_buckets
        if not self._dec_seq_len_buckets:
            self._dec_seq_len_buckets = self._seq_len_buckets
        if not self._microbatch_size_buckets:
            self._microbatch_size_buckets = [1, 2, 4] + [8 * i for i in range(self._global_batch_size_per_rank // 8 + 1)]
        self._microbatch_size_buckets = [x for x in self._microbatch_size_buckets if self._global_batch_size_per_rank % x == 0]

        # calculate mbs and broadcast to all dp and pp ranks
        self._per_seq_len_mbs = [-1] * len(self._seq_len_buckets)
        if mpu.get_data_parallel_rank() == 0:
            args = get_args()
            if args.memory_model == "plopt":
                is_encoder = mpu.is_pipeline_stage_before_split()
                num_layers = mpu.get_num_layers(args, True)  # since we only deal with T5
                n_encoder_layers = num_layers if is_encoder else 0
                n_decoder_layers = num_layers if not is_encoder else 0
                inv_mem_model = InvTransformerMemoryModel(
                    args.hidden_size,
                    args.num_attention_heads,
                    n_encoder_layers,
                    n_decoder_layers,
                    args.ffn_hidden_size,
                    args.kv_channels,
                    tp_degree=mpu.get_tensor_model_parallel_world_size(),
                )
                inv_mem_model.set_reference(self.micro_batch_size, args.encoder_seq_length)
                for idx, seq_len in enumerate(self._seq_len_buckets):
                    self._per_seq_len_mbs[idx] = self._round_mbs(inv_mem_model.get_microbatch_size(seq_len))
            else:
                for idx, seq_len in enumerate(self._seq_len_buckets):
                    self._per_seq_len_mbs[idx] = self._round_mbs((self.micro_batch_size * args.encoder_seq_length) // seq_len)
        # broadcast
        buffer = torch.cuda.IntTensor(self._per_seq_len_mbs)
        torch.distributed.broadcast(
            buffer,
            src=mpu.get_data_parallel_src_rank(),
            group=mpu.get_data_parallel_group(),
        )
        torch.distributed.broadcast(
            buffer,
            src=mpu.get_pipeline_model_parallel_first_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        self._per_seq_len_mbs = buffer.tolist()

        # calculate the starting sample index for each sequence length bucket
        self._per_seq_len_bucket_start_indices = [-1] * len(self._seq_len_buckets)
        for i in range(self.total_samples):
            seq_len = self.dataset.get_seq_len(i)
            bucket_idx = self._get_seq_len_bucket_idx(seq_len)
            if self._per_seq_len_bucket_start_indices[bucket_idx] == -1:
                self._per_seq_len_bucket_start_indices[bucket_idx] = i
        # adjust start indices so each bucket's number of samples is a multiple of
        # effective batch size
        self._per_seq_len_bucket_num_samples = [0] * len(self._seq_len_buckets)
        for bucket_idx, start_idx in enumerate(self._per_seq_len_bucket_start_indices):
            if start_idx == -1:
                continue
            next_bucket_start_idx = -1
            for i in range(bucket_idx + 1, len(self._seq_len_buckets)):
                if self._per_seq_len_bucket_start_indices[i] != -1:
                    next_bucket_start_idx = self._per_seq_len_bucket_start_indices[i]
                    break
            if next_bucket_start_idx == -1:
                num_current_bucket_samples = self.total_samples - start_idx
            else:
                num_current_bucket_samples = next_bucket_start_idx - start_idx
            num_batches = num_current_bucket_samples // self._global_batch_size
            end_idx = start_idx + num_batches * self._global_batch_size
            self._per_seq_len_bucket_num_samples[bucket_idx] = (
                num_batches * self._global_batch_size
            )
            if bucket_idx != len(self._seq_len_buckets) - 1:
                self._per_seq_len_bucket_start_indices[bucket_idx + 1] = end_idx
        adusted_total_samples = sum(self._per_seq_len_bucket_num_samples)
        self.last_batch_size = self.total_samples - adusted_total_samples
        print_rank_0("INFO: adjusted total samples from {} to {}. Data wasted: {} ({:.2f}%).".format(self.total_samples, adusted_total_samples, self.last_batch_size, self.last_batch_size / self.total_samples * 100))
        # now we have the start indices and number of samples for each bucket
        # we can precalculate the batches
        self._batches = []
        for bucket_idx, start_idx in enumerate(self._per_seq_len_bucket_start_indices):
            if self._per_seq_len_bucket_num_samples[bucket_idx] > 0:
                num_global_batches = self._per_seq_len_bucket_num_samples[bucket_idx] // self._global_batch_size
                per_rank_samples = (
                    self._per_seq_len_bucket_num_samples[bucket_idx]
                    // self.data_parallel_size
                )
                start_offset = start_idx + self.data_parallel_rank * per_rank_samples
                n_microbatches_per_global_batch = self._n_microbatches_per_global_batch(self._per_seq_len_mbs[bucket_idx])
                for gb in range(num_global_batches):
                    microbatch = []
                    for mb in range(n_microbatches_per_global_batch):
                        start_idx = start_offset + gb * self._global_batch_size + mb * self._per_seq_len_mbs[bucket_idx]
                        microbatch.append((start_idx, start_idx + self._per_seq_len_mbs[bucket_idx]))
                    self._batches.append(microbatch)
        # calculate per global batch decoder seq length
        self._per_global_batch_decoder_seq_len = [] # 2D list [bucket, global_batch]
        for bucket_idx, start_idx in enumerate(self._per_seq_len_bucket_start_indices):
            if self._per_seq_len_bucket_num_samples[bucket_idx] > 0:
                per_batch_seq_lens = []
                num_batches = self._per_seq_len_bucket_num_samples[bucket_idx] // self._global_batch_size
                for i in range(num_batches):
                    start_offset = start_idx + i * self._global_batch_size
                    end_offset = start_offset + self._global_batch_size
                    max_seq_len = 0
                    for sample_idx in range(start_offset, end_offset):
                        if self._is_supervised_dataset:
                            dec_seq_len = self.dataset.get_dec_seq_len(sample_idx)
                        else:
                            dec_seq_len = self.dataset.get_seq_len(sample_idx) * self.dataset.masked_lm_prob + 2
                        max_seq_len = max(dec_seq_len, max_seq_len)
                    per_batch_seq_lens.append(max_seq_len)
                self._per_global_batch_decoder_seq_len.append(per_batch_seq_lens)
        # set per sample seq len func for dataset
        per_global_batch_enc_dec_seq_lengths = []
        global_batch_id = 0
        for start_idx, size, seq_len in zip(
            self._per_seq_len_bucket_start_indices,
            self._per_seq_len_bucket_num_samples,
            self._seq_len_buckets,
        ):
            if size > 0:
                num_global_batches = size // self._global_batch_size
                for _ in range(num_global_batches):
                    dec_seq_len = self._per_global_batch_decoder_seq_len[global_batch_id]
                    dec_bucket_idx = self._get_seq_len_bucket_idx(dec_seq_len, decoder=True)
                    dec_seq_len = self._dec_seq_len_buckets[dec_bucket_idx]
                    per_global_batch_enc_dec_seq_lengths.append((seq_len, dec_seq_len))
                    global_batch_id += 1

        def per_sample_seq_len_func(idx):
            global_batch_id = idx // self._global_batch_size
            return per_global_batch_enc_dec_seq_lengths[global_batch_id]
        self.dataset.set_per_sample_seq_len_func(per_sample_seq_len_func)
        self._per_sample_seq_len_func = per_sample_seq_len_func

    def _calc_sample_offsets(self, is_data_iterator=True):
        active_total_samples = self.total_samples - self.last_batch_size
        if is_data_iterator:
            self.epoch = self.consumed_samples // active_total_samples
            if isinstance(self.dataset, RandomSeedDataset):
                self.dataset.set_epoch(self.epoch)
            current_epoch_samples = self.consumed_samples % active_total_samples
            epoch = self.epoch
        else:
            current_epoch_samples = self._shape_consumed_samples % active_total_samples
            epoch = self._shape_consumed_samples // active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if not self.data_sharding:
            print(
                "WARNING: data sharding must be enabled for sorted sampler. Ignoring setting."
            )
        return current_epoch_samples, epoch

    def _dynamic_size_get_batch_order(self, is_data_iterator=True):
        current_epoch_samples, epoch = self._calc_sample_offsets(is_data_iterator=is_data_iterator)
        g = torch.Generator()
        g.manual_seed(epoch)
        n_global_batches = len(self._batches)
        batch_order = torch.randperm(n_global_batches, generator=g).tolist()
        cumulative_samples = 0
        for batch_offset in range(len(batch_order)):
            microbatches = self._batches[batch_offset]
            if cumulative_samples < current_epoch_samples:
                for microbatch in microbatches:
                    cumulative_samples += (
                        microbatch[1] - microbatch[0]
                    ) * self.data_parallel_size
            else:
                break
        return batch_order, batch_offset

    def get_shape_iterator(self):
        if not self._dynamic_batchsize:
            return None

        def shape_generator():
            # generates (mbs, enc_seq_len, dec_seq_len) tuples
            # need to call get_batch_order inside the generator to ensure
            # that shape iterator is in sync with the data iterator upon reset
            batch_order, batch_offset = self._dynamic_size_get_batch_order(is_data_iterator=False)
            for batch_id in batch_order[batch_offset:]:
                microbatches = self._batches[batch_id]
                for mb in microbatches:
                    batch_size = mb[1] - mb[0]
                    enc_seq_len, dec_seq_len = self._per_sample_seq_len_func(mb[0])
                    self._shape_consumed_samples += batch_size * self.data_parallel_size
                    yield batch_size, enc_seq_len, dec_seq_len
        return ShapeIterator(shape_generator)

    def __iter__(self):
        if self._dynamic_batchsize:
            batch_order, batch_offset = self._dynamic_size_get_batch_order()
            for batch_id in batch_order[batch_offset:]:
                microbatches = self._batches[batch_id]
                for mb in microbatches:
                    sample_indices = [idx for idx in range(mb[0], mb[1])]
                    self.consumed_samples += len(sample_indices) * self.data_parallel_size
                    yield sample_indices
        else:
            current_epoch_samples, epoch = self._calc_sample_offsets()
            g = torch.Generator()
            g.manual_seed(epoch)
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
