# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""T5 Style dataset."""

import collections

import numpy as np
import torch

from megatron import get_tokenizer, get_args
from megatron.data.dataset_utils import (
    create_masked_lm_predictions,
    get_samples_mapping,
    get_samples_mapping_supervised,
)
from megatron.utils import print_rank_0


def run_pack_samples(
    input_samples_mapping,
    max_seq_len_input,
    target_samples_mapping=None,
    max_seq_len_target=None,
    inputs_only=False,
):
    """Pack multiple samples into a single sequence."""
    # each sample is a list of tuples (start_idx, end_idx, seq_len)
    input_samples = []
    target_samples = []
    assert target_samples_mapping is not None, \
        "Currently only support supervised tasks"
    if target_samples_mapping is not None:
        assert len(input_samples_mapping) == len(
            target_samples_mapping
        ), "input and target samples mapping should have the same length"
    curr_input_seq_len = 0
    curr_target_seq_len = 0
    curr_input_sequence = []
    curr_target_sequence = []
    avg_samples_per_sequence = 0
    total_enc_untruncated_tokens = 0
    total_dec_untruncated_tokens = 0
    total_enc_tokens = 0
    total_dec_tokens = 0
    total_enc_truncated_tokens = 0
    total_dec_truncated_tokens = 0
    num_truncated_samples = 0
    total_enc_tokens_for_samples_truncated = 0
    total_dec_tokens_for_samples_truncated = 0
    for idx in range(len(input_samples_mapping)):
        truncated = False
        input_sample = input_samples_mapping[idx]
        target_sample = target_samples_mapping[idx]

        input_sample_length = input_sample[2]
        target_sample_length = target_sample[2]
        if inputs_only:
            # use concatenated lengths for input
            input_sample_length += target_sample_length
            target_sample_length = 0

        total_enc_untruncated_tokens += input_sample_length
        total_dec_untruncated_tokens += target_sample[2]
        if input_sample_length > max_seq_len_input:
            truncated = True
            total_enc_truncated_tokens += input_sample_length - max_seq_len_input
            total_enc_tokens_for_samples_truncated += input_sample_length
        if target_sample[2] > max_seq_len_target:
            truncated = True
            total_dec_truncated_tokens += target_sample[2] - max_seq_len_target
            total_dec_tokens_for_samples_truncated += target_sample[2]
        input_seq_len = min(input_sample_length, max_seq_len_input)
        target_seq_len = min(target_sample_length, max_seq_len_target)
        total_enc_tokens += input_seq_len
        total_dec_tokens += target_seq_len
        if truncated:
            num_truncated_samples += 1
        # if adding the current sample will exceed the max sequence length,
        # then start a new sequence
        # Note: if inputs_only, we still put target_samples into
        # curr_target_sequence to simplify the code. Actual concatenation
        # happens later.
        if curr_input_seq_len + input_seq_len > max_seq_len_input or (
           curr_target_seq_len + target_seq_len > max_seq_len_target
        ):
            input_samples.append(curr_input_sequence.copy())
            avg_samples_per_sequence += len(curr_input_sequence)
            curr_input_seq_len = 0
            curr_input_sequence = []
            target_samples.append(curr_target_sequence.copy())
            curr_target_seq_len = 0
            curr_target_sequence = []
        curr_input_seq_len += input_seq_len
        curr_input_sequence.append(tuple(input_sample))
        curr_target_seq_len += target_seq_len
        curr_target_sequence.append(tuple(target_sample))
    # last sequence
    if curr_input_seq_len > 0:
        input_samples.append(curr_input_sequence)
        target_samples.append(curr_target_sequence)
        avg_samples_per_sequence += len(curr_input_sequence)
    print_rank_0(
        ">>>> Pack samples: {} input sequences, {} target sequences, avg samples per sequence: {}, enc batching eff: {}, dec batching eff: {}".format(
            len(input_samples),
            len(target_samples),
            avg_samples_per_sequence / len(input_samples),
            total_enc_tokens / (len(input_samples) * max_seq_len_input),
            total_dec_tokens / (len(target_samples) * max_seq_len_target),
        )
    )
    print_rank_0(
        ">>> Truncated {}/{} samples: enc truncated token ratio: {}, dec truncated token ratio: {}, enc ratio among truncated: {}, dec ratio among truncated: {}".format(
            num_truncated_samples, len(input_samples_mapping), 
            total_enc_truncated_tokens / (total_enc_untruncated_tokens + 1e-6), 
            total_dec_truncated_tokens / (total_dec_untruncated_tokens + 1e-6), 
            total_enc_truncated_tokens / (total_enc_tokens_for_samples_truncated + 1e-6), 
            total_dec_truncated_tokens / (total_dec_tokens_for_samples_truncated + 1e-6),
        )
    )
    return input_samples, target_samples


class T5UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, max_seq_length_dec,
                 short_seq_prob, seed,
                 sort_samples=False, pack_samples=False):

        # Params to store.
        self.name = name
        self.seed = seed
        self.sorted = sort_samples
        self.packed = pack_samples
        self.supervised = False
        self.ordered = True if self.sorted or self.packed else False
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(self.indexed_dataset,
                                                   data_prefix,
                                                   num_epochs,
                                                   max_num_samples,
                                                   self.max_seq_length - 2, # account for added tokens
                                                   short_seq_prob,
                                                   self.seed,
                                                   self.name,
                                                   False,
                                                   sort_samples=sort_samples)
        if pack_samples:
            self.packed_samples, _ = run_pack_samples(
                self.samples_mapping, self.max_seq_length
            )

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.sentinel_tokens = tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

    def get_seq_len(self, idx):
        if self.packed:
            return self.max_seq_length
        else:
            return self.samples_mapping[idx, 2]

    def get_padding_efficiency(self):
        total_actual_input_tokens = 0
        total_padded_input_tokens = 0
        total_actual_target_tokens = 0
        total_padded_target_tokens = 0
        for idx in range(len(self)):
            actual_input_seq_len = 0
            if self.packed:
                for (_, _, seq_len) in self.packed_samples[idx]:
                    actual_input_seq_len += seq_len
            else:
                _, _, seq_len = self.samples_mapping[idx]
                actual_input_seq_len += seq_len
            actual_target_seq_len = int(actual_input_seq_len * self.masked_lm_prob)
            total_actual_input_tokens += actual_input_seq_len
            total_actual_target_tokens += actual_target_seq_len
            total_padded_input_tokens += self.max_seq_length
            total_padded_target_tokens += self.max_seq_length_dec
        return (
            total_actual_input_tokens / total_padded_input_tokens,
            total_actual_target_tokens / total_padded_target_tokens,
        )

    def __len__(self):
        if self.packed:
            return len(self.packed_samples)
        else:
            return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        sample = []
        if self.packed:
            for (start_index, end_index, _) in self.packed_samples[idx]:
                sample.extend(self.indexed_dataset[start_index:end_index])
                for index in range(start_index, end_index):
                    sample.append(self.indexed_dataset[index])
        else:
            start_index, end_index, _ = self.samples_mapping[idx]
            for index in range(start_index, end_index):
                sample.append(self.indexed_dataset[index])
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        return build_training_sample(sample, self.max_seq_length,
                                     self.max_seq_length,  # needed for padding
                                     self.max_seq_length_dec,
                                     self.vocab_id_list,
                                     self.vocab_id_to_token_dict,
                                     self.cls_id, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.masked_lm_prob, np_rng,
                                     self.bos_id, self.eos_id,
                                     self.sentinel_tokens)


class T5SupervisedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        input_indexed_dataset,
        target_indexed_dataset,
        data_prefix,
        num_epochs,
        max_num_samples,
        max_seq_length,
        max_seq_length_dec,
        seed,
        inputs_only=False,
        sort_samples=False,
        pack_samples=False,
        dynamic_batchsize=True,
        offline_build=False,
    ):

        # Params to store.
        self.name = name
        self.seed = seed
        self.sorted = sort_samples
        self.packed = pack_samples
        self.supervised = True
        self.ordered = True
        self.inputs_only = inputs_only
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec
        self.input_max_seq_length = -1
        self.target_max_seq_length = -1
        self.input_seq_lengths = None
        self.target_seq_lengths = None
        self.adjusted_num_samples = None
        self.offline_build = offline_build
        self.dynamic_batchsize = dynamic_batchsize

        # Dataset.
        self.input_indexed_dataset = input_indexed_dataset
        self.target_indexed_dataset = target_indexed_dataset

        # Build the samples mapping.
        (
            self.input_samples_mapping,
            self.target_samples_mapping,
        ) = get_samples_mapping_supervised(
            self.input_indexed_dataset,
            self.target_indexed_dataset,
            data_prefix,
            num_epochs,
            max_num_samples,
            self.max_seq_length,
            self.max_seq_length_dec - 2,  # account for added tokens
            self.seed,
            self.name,
            sort_samples=sort_samples,
            offline_build=offline_build,
        )

        if pack_samples:
            self.packed_input_samples, self.packed_target_samples = run_pack_samples(
                self.input_samples_mapping,
                self.max_seq_length,
                self.target_samples_mapping,
                self.max_seq_length_dec - 2,
                inputs_only=self.inputs_only,
            )

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        # Note: we only care about troughput so these ids are not properly
        # set for the GPT model. Modify this in actual training.
        try:
            self.cls_id = tokenizer.cls
        except (NotImplementedError, AttributeError):
            self.cls_id = 0
        try:
            self.sep_id = tokenizer.sep
        except (NotImplementedError, AttributeError):
            self.sep_id = 0
        try:
            self.mask_id = tokenizer.mask
        except (NotImplementedError, AttributeError):
            self.mask_id = 0
        try:
            self.pad_id = tokenizer.pad
        except (NotImplementedError, AttributeError):
            try:
                self.pad_id = tokenizer.eod
            except (NotImplementedError, AttributeError):
                self.pad_id = 0
        try:
            self.bos_id = tokenizer.bos_token_id
        except (NotImplementedError, AttributeError):
            self.bos_id = 0
        try:
            self.eos_id = tokenizer.eos_token_id
        except (NotImplementedError, AttributeError):
            self.eos_id = 0
        try:
            self.sentinel_tokens = tokenizer.additional_special_tokens_ids
        except (NotImplementedError, AttributeError):
            self.sentinel_tokens = [0]
        assert (
            len(self.sentinel_tokens) > 0
        ), "Provide the argument --vocab-extra-ids 100 to the script"
        print_rank_0(" > input sample mapping shapes: {}".format(self.input_samples_mapping.shape))
        print_rank_0(" > target sample mapping shapes: {}".format(self.target_samples_mapping.shape))

    def get_seq_len(self, idx):
        if self.packed:
            return self.max_seq_length
        else:
            if self.inputs_only:
                return self.input_samples_mapping[idx, 2] + self.target_samples_mapping[idx, 2]
            return self.input_samples_mapping[idx, 2]

    def get_dec_seq_len(self, idx):
        if self.inputs_only:
            return 0
        if self.packed:
            return self.max_seq_length_dec
        else:
            return self.target_samples_mapping[idx, 2]

    def set_adjusted_num_samples(self, adjusted_num_samples):
        self.adjusted_num_samples = adjusted_num_samples

    def __len__(self):
        if self.packed:
            return len(self.packed_input_samples)
        else:
            return self.input_samples_mapping.shape[0]

    def pack_fn(self, tensors):
        tensors_to_cat = []
        if len(tensors) == 0 or tensors is None:
            return tensors_to_cat, 0
        if isinstance(tensors[0], list):
            for idx, tensor in enumerate(tensors):
                tensors_to_cat.extend(tensor)
                if idx != len(tensors) - 1:
                    tensors_to_cat.extend([self.sep_id])
        elif isinstance(tensors[0], torch.Tensor):
            for idx, tensor in enumerate(tensors):
                tensors_to_cat.append(tensor)
                if idx != len(tensors) - 1:
                    tensors_to_cat.append(torch.tensor([self.sep_id]))
            tensors_to_cat = torch.cat(tensors_to_cat)
        elif isinstance(tensors[0], np.ndarray):
            for idx, tensor in enumerate(tensors):
                tensors_to_cat.append(tensor)
                if idx != len(tensors) - 1:
                    tensors_to_cat.append(np.array([self.sep_id]))
            tensors_to_cat = np.concatenate(tensors_to_cat)
        else:
            raise ValueError("Unsupported type in pack_fn: {}".format(type(tensors[0])))
        # currently do not return packing masks
        return tensors_to_cat, 0

    def constructor_fn(self, encoder_input,
                            encoder_extra,
                            decoder_input,
                            decoder_extra,
                            encoder_seqlen,
                            decoder_seqlen):
        return build_supervised_training_sample(
            encoder_input,
            decoder_input,
            encoder_seqlen,
            decoder_seqlen,
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.sentinel_tokens,
        )

    def non_plopt_collate_fn(self, batch):
        # pad to max sequence length
        from torch.utils.data import default_collate
        args = get_args()
        result = []
        micro_batch_size = args.micro_batch_size
        current_micro_batch = []
        assert len(batch) % micro_batch_size == 0, "batch size must be divisible by micro batch size"
        for sequence in batch:
            enc_key = "text" if self.inputs_only else "text_enc"
            seqlen_offset = 1 if self.inputs_only else 0
            padded_sequence = build_supervised_training_sample(
                sequence[enc_key],
                sequence["text_dec"] if not self.inputs_only else None,
                args.encoder_seq_length + seqlen_offset,
                args.decoder_seq_length if not self.inputs_only else 0,
                self.pad_id,
                self.bos_id,
                self.eos_id,
                self.sentinel_tokens,
            )
            current_micro_batch.append(padded_sequence)
            if len(current_micro_batch) == micro_batch_size:
                result.append(default_collate(current_micro_batch))
                current_micro_batch = []
        assert len(current_micro_batch) == 0, "micro batch size must be divisible by batch size"
        if not self.dynamic_batchsize:
            # directly return a batch
            assert len(result) == 1
            return result[0]
        return result

    def __getitem__(self, idx):
        if idx == -1:
            # for dynamic microbatch size
            return None
        input_sample = []
        target_sample = []
        if self.packed:
            for (start_index, end_index, _) in self.packed_input_samples[idx]:
                for index in range(start_index, end_index):
                    input_sample.append(self.input_indexed_dataset[index])
            for (start_index, end_index, _) in self.packed_target_samples[idx]:
                for index in range(start_index, end_index):
                    target_sample.append(self.target_indexed_dataset[index])
        else:
            input_start_index, input_end_index, _ = self.input_samples_mapping[idx]
            target_start_index, target_end_index, _ = self.target_samples_mapping[idx]
            for index in range(input_start_index, input_end_index):
                input_sample.append(self.input_indexed_dataset[index])
            for index in range(target_start_index, target_end_index):
                target_sample.append(self.target_indexed_dataset[index])

        if self.inputs_only:
            # concat target to input
            input_sample = input_sample + target_sample
            target_sample = []

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # np_rng = np.random.RandomState(seed=(self.seed + idx))
        if self.packed:
            # run pack fn on the samples
            input_sample, _ = self.pack_fn(input_sample)
            input_sample = [input_sample]
            if not self.inputs_only:
                target_sample, _ = self.pack_fn(target_sample)
                target_sample = [target_sample]

        if not self.inputs_only:
            return build_unpadded_sample(
                input_sample, self.max_seq_length, target_sample, self.max_seq_length_dec
            )
        return build_unpadded_sample(
            input_sample, self.max_seq_length, None, self.max_seq_length_dec
        )


def build_training_sample(sample, target_seq_length,
                          max_seq_length, max_seq_length_dec,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_id, sep_id, mask_id, pad_id,
                          masked_lm_prob, np_rng, bos_id=None,
                          eos_id=None, sentinel_tokens=None):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
        bos_id: start of decoder example id
        eos_id: end of generation id
        sentinel_tokens: unique value to be substituted for every replaced span
    """

    assert target_seq_length <= max_seq_length

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = len(tokens) > max_num_tokens
    tokens = tokens[:max_num_tokens]

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng,
        max_ngrams=10, geometric_dist=True, masking_style="t5")

    # Padding.
    tokens_enc, tokens_dec_in, labels, enc_mask, \
    dec_mask, enc_dec_mask, loss_mask \
        = pad_and_convert_to_numpy(tokens, masked_positions,
                                   masked_labels, pad_id, max_seq_length,
                                   max_seq_length_dec, masked_spans,
                                   bos_id, eos_id, sentinel_tokens)

    train_sample = {
        'text_enc': tokens_enc,
        'text_dec': tokens_dec_in,
        'labels': labels,
        'loss_mask': loss_mask,
        'truncated': int(truncated),
        'enc_mask': enc_mask,
        'dec_mask': dec_mask,
        'enc_dec_mask': enc_dec_mask,
    }
    return train_sample


def build_supervised_training_sample(input_sample, target_sample,
                                     max_seq_length, max_seq_length_dec,
                                     pad_id, bos_id=None, eos_id=None,
                                     sentinel_tokens=None):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
        bos_id: start of decoder example id
        eos_id: end of generation id
        sentinel_tokens: unique value to be substituted for every replaced span
    """

    # flatten sentences into one list
    if isinstance(input_sample[0], list):
        input_tokens = [token for sentence in input_sample for token in sentence]
    else:
        input_tokens = input_sample
    if target_sample is not None:
        if isinstance(target_sample[0], list):
            target_tokens = [token for sentence in target_sample for token in sentence]
        else:
            target_tokens = target_sample

    # Truncate to `max_seq_length`.
    input_max_num_tokens = max_seq_length
    input_truncated = len(input_tokens) > input_max_num_tokens
    input_tokens = input_tokens[:input_max_num_tokens]

    if target_sample is not None:
        target_max_num_tokens = max_seq_length_dec - 2
        target_truncated = len(target_tokens) > target_max_num_tokens
        target_tokens = target_tokens[:target_max_num_tokens]

    # Padding.
    if target_sample is not None:
        (
            tokens_enc,
            tokens_dec_in,
            labels,
            enc_mask,
            dec_mask,
            enc_dec_mask,
            loss_mask,
        ) = pad_and_convert_to_numpy(
            input_tokens,
            [],
            [],
            pad_id,
            max_seq_length,
            max_seq_length_dec,
            target_tokens,
            [],
            bos_id,
            eos_id,
            sentinel_tokens,
        )
        train_sample = {
            "text_enc": tokens_enc,
            "text_dec": tokens_dec_in,
            "labels": labels,
            "loss_mask": loss_mask,
            "input_truncated": int(input_truncated),
            "target_truncated": int(target_truncated),
            "enc_mask": enc_mask,
            "dec_mask": dec_mask,
            "enc_dec_mask": enc_dec_mask,
        }
    else:
        # decoder only
        tokens_enc = pad_and_convert_to_numpy_inputs_only(
            input_tokens, pad_id, max_seq_length, eos_id
        )
        train_sample = {
            "text": tokens_enc,
        }
    return train_sample


def build_unpadded_sample(
    input_sample, max_seq_length, target_sample=None, max_seq_length_dec=None,
):
    # flatten sentences into one list
    input_tokens = [token for sentence in input_sample for token in sentence]
    if target_sample is not None:
        target_tokens = [token for sentence in target_sample for token in sentence]

    # Truncate to `max_seq_length`.
    input_max_num_tokens = max_seq_length
    input_tokens = input_tokens[:input_max_num_tokens]

    if target_sample is not None:
        target_max_num_tokens = max_seq_length_dec - 2
        target_tokens = target_tokens[:target_max_num_tokens]

    text_enc_key = "text_enc" if target_sample is not None else "text"
    train_sample = {
        text_enc_key: input_tokens,
    }
    if target_sample is not None:
        train_sample["text_dec"] = target_tokens
    return train_sample


def pad_and_convert_to_numpy_inputs_only(tokens, pad_id, max_seq_length, eos_id):
    num_tokens = len(tokens)
    if len(tokens) < max_seq_length:
        inputs = tokens + [eos_id]
        num_tokens += 1
    else:
        inputs = tokens
    # padding mask.
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    filler = [pad_id] * padding_length
    tokens_enc = np.array(inputs + filler, dtype=np.int64)
    return tokens_enc

def pad_and_convert_to_numpy(tokens, masked_positions,
                             masked_labels, pad_id,
                             max_seq_length, max_seq_length_dec,
                             decoder_tokens=None,
                             masked_spans=None, bos_id=None,
                             eos_id=None, sentinel_tokens=None):
    """Pad sequences and convert them to numpy."""

    sentinel_tokens = collections.deque(sentinel_tokens)
    t5_input = []
    (t5_decoder_in, t5_decoder_out) = ([bos_id], [])
    (start_index, end_index) = (0, None)

    if decoder_tokens is not None:
        assert len(masked_positions) == len(masked_labels) == 0
        t5_decoder_in.extend(decoder_tokens)
        t5_decoder_out.extend(decoder_tokens)
    else:
        for span in masked_spans:
            flag = sentinel_tokens.popleft()

            # Append the same tokens in decoder input and output
            t5_decoder_in.append(flag)
            t5_decoder_in.extend(span.label)
            t5_decoder_out.append(flag)
            t5_decoder_out.extend(span.label)

            end_index = span.index[0]
            t5_input.extend(tokens[start_index:end_index])
            t5_input.append(flag)

            # the next start index is the token after the last span token
            start_index = span.index[-1] + 1

    # Add <eos> token to the t5_decoder_out
    t5_decoder_out.append(eos_id)

    # Add the remaining tokens to the t5 input
    t5_input.extend(tokens[start_index:])

    # assert (len(t5_input) - len(masked_spans)) + \
    #        (len(t5_decoder_in) - (len(masked_spans) + 1)) == len(tokens)

    # Some checks.

    # Encoder-side padding mask.
    num_tokens = len(t5_input)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(masked_positions) == len(masked_labels)

    # Tokens..
    filler = [pad_id] * padding_length
    tokens_enc = np.array(t5_input + filler, dtype=np.int64)

    # Decoder-side padding mask.
    num_tokens_dec = len(t5_decoder_in)
    padding_length_dec = max_seq_length_dec - num_tokens_dec
    assert padding_length_dec >= 0
    filler_dec = [pad_id] * padding_length_dec
    tokens_dec_in = np.array(t5_decoder_in + filler_dec, dtype=np.int64)

    # Create attention masks
    enc_mask = make_attention_mask(tokens_enc, tokens_enc)
    enc_dec_mask = make_attention_mask(tokens_dec_in, tokens_enc)
    dec_mask = make_attention_mask(tokens_dec_in, tokens_dec_in)
    dec_mask = dec_mask * make_history_mask(tokens_dec_in)

    # Labels mask.
    labels = t5_decoder_out + ([-1] * padding_length_dec)
    labels = np.array(labels, dtype=np.int64)

    # Loss mask
    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
    loss_mask = np.array(loss_mask, dtype=np.int64)

    return tokens_enc, tokens_dec_in, labels, enc_mask, \
           dec_mask, enc_dec_mask, loss_mask


def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask


def make_attention_mask_3d(source_block, target_block):
    """
    Returns a 3-dimensional (3-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[:, None, :] >= 1) * (source_block[:, :, None] >= 1)
    # (batch, source_length, target_length)
    # mask = mask.astype(np.int64)
    return mask


def make_history_mask(block):
    length = block.shape[0]
    arange = np.arange(length)
    history_mask = (arange[None, ] <= arange[:, None])
    history_mask = history_mask.astype(np.int64)
    return history_mask


def make_history_mask_3d(block):
    batch, length = block.shape
    arange = torch.arange(length, device=block.device)
    history_mask = (arange[None, ] <= arange[:, None])[None, ]
    history_mask = history_mask.expand(batch, length, length)
    return history_mask
