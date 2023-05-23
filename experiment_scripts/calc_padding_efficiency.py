import argparse
import pickle
import numpy as np
import jsonlines
import os

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default="./experiments")
parser.add_argument("--output_file", type=str, default="./experiment_results/padding_efficiency.jsonl")

args = parser.parse_args()


def get_batching_efficiency(max_enc_seqlen, max_dec_seqlen, dir):
    if "plopt" in dir:
        # read from plopt log
        per_iter_mb_shapes = []
        per_iter_seqlens = []
        seqlens_prefix = os.path.join(dir, "plopt_ep_stats", "orig_seq_lens")
        fns = sorted(os.listdir(seqlens_prefix), key=lambda x: int(x.split(".")[0].split("_")[1]))
        for fn in fns:
            with open(os.path.join(seqlens_prefix, fn), "rb") as f:
                input_seqlens, target_seqlens = pickle.load(f)
            per_iter_seqlens.append((input_seqlens, target_seqlens))
        mb_shapes_prefix = os.path.join(dir, "plopt_ep_stats", "per_iter_mb_shapes")
        fns = sorted(os.listdir(mb_shapes_prefix), key=lambda x: int(x.split(".")[0].split("_")[1]))
        for fn in fns:
            iteration = int(fn.split(".")[0].split("_")[1])
            while len(per_iter_mb_shapes) <= iteration:
                per_iter_mb_shapes.append([])
            with open(os.path.join(mb_shapes_prefix, fn), "rb") as f:
                mb_shapes = pickle.load(f)
            per_iter_mb_shapes[iteration] += mb_shapes
        # compute efficiency
        n_iters = min(len(per_iter_mb_shapes), len(per_iter_seqlens))
        enc_effs = []
        dec_effs = []
        for i in range(n_iters):
            truncated_enc_seqlens = np.minimum(per_iter_seqlens[i][0], max_enc_seqlen)
            truncated_dec_seqlens = np.minimum(per_iter_seqlens[i][1], max_dec_seqlen)
            all_enc_tokens = truncated_enc_seqlens.sum()
            all_dec_tokens = truncated_dec_seqlens.sum()
            all_microbatch_enc_tokens = 0
            all_microbatch_dec_tokens = 0
            for mb in per_iter_mb_shapes[i]:
                all_microbatch_enc_tokens += mb[0] * mb[1]
                all_microbatch_dec_tokens += mb[0] * mb[2]
            if all_microbatch_enc_tokens == 0:
                # missing data
                continue
            enc_effs.append(all_enc_tokens / all_microbatch_enc_tokens)
            if all_microbatch_dec_tokens != 0:
                dec_effs.append(all_dec_tokens / all_microbatch_dec_tokens)
            else:
                dec_effs.append(0)
        return np.mean(enc_effs), np.mean(dec_effs)
    else:
        # directly parse from log
        with open(os.path.join(dir, "stdout_stderr.log"), "r") as f:
            max_samples = 0
            result_enc_eff = 0
            result_dec_eff = 0
            for line in f:
                if line.startswith(">>>> Pack samples:"):
                    splitted_line = line.replace("/", " ").replace(",", " ").split(" ")
                    all_numbers = []
                    for token in splitted_line:
                        try:
                            all_numbers.append(float(token))
                        except:
                            pass
                    nsamples, _, _, enc_eff, dec_eff = all_numbers
                    if nsamples > max_samples:
                        result_enc_eff = enc_eff
                        result_dec_eff = dec_eff
                        max_samples = nsamples
            return result_enc_eff, result_dec_eff

with jsonlines.open(args.output_file, "w") as writer:
    for exp_name in os.listdir(args.exp_dir):
        if "bug" in exp_name or "abl" in exp_name:
            continue
        if not os.path.isdir(os.path.join(args.exp_dir, exp_name)):
            continue
        for spec_name in os.listdir(os.path.join(args.exp_dir, exp_name)):
            spec_path = os.path.join(args.exp_dir, exp_name, spec_name)
            if "plopt" in exp_name:
                # read from plopt log
                per_iter_mb_shapes = []
                per_iter_seqlens = []
                seqlens_prefix = os.path.join(spec_path, "plopt_ep_stats", "orig_seq_lens")
                if "t5" in exp_name:
                    seqlen = int(spec_name.split("_")[3][5:])
                else:
                    seqlen = int(spec_name.split("_")[3][2:])
                if not os.path.exists(seqlens_prefix):
                    continue
                fns = sorted(os.listdir(seqlens_prefix), key=lambda x: int(x.split(".")[0].split("_")[1]))
                for fn in fns:
                    with open(os.path.join(seqlens_prefix, fn), "rb") as f:
                        input_seqlens, target_seqlens = pickle.load(f)
                    per_iter_seqlens.append((input_seqlens, target_seqlens))
                mb_shapes_prefix = os.path.join(spec_path, "plopt_ep_stats", "per_iter_mb_shapes")
                fns = sorted(os.listdir(mb_shapes_prefix), key=lambda x: int(x.split(".")[0].split("_")[1]))
                for fn in fns:
                    iteration = int(fn.split(".")[0].split("_")[1])
                    while len(per_iter_mb_shapes) <= iteration:
                        per_iter_mb_shapes.append([])
                    with open(os.path.join(mb_shapes_prefix, fn), "rb") as f:
                        mb_shapes = pickle.load(f)
                    per_iter_mb_shapes[iteration] += mb_shapes
                # compute efficiency
                n_iters = min(len(per_iter_mb_shapes), len(per_iter_seqlens))
                enc_tokens = 0
                enc_padded_tokens = 0
                dec_tokens = 0
                dec_padded_tokens = 0
                for i in range(n_iters):
                    truncated_enc_seqlens = np.minimum(per_iter_seqlens[i][0], seqlen)
                    truncated_dec_seqlens = np.minimum(per_iter_seqlens[i][1], seqlen)
                    all_enc_tokens = truncated_enc_seqlens.sum()
                    all_dec_tokens = truncated_dec_seqlens.sum()
                    all_microbatch_enc_tokens = 0
                    all_microbatch_dec_tokens = 0
                    for mb in per_iter_mb_shapes[i]:
                        all_microbatch_enc_tokens += mb[0] * mb[1]
                        all_microbatch_dec_tokens += mb[0] * mb[2]
                    if all_microbatch_enc_tokens == 0:
                        # missing data
                        continue
                    enc_tokens += all_enc_tokens
                    enc_padded_tokens += all_microbatch_enc_tokens
                    if all_microbatch_dec_tokens != 0:
                        dec_tokens += all_dec_tokens
                        dec_padded_tokens += all_microbatch_dec_tokens
                obj = {
                    "exp_name": exp_name,
                    "spec_name": spec_name,
                    "enc_tokens": int(enc_tokens),
                    "dec_tokens": int(dec_tokens),
                    "enc_padded_tokens": int(enc_padded_tokens),
                    "dec_paddded_tokens": int(dec_padded_tokens),
                }
                writer.write(obj)
            else:
                # directly parse from log
                with open(os.path.join(spec_path, "stdout_stderr.log"), "r") as f:
                    max_samples = 0
                    result_enc_eff = 0
                    result_dec_eff = 0
                    for line in f:
                        if line.startswith(">>>> Pack samples:"):
                            splitted_line = line.replace("/", " ").replace(",", " ").split(" ")
                            all_numbers = []
                            for token in splitted_line:
                                try:
                                    all_numbers.append(float(token))
                                except:
                                    pass
                            nsamples, _, _, enc_eff, dec_eff = all_numbers
                            if nsamples > max_samples:
                                result_enc_eff = enc_eff
                                result_dec_eff = dec_eff
                                max_samples = nsamples
                    obj = {
                        "exp_name": exp_name,
                        "spec_name": spec_name,
                        "enc_eff": result_enc_eff,
                        "dec_eff": result_dec_eff,
                    }
                    writer.write(obj)
