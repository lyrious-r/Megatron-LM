import os
import argparse

import pandas as pd

from exp_name_utils import augment_df

parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str, required=True, 
                    help='Path to the input data jsonl, generated with '
                    'collect_throughput_stats.py')
parser.add_argument('--config_dir', type=str, required=True,
                    help='Path to the directory to store the best configs.')

args = parser.parse_args()

df = pd.read_json(args.input_data, lines=True)
df = augment_df(df)

# for each experiment, get the one with lowest avg iter time
best_df = df.loc[df.groupby(["exp_name", "seqlen", "global_batch_size"])["avg_iter_time"].idxmin()]
# transform the df into best config format
def get_micro_batch_size(row):
    if "mbs" in row["spec_name"]:
        mbs = int(row["spec_name"].split("mbs")[1].split("_")[0])
    else:
        mbs = 1 # unused
    return mbs

best_df["encoder_seq_length"] = best_df["seqlen"]
best_df["decoder_seq_length"] = 0
best_df.loc[best_df["model"] == "T5", "decoder_seq_length"] = best_df[best_df["model"] == "T5"]["seqlen"]
best_df["seq_length"] = best_df["seqlen"]
best_df["tokens_per_global_batch"] = best_df["global_batch_size"]
best_df["tensor_parallel_size"] = best_df["tp_degree"]
best_df["pipeline_parallel_size"] = best_df["pp_degree"]
best_df["micro_batch_size"] = best_df.apply(get_micro_batch_size, axis=1)
best_df["recompute_level"] = best_df["rc"]
best_df["enable_deepspeed"] = best_df["zero_degree"] > 0
best_df["deepspeed_zero_stage"] = best_df["zero_degree"]

if not os.path.exists(args.config_dir):
    os.makedirs(args.config_dir)
for exp_name in best_df["exp_name"].unique():
    exp_df = best_df[best_df["exp_name"] == exp_name]
    if exp_name.endswith("_grid"):
        exp_name = exp_name[:-5]
    exp_df = exp_df[["encoder_seq_length", "decoder_seq_length", "seq_length", "tokens_per_global_batch", "tensor_parallel_size", "pipeline_parallel_size", "micro_batch_size", "recompute_level", "enable_deepspeed", "deepspeed_zero_stage"]]
    exp_df.to_json(os.path.join(args.config_dir, f"{exp_name}.jsonl"), orient="records", lines=True)
