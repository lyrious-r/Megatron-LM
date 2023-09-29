import os
import argparse

import pandas as pd

from exp_name_utils import augment_df

parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str, required=True, 
                    help='Path to the grid search data jsonl, generated with '
                    'collect_throughput_stats.py')
parser.add_argument('--config_dir', type=str, required=True,
                    help='Path to the directory to store the control configs.')

args = parser.parse_args()

df = pd.read_json(args.input_data, lines=True)
df = augment_df(df)

# for each experiment, get the one with lowest avg iter time
best_df = df.loc[df.groupby(["exp_name", "seqlen", "global_batch_size"])["avg_iter_time"].idxmin()]

# dict of (model_size, seqlen, global_batch_size) -> (tp_degree, pp_degree)
best_parallelism_for_dynapipe = {}
for index, row in best_df.iterrows():
    model_size = row["model_size"]
    seqlen = row["seqlen"]
    global_batch_size = row["global_batch_size"]
    tp_degree = row["tp_degree"]
    pp_degree = row["pp_degree"]
    best_parallelism_for_dynapipe[(model_size, seqlen, global_batch_size)] = (tp_degree, pp_degree)

def control_filter(row):
    framework = row["framework"]
    if framework != "baseline":
        return False
    model_size = row["model_size"]
    seqlen = row["seqlen"]
    global_batch_size = row["global_batch_size"]
    tp_degree = row["tp_degree"]
    pp_degree = row["pp_degree"]
    if (model_size, seqlen, global_batch_size) in best_parallelism_for_dynapipe:
        return (tp_degree, pp_degree) == best_parallelism_for_dynapipe[(model_size, seqlen, global_batch_size)]
    else:
        return False

control_df = df[df.apply(control_filter, axis=1)].copy()
# select the best config for each experiment
control_df = control_df.loc[control_df.groupby(["exp_name", "seqlen", "global_batch_size"])["avg_iter_time"].idxmin()]

# transform the df into best config format
def get_micro_batch_size(row):
    if "mbs" in row["spec_name"]:
        mbs = int(row["spec_name"].split("mbs")[1].split("_")[0])
    else:
        mbs = 1 # unused
    return mbs

control_df["encoder_seq_length"] = control_df["seqlen"]
control_df["decoder_seq_length"] = 0
control_df.loc[control_df["model"] == "T5", "decoder_seq_length"] = control_df[control_df["model"] == "T5"]["seqlen"]
control_df["seq_length"] = control_df["seqlen"]
control_df["tokens_per_global_batch"] = control_df["global_batch_size"]
control_df["tensor_parallel_size"] = control_df["tp_degree"]
control_df["pipeline_parallel_size"] = control_df["pp_degree"]
control_df["micro_batch_size"] = control_df.apply(get_micro_batch_size, axis=1)
control_df["recompute_level"] = control_df["rc"]
control_df["enable_deepspeed"] = control_df["zero_degree"] > 0
control_df["deepspeed_zero_stage"] = control_df["zero_degree"]

if not os.path.exists(args.config_dir):
    os.makedirs(args.config_dir)
for exp_name in control_df["exp_name"].unique():
    exp_df = control_df[control_df["exp_name"] == exp_name]
    if exp_name.endswith("_grid"):
        exp_name = exp_name[:-5]
    exp_df = exp_df[["encoder_seq_length", "decoder_seq_length", "seq_length", "tokens_per_global_batch", "tensor_parallel_size", "pipeline_parallel_size", "micro_batch_size", "recompute_level", "enable_deepspeed", "deepspeed_zero_stage"]]
    exp_df.to_json(os.path.join(args.config_dir, f"{exp_name}.jsonl"), orient="records", lines=True)
