import argparse
import os
import numpy as np
import re
from datetime import datetime
import jsonlines


parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default="./experiments")
parser.add_argument("--output_file", type=str, default="./experiment_results/final_exp_stats.jsonl")

args = parser.parse_args()

def get_num_tokens(enc_mapping, dec_mapping, max_seqlen, model_type="gpt"):
    enc_seqlens = enc_mapping[:, 2]
    dec_seqlens = dec_mapping[:, 2]
    if model_type == "gpt":
        seqlens = enc_seqlens + dec_seqlens
        return sum(np.clip(seqlens, 0, max_seqlen))
    else:
        enc_tokens = np.sum(np.clip(enc_seqlens, 0, max_seqlen))
        dec_tokens = np.sum(np.clip(dec_seqlens, 0, max_seqlen))
        return enc_tokens + dec_tokens

def find_last_datetime_in_file(filename):
    datetime_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"
    last_datetime_str = ""
    with open(filename, 'r') as file:
        for line in file:
            matches = re.findall(datetime_pattern, line)
            if matches:
                last_datetime_str = matches[-1]
    if last_datetime_str:
        return datetime.strptime(last_datetime_str, '%Y-%m-%d %H:%M:%S,%f')
    else:
        return None


# for each experiment, we get:
# 1. the total number of tokens in the dataset
# 2. Wall time for training (just for reference)
# 3. Avg. iteration time
# 4. Number of iterations executed
with jsonlines.open(args.output_file, mode='w') as writer:
    for exp_name in os.listdir(args.exp_dir):
        if exp_name.endswith("best"):
            exp_full_path = os.path.join(args.exp_dir, exp_name)
            for spec_name in os.listdir(exp_full_path):
                log_file = os.path.join(exp_full_path, spec_name, "stdout_stderr.log")
                if "gpt" in exp_name:
                    seqlen = int(spec_name.split("_")[3][2:])
                else:
                    seqlen = int(spec_name.split("_")[3][5:])
                assert os.path.exists(log_file)
                start_dt = None
                end_dt = None
                total_tokens = None
                per_iter_times = []
                max_iter = -1
                with open(log_file, "r") as f:
                    for line in f:
                        if "> loading indexed mapping from" in line and total_tokens is None:
                            enc_mapping_path = line.split(" ")[-3].strip()
                            dec_mapping_path = line.split(" ")[-1].strip()
                            enc_mapping = np.load(enc_mapping_path)
                            dec_mapping = np.load(dec_mapping_path)
                            enc_mapping = enc_mapping[:100000]
                            dec_mapping = dec_mapping[:100000]
                            total_tokens = get_num_tokens(enc_mapping, dec_mapping, seqlen, model_type=exp_name.split("_")[0])
                        elif "[before the start of training step]" in line:
                            # start wall time
                            datetime_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
                            dt_string = re.findall(datetime_pattern, line)[0]
                            start_dt = datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
                        elif "[after training is done]" in line:
                            # end wall time
                            datetime_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
                            dt_string = re.findall(datetime_pattern, line)[0]
                            end_dt = datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
                        elif "elapsed time per iteration (ms):" in line:
                            elapsed = float(line.split("elapsed time per iteration (ms):")[1].split()[0].strip())
                            per_iter_times.append(elapsed)
                        elif "Running iteration" in line:
                            # iteration number
                            iter_num = int(line.split("Running iteration")[1].split("...")[0].strip())
                            max_iter = max(max_iter, iter_num)
                if end_dt is None:
                    # use the last iteration time to estimate the end time
                    end_dt = find_last_datetime_in_file(log_file)
                result_json = {
                    "exp_name": exp_name,
                    "spec_name": spec_name,
                    "num_tokens": int(total_tokens),
                    "avg_iter_time": float(np.mean(per_iter_times[2:])),
                    "num_iters": max_iter,
                    "start_time": start_dt.timestamp()*1000,
                    "end_time": end_dt.timestamp()*1000
                }
                writer.write(result_json)







