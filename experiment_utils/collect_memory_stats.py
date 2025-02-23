import argparse
import os
import numpy as np
import re
from datetime import datetime
import jsonlines


parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, required=True, help="Path to the experiment sub-directory, e.g., ../experiments/best_throughput")
parser.add_argument("--output_file", type=str, help="Path to the output file, default to exp dir name + .jsonl")

args = parser.parse_args()

assert os.path.isdir(args.exp_dir)
if args.output_file is None:
    args.output_file = args.exp_dir.rstrip("/") + ".jsonl"
print("Writing results to {}".format(args.output_file))



# for each experiment, we get:
# 1. the total number of tokens in the dataset
# 2. Wall time for training (just for reference)
# 3. Avg. iteration time
# 4. Number of iterations executed
with jsonlines.open(args.output_file, mode='w') as writer:
    for exp_name in os.listdir(args.exp_dir):
        exp_full_path = os.path.join(args.exp_dir, exp_name)
        if os.path.isdir(exp_full_path):
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
                crash_iter = -1
                gmlake_opnum_iters = []
                mem_data = {}
                pattern = r"\[Rank (\d+)\] \(after (\d+) iterations\) memory \(MB\) \| allocated: ([\d\.]+) \| max allocated: ([\d\.]+) \| reserved: ([\d\.]+) \| max reserved: ([\d\.]+)"
                with open(log_file, "r") as f:
                    contents = f.read()
                    if ("after training is done" in contents or 
                        "Taking poison pill..." in contents or 
                        "Training finished successfully." in contents or 
                        "StopIteration" in contents):
                        # this experiment finished successfully
                        pass
                    else:
                        continue
                with open(log_file, "r") as f:
                    for line in f:
                        match = re.match(pattern, line)
                        if match:
                            rank = int(match.group(1))
                            iter_num = int(match.group(2))
                            allocated_mem = float(match.group(3))
                            max_allocated_mem = float(match.group(4))
                            reserved_mem = float(match.group(5))
                            max_reserved_mem = float(match.group(6))
                            max_iter = max(max_iter, iter_num)
                            if iter_num not in mem_data:
                                mem_data[iter_num] = {}
                                mem_data[iter_num][rank] = {
                                    "allocated": allocated_mem,
                                    "max_allocated": max_allocated_mem,
                                    "reserved": reserved_mem,
                                    "max_reserved": max_reserved_mem
                                }
                result_json = {
                    "exp_name": exp_name,
                    "spec_name": spec_name,
                    "num_iters": max_iter,
                    "mem_data": mem_data,
                }
                writer.write(result_json)







