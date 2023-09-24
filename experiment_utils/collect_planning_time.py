import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, required=True, help="Path to the experiment sub-directory, e.g., ../experiments/best_throughput")
parser.add_argument("--output_file", type=str, help="Path prefix to the output files. Two files will be generated: "
                                                           "<prefix>_memory_estimated.csv, <prefix>_memory_actual.csv. "
                                                            "Default to exp dir name + planning_time.csv .")

args = parser.parse_args()
assert os.path.isdir(args.exp_dir)
if args.output_file is None:
    args.output_file = args.exp_dir.rstrip("/") + "_planning_time.csv"

with open(args.output_file, 'w') as f:
    f.write("exp_name,esl,dsl,gbs,iteration,time\n")
    for exp_name in os.listdir(args.exp_dir):
        exp_full_path = os.path.join(args.exp_dir, exp_name)
        if "dynapipe" not in exp_name or not os.path.isdir(exp_full_path):
            continue
        for spec_name in os.listdir(exp_full_path):
            if "t5" in exp_name:
                esl = int(spec_name.split('_')[3][5:])
                dsl = int(spec_name.split('_')[4][5:])
                gbs = int(spec_name.split('_')[5][3:])
            else:
                esl = int(spec_name.split('_')[3][2:])
                dsl = 0
                gbs = int(spec_name.split('_')[4][3:])
            # read log file to determine if it is a successful run
            stdout_stderr_fn = "stdout_stderr.log"
            if not os.path.isfile(os.path.join(exp_full_path, spec_name, stdout_stderr_fn)):
                continue
            with open(os.path.join(exp_full_path, spec_name, stdout_stderr_fn), 'r') as log_file:
                if not "Training finished successfully." in log_file.read():
                    continue
            preprocessing_dir = os.path.join(exp_full_path, spec_name, "dynapipe_logs", "preprocessing")
            if not os.path.isdir(preprocessing_dir):
                continue
            for fn in os.listdir(preprocessing_dir):
                if fn.endswith(".log"):
                    time_dict = defaultdict(float)
                    with open(os.path.join(preprocessing_dir, fn), 'r') as log_file:
                        for line in log_file:
                            if "Micro-batch generation" in line:
                                l = line.split("Micro-batch generation for iteration")[1].strip()
                                iteration = int(l.split()[0])
                                time = float(l.split()[2])
                                time_dict[iteration] += time
                            elif "Schedule generation for DP group" in line:
                                l = line.split("Schedule generation for DP group")[1].strip()
                                iteration = int(l.split()[2])
                                time = float(l.split()[4])
                                time_dict[iteration] += time
                    for iteration, time in time_dict.items():
                        f.write(f"{exp_name},{esl},{dsl},{gbs},{iteration},{time}\n")

