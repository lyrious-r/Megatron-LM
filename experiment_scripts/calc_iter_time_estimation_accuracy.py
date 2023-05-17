import argparse
import os
import json
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--exp_dir', type=str, default='./experiments')
parser.add_argument('-eo', '--est_out', type=str, default='./experiment_results/estimated_iter_time.csv')
parser.add_argument('-ao', '--act_out', type=str, default='./experiment_results/all_iter_times.csv')

args = parser.parse_args()

def get_iter_time(fn: str):
    times = []
    with open(fn, "r") as f:
        for line in f:
            if "elapsed time per iteration (ms):" in line:
                elapsed = float(line.split("elapsed time per iteration (ms):")[1].split()[0].strip())
                times.append(elapsed)
            if "CUDA out of memory" in line:
                return float("inf")
    if len(times) == 0:
        return float("inf")
    return np.mean(times[2:]) # skip first 20 iteration as warmup

with open(args.est_out, 'w') as est_f:
    est_f.write("exp_name,spec_name,dpg,iteration,estimated_time\n")
    with open(args.act_out, 'w') as act_f:
        act_f.write("exp_name,spec_name,measured_time\n")
        for exp_name in tqdm(os.listdir(args.exp_dir), desc="Experiments"):
            exp_full_path = os.path.join(args.exp_dir, exp_name)
            if "plopt" not in exp_name or not os.path.isdir(exp_full_path) or "bug" in exp_name:
                continue
            for spec_name in tqdm(os.listdir(exp_full_path), desc="Specs", leave=False):
                # read log file to determine if it is a successful run
                stdout_stderr_fn = "stdout_stderr.log"
                log_path = os.path.join(exp_full_path, spec_name, stdout_stderr_fn)
                if not os.path.isfile(log_path):
                    continue
                with open(log_path, 'r') as log_file:
                    if not "Training finished successfully." in log_file.read():
                        continue
                # try to get the measured time
                measured_time = get_iter_time(log_path)
                if measured_time != float("inf"):
                    act_f.write("{},{},{}\n".format(exp_name, spec_name, measured_time))
                # get estimated time by reading simulated traces
                ep_stats_dir = os.path.join(exp_full_path, spec_name, "plopt_ep_stats")
                traces_dir = os.path.join(ep_stats_dir, "per_iter_simulated_traces")
                if not os.path.isdir(traces_dir):
                    continue
                for fn in os.listdir(traces_dir):
                    if fn.endswith(".json"):
                        dpg = int(fn.split("_")[0][3:])
                        iteration = int(fn.split("_")[1].split(".")[0])
                        with open(os.path.join(traces_dir, fn), 'r') as trace_json:
                            json_data = json.load(trace_json)
                            # get the end time of the last op
                            latest_end_time = 0
                            for ev in json_data["traceEvents"]:
                                if ev["ph"] == "X":
                                    latest_end_time = max(latest_end_time, ev["ts"] + ev["dur"])
                            est_f.write("{},{},{},{},{}\n".format(exp_name, spec_name, dpg, iteration, latest_end_time / 1e3))


