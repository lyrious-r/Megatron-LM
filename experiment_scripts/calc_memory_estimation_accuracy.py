import argparse
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--exp_dir', type=str, default='./experiments')
parser.add_argument('-eo', '--est_out', type=str, default='./experiment_results/estimated_memory.csv')
parser.add_argument('-ao', '--act_out', type=str, default='./experiment_results/actual_memory.csv')

args = parser.parse_args()

with open(args.est_out, 'w') as est_f:
    est_f.write("exp_name,spec_name,dpg,iteration,memory\n")
    with open(args.act_out, 'w') as act_f:
        act_f.write("exp_name,spec_name,dr,pr,tr,iteration,memory\n")
        for exp_name in tqdm(os.listdir(args.exp_dir), desc="Experiments"):
            exp_full_path = os.path.join(args.exp_dir, exp_name)
            if "dynapipe" not in exp_name or not os.path.isdir(exp_full_path) or "bug" in exp_name:
                continue
            for spec_name in tqdm(os.listdir(exp_full_path), desc="Specs", leave=False):
                # read log file to determine if it is a successful run
                stdout_stderr_fn = "stdout_stderr.log"
                if not os.path.isfile(os.path.join(exp_full_path, spec_name, stdout_stderr_fn)):
                    continue
                with open(os.path.join(exp_full_path, spec_name, stdout_stderr_fn), 'r') as log_file:
                    if not "Training finished successfully." in log_file.read():
                        continue
                # get estimated memory usage
                ep_stats_dir = os.path.join(exp_full_path, spec_name, "dynapipe_ep_stats")
                estimated_memory_dir = os.path.join(ep_stats_dir, "estimated_memory")
                if not os.path.isdir(estimated_memory_dir):
                    continue
                for fn in os.listdir(estimated_memory_dir):
                    if fn.endswith(".txt"):
                        with open(os.path.join(estimated_memory_dir, fn), 'r') as log_file:
                            for line in log_file:
                                dpg, iteration, memory = line.split(":")
                                dpg = int(dpg.strip())
                                iteration = int(iteration.strip())
                                memory = float(memory.strip())
                                est_f.write("{},{},{},{},{}\n".format(exp_name, spec_name, dpg, iteration, memory))
                # get actual memory usage
                memory_stats_dir = os.path.join(exp_full_path, spec_name, "dynapipe_memory_stats")
                if not os.path.isdir(memory_stats_dir):
                    assert False, f"memory_stats_dir {memory_stats_dir} does not exist"
                    continue
                for subdir in tqdm(os.listdir(memory_stats_dir), desc="Subdirs", leave=False):
                    dr = int(subdir.split("_")[0][2:])
                    pr = int(subdir.split("_")[1][2:])
                    tr = int(subdir.split("_")[2][2:])
                    subdir_path = os.path.join(memory_stats_dir, subdir)
                    # we have to look at the microbatch stats, since we 
                    # zeroed out the memory stats for each microbatch
                    mbstats_dir = os.path.join(subdir_path, "microbatch_stats")
                    max_mem_per_iter = {}
                    for fn in tqdm(os.listdir(mbstats_dir), desc="Instructions", leave=False):
                        if fn.endswith(".txt"):
                            with open(os.path.join(mbstats_dir, fn), 'r') as log_file:
                                # it is actually a json file
                                iteration = int(fn.split("_")[0][4:])
                                memory_json = json.load(log_file)
                                peak_memory = memory_json["peak_allocated_memory"] / 1e6 # convert to MB
                                if iteration not in max_mem_per_iter or peak_memory > max_mem_per_iter[iteration]:
                                    max_mem_per_iter[iteration] = peak_memory
                    for iteration, peak_memory in max_mem_per_iter.items():
                        act_f.write("{},{},{},{},{},{},{}\n".format(exp_name, spec_name, dr, pr, tr, iteration, peak_memory))


