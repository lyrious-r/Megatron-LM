import os
import argparse
import numpy as np

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
    return np.mean(times[1:]) # skip first iteration as warmup

def get_stats_from_dirname(fn: str):
    parts = fn.split("_")
    encsl = int(parts[0][5:])
    decsl = int(parts[1][5:])
    gbs = int(parts[2][3:])
    mbs = int(parts[3][3:])
    rc = parts[4][2:]
    zero_level = int(parts[5][4:])
    return encsl, decsl, gbs, mbs, rc, zero_level


parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str)
parser.add_argument("seqlen", type=int)
parser.add_argument("gbs", type=int)

args = parser.parse_args()

subdirs = [o for o in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir,o))]
filtered_subdirs = []
for subdir in subdirs:
    encsl, decsl, gbs, mbs, rc, zero_level = get_stats_from_dirname(subdir)
    if encsl == args.seqlen and decsl == args.seqlen and gbs == args.gbs:
        filtered_subdirs.append(subdir)

best_time = float("inf")
best_mbs = 0
best_rc = None
best_zero_level = None
for matched_dir in filtered_subdirs:
    fn = os.path.join(args.dir, matched_dir, "stdout_stderr.log")
    if os.path.exists(fn):
        iter_time = get_iter_time(fn)
        if iter_time < best_time:
            best_time = iter_time
            encsl, decsl, gbs, mbs, rc, zero_level = get_stats_from_dirname(matched_dir)
            best_mbs = mbs
            best_rc = rc
            best_zero_level = zero_level

if best_time == float("inf"):
    print("No valid runs found.")
else:
    print(f"Best time: {best_time:.3f} ms, mbs: {best_mbs}, rc: {best_rc}, zero_level: {best_zero_level}")