import os
import sys
import argparse
from dataclasses import dataclass, asdict
import jsonlines
import numpy as np

# copied from run_experiment.py
RC_MAP = {
    "none": 0,
    "selective": 1,
    "full": 2,
}


@dataclass(eq=True)
class ExperimentConfig:
    enc_seqlen: int = 0
    dec_seqlen: int = 0
    gbs: int = 0
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    mbs: int = 1
    rc: str = "none"
    ds_level: int = 0
    spp: bool = False
    status: str = "unknown"

    def speed_dominates(self, other):
        assert isinstance(
            other, ExperimentConfig
        ), "Can only compare with ExperimentConfig"
        # Config A speed_dominates config B if A is almost surely faster than B
        # if sequence length, gbs or parallelism are different, no dominance
        if (
            self.enc_seqlen != other.enc_seqlen
            or self.dec_seqlen != other.dec_seqlen
            or self.gbs != other.gbs
            or self.dp_size != other.dp_size
            or self.tp_size != other.tp_size
            or self.pp_size != other.pp_size
            or self.spp != other.spp
        ):
            return False
        # dominance happens if ds_level is lower, and mbs is higher,
        # and rc level is lower
        # Note: we only test the lowest rc level that can run to reduce
        # grid search time
        if (
            RC_MAP[self.rc] < RC_MAP[other.rc] or
            (RC_MAP[self.rc] == RC_MAP[other.rc] and
            (self.ds_level <= other.ds_level) and
            (self.mbs >= other.mbs and self.pp_size == 1))
        ):
            return True
        return False

    def memory_dominates(self, other):
        assert isinstance(
            other, ExperimentConfig
        ), "Can only compare with ExperimentConfig"
        # Config A memory_dominates config B if A is almost surely
        # consumes more memory than B
        # if gbs or parallelism are different, no dominance
        if (
            self.gbs != other.gbs
            or self.dp_size != other.dp_size
            or self.tp_size != other.tp_size
            or self.pp_size != other.pp_size
            or self.spp != other.spp
        ):
            return False
        # dominance happens if sequence length is higher, mbs is higher,
        # rc level is lower, and ds_level is lower
        if (
            self.enc_seqlen >= other.enc_seqlen
            and self.dec_seqlen >= other.dec_seqlen
            and self.mbs >= other.mbs
            and self.ds_level <= other.ds_level
            and RC_MAP[self.rc] <= RC_MAP[other.rc]
        ):
            return True
        return False

    @staticmethod
    def parse_experiment_status(exp_dir):
        log_path = os.path.join(exp_dir, "stdout_stderr.log")
        if not os.path.exists(log_path):
            return "unknown"
        with open(log_path, "r") as f:
            contents = f.read()
        if ("after training is done" in contents or 
            "Taking poison pill..." in contents or 
            "Training finished successfully." in contents):
            return "success"
        else:
            return "failure"

    @staticmethod
    def parse_history_experiments(exp_dir):
        exp_spec = os.path.basename(os.path.normpath(exp_dir))
        config_items = exp_spec.split("_")
        config = ExperimentConfig()
        for item in config_items:
            if item.startswith("dp"):
                config.dp_size = int(item[2:])
            elif item.startswith("tp"):
                config.tp_size = int(item[2:])
            elif item.startswith("pp"):
                config.pp_size = int(item[2:])
            elif item.startswith("sl"):
                config.enc_seqlen = int(item[2:])
            elif item.startswith("encsl"):
                config.enc_seqlen = int(item[5:])
            elif item.startswith("decsl"):
                config.dec_seqlen = int(item[5:])
            elif item.startswith("gbs"):
                config.gbs = int(item[3:])
            elif item.startswith("mbs"):
                config.mbs = int(item[3:])
            elif item.startswith("rc"):
                config.rc = item[2:]
            elif item.startswith("zero"):
                config.ds_level = int(item[4:])
            elif item.startswith("spp"):
                config.spp = True
        # test status
        config.status = ExperimentConfig.parse_experiment_status(exp_dir)
        return config

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

def parse_exp_logs(enc_seqlen, dec_seqlen, gbs, out_file, export_all_exps=False):
    subdirs = [o for o in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir,o))]
    filtered_subdirs = []
    for subdir in subdirs:
        exp_config = ExperimentConfig.parse_history_experiments(os.path.join(args.dir, subdir))
        if exp_config.status != "success":
            continue
        if exp_config.enc_seqlen == enc_seqlen and exp_config.dec_seqlen == dec_seqlen and exp_config.gbs == gbs:
            filtered_subdirs.append(subdir)

    best_time = float("inf")
    best_config = None
    all_exps = []
    for matched_dir in filtered_subdirs:
        fn = os.path.join(args.dir, matched_dir, "stdout_stderr.log")
        if os.path.exists(fn):
            iter_time = get_iter_time(fn)
            if export_all_exps:
                exp_config = ExperimentConfig.parse_history_experiments(os.path.join(args.dir, matched_dir))
                log_json = {
                    "Encoder SeqLen": enc_seqlen,
                    "Decoder SeqLen": dec_seqlen,
                    "Global Batch Size": gbs,
                    "Iteration Time (ms)": iter_time,
                    "Config": asdict(exp_config),
                }
                all_exps.append(log_json)
            else:
                if iter_time < best_time:
                    best_time = iter_time
                    exp_config = ExperimentConfig.parse_history_experiments(os.path.join(args.dir, matched_dir))
                    best_config = exp_config
    if best_config is None and len(all_exps) == 0:
        print("No successful experiment found for enc seqlen {}, dec seqlen {}, gbs {}".format(enc_seqlen, dec_seqlen, gbs))
        return
    if not export_all_exps:
        log_json = {
            "Encoder SeqLen": enc_seqlen,
            "Decoder SeqLen": dec_seqlen,
            "Global Batch Size": gbs,
            "Best Time (ms)": best_time,
            "Best Config": asdict(best_config),
        }
        with jsonlines.open(out_file, mode='a') as writer:
            writer.write(log_json)
    else:
        with jsonlines.open(out_file, mode='a') as writer:
            writer.write_all(all_exps)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--dir", type=str, required=True)
parser.add_argument("-e", "--enc_seqlen", nargs="+", type=int)
parser.add_argument("-d", "--dec_seqlen", nargs="+", type=int)
parser.add_argument("-g", "--gbs", nargs="+", type=int)
parser.add_argument("-a", "--all", action="store_true", default=False,
                    help="Parse all experiments in the directory instead of only the best ones.")
parser.add_argument("-o", "--out", type=str)

args = parser.parse_args()

if not args.enc_seqlen:
    print("Encoder seqlen not set, parsing all files.")
    enc_seqlens = set()
    dec_seqlens = set()
    gbs = set()
    for fn in os.listdir(args.dir):
        path = os.path.join(args.dir, fn)
        if os.path.isdir(path):
            config = ExperimentConfig.parse_history_experiments(path)
            if config.status == "success":
                gbs.add(config.gbs)
                enc_seqlens.add(config.enc_seqlen)
                dec_seqlens.add(config.dec_seqlen)
    args.enc_seqlen = sorted(list(enc_seqlens))
    args.dec_seqlen = sorted(list(dec_seqlens))
    args.gbs = sorted(list(gbs))
else:
    if not args.dec_seqlen:
        args.dec_seqlen = [0]

if not args.out:
    exp_name = os.path.basename(os.path.normpath(args.dir))
    if args.all:
        exp_name += "_all_specs"
    args.out = exp_name + ".jsonl"

print("Collecting logs for enc_seqlen range: {}, dec_seqlen range: {}, gbs range: {}".format(args.enc_seqlen, args.dec_seqlen, args.gbs))
print("Writing to", args.out)

if os.path.exists(args.out):
    print("ERROR: output file exists. Aborting.")
    sys.exit(1)

for enc_seqlen in args.enc_seqlen:
    for dec_seqlen in args.dec_seqlen:
        for gbs in args.gbs:
            parse_exp_logs(enc_seqlen, dec_seqlen, gbs, args.out, args.all)