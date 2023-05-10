import os
import argparse

parser = argparse.ArgumentParser(description='Batch scp padding eff data to remote server')
parser.add_argument('-r', "--remote", type=str, required=True, help='remote server')
parser.add_argument('-i', "--input", type=str, required=True, help='input dir')
parser.add_argument('-o', "--output", type=str, default="./run_scp.sh", help='output dir')
parser.add_argument('--log', action="store_true", help='True if sending log file')

if __name__ == '__main__':
    args = parser.parse_args()
    remote = args.remote
    input_dir = args.input
    if args.log:
        with open(args.output, "w") as f:
            f.write("#!/bin/bash\n")
            for exp_dir in os.listdir(input_dir):
                exp_dir = os.path.join(input_dir, exp_dir)
                if not os.path.isdir(exp_dir):
                    continue
                log_fn = "stdout_stderr.log"
                log_full_path = os.path.join(exp_dir, log_fn)
                if os.path.isfile(log_full_path):
                    cmd = 'scp {} {}:{}'.format(log_full_path, remote, log_full_path)
                    f.write(cmd + "\n")
    else:
        with open(args.output, "w") as f:
            f.write("#!/bin/bash\n")
            for exp_dir in os.listdir(input_dir):
                exp_dir = os.path.join(input_dir, exp_dir)
                if not os.path.isdir(exp_dir):
                    continue
                subdir = "plopt_ep_stats"
                seqlen_dir = "orig_seq_lens"
                seqlen_full_dir = os.path.join(exp_dir, subdir, seqlen_dir)
                if os.path.isdir(seqlen_full_dir):
                    cmd = 'scp {}/* {}:{}'.format(seqlen_full_dir, remote, seqlen_full_dir)
                    f.write(cmd + "\n")
                mbshapes_dir = "per_iter_mb_shapes"
                mbshapes_full_dir = os.path.join(exp_dir, subdir, mbshapes_dir)
                if os.path.isdir(mbshapes_full_dir):
                    cmd = 'scp {}/* {}:{}'.format(mbshapes_full_dir, remote, mbshapes_full_dir)
                    f.write(cmd + "\n")
