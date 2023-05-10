import os
import argparse

parser = argparse.ArgumentParser(description='Batch scp padding eff data to remote server')
parser.add_argument('-r', "--remote", type=str, required=True, help='remote server')
parser.add_argument('-i', "--input", type=str, required=True, help='input dir')
parser.add_argument('-o', "--output", type=str, default="./run_scp.sh", help='output dir')

if __name__ == '__main__':
    args = parser.parse_args()
    remote = args.remote
    input_dir = args.input

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
