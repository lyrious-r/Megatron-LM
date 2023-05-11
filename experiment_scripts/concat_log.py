import os
import argparse

parser = argparse.ArgumentParser(description='Batch scp padding eff data to remote server')
parser.add_argument('-i', "--input", type=str, required=True, help='input dir')

if __name__ == '__main__':
    args = parser.parse_args()
    input_dir = args.input
    for exp_dir in os.listdir(input_dir):
        exp_dir = os.path.join(input_dir, exp_dir)
        if not os.path.isdir(exp_dir):
            continue
        log_fn = "stdout_stderr_1.log"
        out_log_fn = "stdout_stderr.log"
        log_full_path = os.path.join(exp_dir, log_fn)
        out_log_full_path = os.path.join(exp_dir, out_log_fn)
        if os.path.isfile(out_log_full_path):
            with open(out_log_full_path, "a") as f:
                with open(log_full_path, "r") as f2:
                    f.write(f2.read())