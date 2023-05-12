import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--exp_dir', type=str, default='./experiments')
parser.add_argument('-o', '--out', type=str, default='./experiment_results/preprocessing_time.csv')

args = parser.parse_args()

with open(args.out, 'w') as f:
    for exp_name in os.listdir(args.exp_dir):
        exp_full_path = os.path.join(args.exp_dir, exp_name)
        if "plopt" not in exp_name or not os.path.isdir(exp_full_path):
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
            spp = "spp" in spec_name
            # read log file to determine if it is a successful run
            stdout_stderr_fn = "stdout_stderr.log"
            if not os.path.isfile(os.path.join(exp_full_path, spec_name, stdout_stderr_fn)):
                continue
            with open(os.path.join(exp_full_path, spec_name, stdout_stderr_fn), 'r') as log_file:
                if not "Training finished successfully." in log_file.read():
                    continue
            preprocessing_dir = os.path.join(exp_full_path, spec_name, "plopt_logs", "preprocessing")
            if not os.path.isdir(preprocessing_dir):
                continue
            for fn in os.listdir(preprocessing_dir):
                if fn.endswith(".log"):
                    with open(os.path.join(preprocessing_dir, fn), 'r') as log_file:
                        for line in log_file:
                            if "EP generation for iteration" in line:
                                time = float(line.split()[-2])
                                iteration = int(line.split()[-4])
                                f.write(f"{exp_name},{esl},{dsl},{gbs},{spp},{iteration},{time}\n")

