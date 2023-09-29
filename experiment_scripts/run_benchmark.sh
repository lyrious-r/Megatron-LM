# baseline benchmarks
python3 run_experiment.py --experiment_name gpt_6.7b_16l_baseline_small_best
python3 run_experiment.py --experiment_name t5_11b_12l_baseline_small_best
python3 run_experiment.py --experiment_name gpt_6.7b_32l_baseline_mid_best
python3 run_experiment.py --experiment_name t5_11b_24l_baseline_mid_best
# dynapipe benchmarks
python3 run_experiment.py --experiment_name gpt_6.7b_16l_dynapipe_small_best
python3 run_experiment.py --experiment_name t5_11b_12l_dynapipe_small_best
python3 run_experiment.py --experiment_name gpt_6.7b_32l_dynapipe_mid_best
python3 run_experiment.py --experiment_name t5_11b_24l_dynapipe_mid_best
# baseline (c) benchmarks
python3 run_experiment.py --experiment_name gpt_6.7b_16l_baseline_small_control
python3 run_experiment.py --experiment_name t5_11b_12l_baseline_small_control
python3 run_experiment.py --experiment_name gpt_6.7b_32l_baseline_mid_control
python3 run_experiment.py --experiment_name t5_11b_24l_baseline_mid_control
# collect benchmark results
python3 ./experiment_utils/collect_throughput_stats.py --exp_dir ./experiments/best_throughput
python3 ./experiment_utils/collect_throughput_stats.py --exp_dir ./experiments/controlled_baseline