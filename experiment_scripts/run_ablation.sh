# grid search for mbs for token-based partition
python3 run_experiment.py --experiment_name t5_11b_24l_dynapipe_mid_ablgrid
# run other ablations
python3 run_experiment.py --experiment_name gpt_6.7b_32l_dynapipe_mid_abl
python3 run_experiment.py --experiment_name t5_11b_24l_baseline_mid_abl
python3 run_experiment.py --experiment_name t5_11b_24l_dynapipe_mid_abl

# collect results
python3 ./experiment_utils/collect_throughput_stats.py --exp_dir ./experiments/ablation
python3 ./experiment_utils/collect_throughput_stats.py --exp_dir ./experiments/ablation_grid