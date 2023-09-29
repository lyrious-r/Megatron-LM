# This script will run all the experiments for artifact evaluation.
# It uses the preprocessed data and pre-generated grid search results
# by default. Please refer to the main README for instructions on how 
# to generate them from scratch.

cd /root/Megatron-LM

# Download the preprocessed datasets
./experiment_scripts/download_preprocessed_dataset.sh

# Skip cost model generation since we use a pre-generated one
# Skip grid search and use pre-generated results

### Fig.12 and Fig.13
# run benchmark and collect data. Generates ./experiments/best_throughput.jsonl
./experiment_scripts/run_benchmark.sh
# plot figure
python3 ./experiment_utils/plot_fig12_fig13.py --best_throughput_data ./experiments/best_throughput.jsonl --controlled_throughput_data ./experiments/controlled_baseline.jsonl --out_dir ./reproduced_figures
### Fig.14
# collect the statistics from logs. Generates ./experiments/best_throughput_batch_eff.jsonl
python3 ./experiment_utils/collect_batching_efficiency_stats.py --exp_dir ./experiments/best_throughput
# plot the figure
python3 ./experiment_utils/plot_fig14.py --batch_eff_data ./experiments/best_throughput_batch_eff.jsonl --out_dir ./reproduced_figures

### Fig.15
# run ablation experiments. Generates ./experiments/ablation.jsonl and ./experiments/ablation_grid.jsonl
./experiment_scripts/run_ablation.sh
# plot the figure
python3 ./experiment_utils/plot_fig15.py --ablation_data ./experiments/ablation.jsonl ./experiments/ablation_grid.jsonl --out_dir ./reproduced_figures

### Fig.16
# collect stats from logs. Generates ./experiments/best_throughput_planning_time.csv
python3 ./experiment_utils/collect_planning_time.py --exp_dir ./experiments/best_throughput
# plot the figure
python3 ./experiment_utils/plot_fig16.py --planning_time_data ./experiments/best_throughput_planning_time.csv --out_dir ./reproduced_figures

### Fig.17
# collect iteration time stats from logs. 
# Generates ./experiments/best_throughput_iter_time_actual.csv
# and ./experiments/best_throughput_iter_time_estimated.csv
python3 ./experiment_utils/collect_iter_time_estimation_accuracy.py --exp_dir ./experiments/best_throughput
# collect memory stats from logs.
# Generates ./experiments/best_throughput_memory_actual.csv
# and ./experiments/best_throughput_memory_estimated.csv
python3 ./experiment_utils/collect_memory_estimation_accuracy.py --exp_dir ./experiments/best_throughput
# plot the figure
python3 ./experiment_utils/plot_fig17_a.py --data_prefix ./experiments/best_throughput --out_dir ./reproduced_figures
python3 ./experiment_utils/plot_fig17_b.py --data_prefix ./experiments/best_throughput --out_dir ./reproduced_figures