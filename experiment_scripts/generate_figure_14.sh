# collect the statistics from logs. this command will generate ./experiments/best_throughput_batch_eff.jsonl
python3 ./experiment_utils/collect_batching_efficiency_stats.py --exp_dir ./experiments/best_throughput
# plot the figure
python3 ./experiment_utils/plot_fig14.py --batch_eff_data ./experiments/best_throughput_batch_eff.jsonl --out_dir ./reproduced_figures