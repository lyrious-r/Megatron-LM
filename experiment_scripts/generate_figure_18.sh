# collect the iteration time statistics from logs. this command will generate ./experiments/best_throughput_iter_time_actual.csv and ./experiments/best_throughput_iter_time_estimated.csv
python3 ./experiment_utils/collect_iter_time_estimation_accuracy.py --exp_dir ./experiments/best_throughput
# collect the memory statistics from logs. this command will generate ./experiments/best_throughput_memory_actual.csv and ./experiments/best_throughput_memory_estimated.csv
python3 ./experiment_utils/collect_memory_estimation_accuracy.py --exp_dir ./experiments/best_throughput
# plot the figure
python3 ./experiment_utils/plot_fig18_a.py --data_prefix ./experiments/best_throughput --out_dir ./reproduced_figures
python3 ./experiment_utils/plot_fig18_b.py --data_prefix ./experiments/best_throughput --out_dir ./reproduced_figures