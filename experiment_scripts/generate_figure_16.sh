# collect the statistics from logs. this command will generate ./experiments/best_throughput_planning_time.csv
python3 ./experiment_utils/collect_planning_time.py --exp_dir ./experiments/best_throughput
# plot the figure
python3 ./experiment_utils/plot_fig16.py --planning_time_data ./experiments/best_throughput_planning_time.csv --out_dir ./reproduced_figures