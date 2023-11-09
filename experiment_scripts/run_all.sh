# This script will run all the experiments for artifact evaluation.
# It uses the preprocessed data and pre-generated grid search results
# by default. Please refer to the main README for instructions on how 
# to generate them from scratch.

cd /root/Megatron-LM

# Skip cost model generation since we use a pre-generated one
# Skip grid search and use pre-generated results

### Fig.13 and Fig.14
# run benchmark and collect data.
echo "Running benchmark for Fig.13 and Fig.14."
./experiment_scripts/run_benchmark.sh
echo "Benchmark for Fig.13 and Fig.14 completed."
# plot figure
./experiment_scripts/generate_figure_13_14.sh
echo "Finished plotting Fig.13 and Fig.14."
### Fig.15
./experiment_scripts/generate_figure_15.sh
echo "Finished plotting Fig.15."
### Fig.17
./experiment_scripts/generate_figure_17.sh
echo "Finished plotting Fig.17."
### Fig.18
./experiment_scripts/generate_figure_18.sh
echo "Finished plotting Fig.18."

### Fig.16
# run ablation experiments.
echo "Running ablation experiments."
./experiment_scripts/run_ablation.sh
echo "Ablation experiments completed."
# plot the figure
./experiment_scripts/generate_figure_16.sh
echo "Finished plotting Fig.16."

echo "All experiments completed. Please check the generated figures in ./experiments/reproduced_figures"