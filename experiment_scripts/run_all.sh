# This script will run all the experiments for artifact evaluation.
# It uses the preprocessed data and pre-generated grid search results
# by default. Please refer to the main README for instructions on how 
# to generate them from scratch.

cd /root/Megatron-LM

# Skip cost model generation since we use a pre-generated one
# Skip grid search and use pre-generated results

### Fig.12 and Fig.13
# run benchmark and collect data.
./experiment_scripts/run_benchmark.sh
# plot figure
./experiment_scripts/generate_figure_12_13.sh
### Fig.14
./experiment_scripts/generate_figure_14.sh
### Fig.16
./experiment_scripts/generate_figure_16.sh
### Fig.17
./experiment_scripts/generate_figure_17.sh

### Fig.15
# run ablation experiments.
./experiment_scripts/run_ablation.sh
# plot the figure
./experiment_scripts/generate_figure_15.sh