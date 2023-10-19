## Artifact for DynaPipe: Optimizing Multi-task Training through Dynamic Pipelines

This repository contains the artifact for reproducing the experiments in the paper `DynaPipe: Optimizing Multi-task Training through Dynamic Pipelines`. The main implementation of DynaPipe can be found at [DynaPipe](https://github.com/chenyu-jiang/EuroSys24-AE-Spring-92).

This repository is based on a fork of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). Main modifications include adding support for packing in the dataloader, implementing the pipeline instructions for DynaPipe, and adding the scripts for running the experiments.

## Directory Hierachy
The hierarchy follows that of the original Megatron-LM repository. We highlight the modifications below

```
|--cost_models
|  :contains pre-generated cost models for gpt_7.6b and t5_11b
|
|--datasets
|  : (to be generated) contains pre-processed datasets needed 
|    for the experiments
|
|--docker
|  :contains the Dockerfile and supporting scripts for setting
|   up the container for evaluation
...
|
|--experiment_configs
|  :contains config files which specify various parameters read
|   by experiment scripts
|
|--experiment_scripts
|  :contains scripts for running the experiments
|
|--experiment_utils
|  :utility scripts for collecting experiment logs and generating
|   figures in the paper
|
|--experiments
|  : (to be generated) contains the logs and statistics generated
|    by the experiments
|
...
|
|--megatron
|  |--data
|  |   |--data_samplers.py
|  |   |--t5_dataset.py
|  |   :these files are modified to support packing.
|  |    We also replace the default dataloader with 
|  |    the DynaPipe dataloader (for non-baseline
|  |    experiments).
|  |
|  |--pipeline_executor.py
|     :implements DynaPipe pipeline instructions
|
|--reproduced_figures
|  : (to be generated) contains the figures reproduced by the
|    experiments
|
...
|
|--microbenchmark_gpt.py
|--microbenchmark_t5.py
|--gpt_microbenchmark_wrapper.py
|--t5_microbenchmark_wrapper.py
|--run_cost_model_benchmarks.py
|  :these files are used for generating the cost models
|
|--run_experiment.py
|  :entry script for automating most experiments
```

## Setup
### Hardware Requirement
The original experiments are performed using up to 4 AWS EC2 p4d instances (each with 8 NVIDIA A100 40GB GPUs). For artifact evaluation, we provide temporary access to a single p4d instance. 

In general, the experiments can be run on machines with multiple GPUs, provided that PyTorch and Megatron-LM supports them.

### Software Requirement
Please use the Dockerfile to setup the environment.

Main software dependencies include:
* PyTorch (>= 2.1.0)
* [DynaPipe](https://github.com/chenyu-jiang/EuroSys24-AE-Spring-92)
* Megatron-LM (this repo)
* A slightly modified version of DeepSpeed: [https://github.com/chenyu-jiang/DeepSpeed](https://github.com/chenyu-jiang/DeepSpeed). We removed a timer that introduce unnecessary synchronization which disrupts our schedule and disabled overflow checking for more consistent throughput measurement.

Please check the Dockerfile for how to install these packages.

### Container Setup

Run the following commands to build the container image:
```bash
git clone https://github.com/chenyu-jiang/Megatron-LM.git
cd Megatron-LM/docker
./build_image.sh
```
(Note: for artifact evaluation, the provided machine already contain a pre-built image.)

To create a container, run (inside the docker directory):
```bash
./run.sh
```
You will find this repository at `/root/Megatron-LM` inside the container.

## Dataset Preparation
Our experiments used the FLAN dataset. Due to its size, the download process can take 12+ hours (depending on network speed). The download process is also prone to errors and availability issues caused by version mismatch between `tf-dataset` and the downloading code. To reduce time for artifact evaluation, we also include a pre-processed version of the dataset in the provided machine.

(For artifact evaluation) To copy the preprocessed datasets into the `datasets` directory, run outside the container:
```bash
cd ~/preprocessed_datasets
docker cp datasets dynapipe:/root/Megatron-LM
```

To generate the dataset from scratch, follow the following steps:


1. Clone the repository for the dataset (a fork of the original repository with some version mismatch fixed. Also added a downloading script) and install dependencies:
```bash
git clone https://github.com/chenyu-jiang/text-to-text-transfer-transformer.git
cd text-to-text-transfer-transformer
pip3 install -r requirements.txt
```
2. Download the raw dataset (generates `supervised_proportional.jsonl`):
```bash
python3 prepare_dataset.py
```

3. Perform some initial cleaning (generates `cleaned_supervised_proportional.jsonl`):
```bash
python3 clean_dataset.py
```

4. Preprocess the dataset with Megatron-LM's data loader script (generates `.bin` and `.idx` files)
```bash
cd /root/Megatron-LM
./experiment_scripts/run_preprocess_flan.sh /root/text-to-text-transfer-transformer/cleaned_supervised_proportional.jsonl
```

5. Copy the generated files to the `datasets` directory:
```bash
cp /root/text-to-text-transfer-transformer/*.bin /root/text-to-text-transfer-transformer/*.idx /root/Megatron-LM/datasets
```

## One-line command for all experiments
Note: all the following commands (and commands in later sections) should be executed inside the container under the `/root/Megatron-LM` directory.

We provide a script for running all experiments at once. To do this, run the following commands in the container:
```bash
./experiment_scripts/run_all.sh
```
The reproduced figures will be in the `/root/Megatron-LM/reproduced_figures` directory.
The following sections describe the steps to manually run each experiment in detail.

## Cost Model Generation
DynaPipe needs cost models to optimally generate micro-batches and compute pipeline schedule. These cost models are generated through profiling the model under different micro-batch sizes and sequence lengths. Since we benchmark multiple models and grid search for the optimal parallelism in the experiments, we needs to generate multiple cost models and such process can be slow on a single p4d node (12+ hours). We provide pre-generated cost models for the GPT-7.6B and T5-11B models in the `cost_models` directory.

(Skippable for artifact evaluation) To generate the cost models from scratch, run:
```bash
./experiment_scripts/gen_cost_model.sh
```

## Experiment Steps
### Grid search
In this step, we grid search for the best parallelism for both baseline (Megatron-LM) and DynaPipe. For baselines, we also grid search for the optimal micro-batch size and recomputation (gradient checkpointing) strategy. Due to the large number of experiments, this step can take ~50 hours to complete on a single p4d node. Therefore we also provide the results of the grid search (best configurations) in the `experiment_configs/best_configs` directory and the controlled config (i.e., where we use the same parallelism of DynaPipe to run the baseline) in the `experiment_configs/control_configs` directory.

(Skippable for artifact evaluation) To run the grid search from scratch, run:
```bash
# make a copy of the best_configs directory
mv ./experiment_configs/best_configs ./experiment_configs/best_configs_backup
# make a copy of the control_configs directory
mv ./experiment_configs/control_configs ./experiment_configs/control_configs_backup
./experiment_scripts/run_grid_search.sh
```

The script will perform all needed profiling, generate the best and controlled config for each setting and dump results in the corresponding directories.

### Fig.12 and Fig.13
Fig.12 and Fig.13 uses best/control configs (obtained by the grid search) to run full benchmarks for throughput comparison. Note for artifact evaluation, only Fig.12 (a)(b)(e)(f) and Fig.13 (a)(b)(e)(f) can be generated on a single p4d node. The other figures require multiple p4d nodes.

To run the experiments, run:
```bash
./experiment_scripts/run_benchmark.sh
```
The benchmarking takes about 18 hours to complete. This will generate `best_throughput.jsonl` and `controlled_baseline.jsonl` containing the throughput results in `experiments` directory. To regenerate figure 12 and figure 13, run:
```bash
./experiment_scripts/generate_figure_12_13.sh
```

The generated figures will be in the `reproduced_figures` directory.

### Fig.14
Fig.14 compares the batching efficiency of DynaPipe and baseline. Such statistics are collected during the benchmarking process. To generate the figure, run:
```bash
./experiment_scripts/generate_figure_14.sh
```

### Fig.15
Fig.15 performs ablation study on our micro-batch partitioning and scheduling algorithms. The config files for ablation experiments are located in `experiment_configs/ablation_configs`. To run the experiments, run:
```bash
./experiment_scripts/run_ablation.sh
```
The ablation experiments takes about 8 hours to complete. Results will be saved in `experiments/ablation.jsonl` and `experiments/ablation_grid.jsonl`. 

To generate the figure, run:
```bash
./experiment_scripts/generate_figure_15.sh
```

### Fig.16
Fig.16 shows the execution time for DynaPipe. The statistics are also collected during the benchmarking process when reproducing Fig.12 and 13. To generate the figure, run:
```bash
./experiment_scripts/generate_figure_16.sh
```
(Note: since we did not run benchmark on more than 1 p4d nodes, the planning time distribution in the generated figure is expected to be slightly different from the original paper.)

### Fig.17
Fig.17 shows the prediction accuracy of DynaPipe's cost models. The memory and iteration time data is collected through the benchmark, which are compared against the predictions. To generate the figure, run:
```bash
./experiment_scripts/generate_figure_17.sh
```

