cd /root/Megatron-LM
# baselines
# gpt (4GPUs)
python3 run_experiment.py --experiment_name gpt_6.7b_16l_baseline_small_grid --sequence_length_range 512,1024,2048,4096,8192 --global_batch_size_range 65536
python3 run_experiment.py --experiment_name gpt_6.7b_16l_baseline_small_grid --sequence_length_range 2048 --global_batch_size_range 16384,32768,65536,131072
# gpt (8GPUs)
python3 run_experiment.py --experiment_name gpt_6.7b_32l_baseline_mid_grid --sequence_length_range 512,1024,2048,4096,8192 --global_batch_size_range 65536
python3 run_experiment.py --experiment_name gpt_6.7b_32l_baseline_mid_grid --sequence_length_range 2048 --global_batch_size_range 16384,32768,65536,131072
# t5 (4GPUs)
python3 run_experiment.py --experiment_name t5_11b_12l_baseline_small_grid --sequence_length_range 512,1024,2048,4096,8192 --global_batch_size_range 65536
python3 run_experiment.py --experiment_name t5_11b_12l_baseline_small_grid --sequence_length_range 2048 --global_batch_size_range 16384,32768,65536,131072
# t5 (8GPUs)
python3 run_experiment.py --experiment_name t5_11b_24l_baseline_mid_grid --sequence_length_range 512,1024,2048,4096,8192 --global_batch_size_range 65536
python3 run_experiment.py --experiment_name t5_11b_24l_baseline_mid_grid --sequence_length_range 2048 --global_batch_size_range 16384,32768,65536,131072

# dynapipe
# gpt (4GPUs)
python3 run_experiment.py --experiment_name gpt_6.7b_16l_dynapipe_small_grid --sequence_length_range 512,1024,2048,4096,8192 --global_batch_size_range 65536
python3 run_experiment.py --experiment_name gpt_6.7b_16l_dynapipe_small_grid --sequence_length_range 2048 --global_batch_size_range 16384,32768,65536,131072
# gpt (8GPUs)
python3 run_experiment.py --experiment_name gpt_6.7b_32l_dynapipe_mid_grid --sequence_length_range 512,1024,2048,4096,8192 --global_batch_size_range 65536
python3 run_experiment.py --experiment_name gpt_6.7b_32l_dynapipe_mid_grid --sequence_length_range 2048 --global_batch_size_range 16384,32768,65536,131072
# t5 (4GPUs)
python3 run_experiment.py --experiment_name t5_11b_12l_dynapipe_small_grid --sequence_length_range 512,1024,2048,4096,8192 --global_batch_size_range 65536
python3 run_experiment.py --experiment_name t5_11b_12l_dynapipe_small_grid --sequence_length_range 2048 --global_batch_size_range 16384,32768,65536,131072
# t5 (8GPUs)
python3 run_experiment.py --experiment_name t5_11b_24l_dynapipe_mid_grid --sequence_length_range 512,1024,2048,4096,8192 --global_batch_size_range 65536
python3 run_experiment.py --experiment_name t5_11b_24l_dynapipe_mid_grid --sequence_length_range 2048 --global_batch_size_range 16384,32768,65536,131072

# collect grid search results and generate best_configs and control_configs jsonls
python3 ./experiment_utils/collect_throughput_stats.py --exp_dir ./experiments/grid_search
python3 ./experiment_utils/generate_best_config.py --input_data ./experiments/grid_search.jsonl --config_dir ./experiment_configs/best_configs
python3 ./experiment_utils/generate_control_config.py --input_data ./experiments/grid_search.jsonl --config_dir ./experiment_configs/control_configs
