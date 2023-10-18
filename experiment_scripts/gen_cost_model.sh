echo "Creating cost models for GPT and T5 11B, using 4 different tensor parallel degrees (1, 2, 4, 8 GPUs)." 
python3 run_cost_model_benchmarks.py --out_dir ./cost_model_gpt_6.7b --model_config ./experiment_configs/model_configs/gpt_6.7b_16l.json
python3 run_cost_model_benchmarks.py --out_dir ./cost_model_t5_11b --model_config ./experiment_configs/model_configs/t5_11b_12l.json
mkdir -p ./cost_models
python3 gen_cost_model_from_profile.py --profile_dir ./cost_model_gpt_6.7b --out_path ./cost_models/gpt_6.7b_cm.pkl
python3 gen_cost_model_from_profile.py --profile_dir ./cost_model_t5_11b --out_path ./cost_models/t5_11b_cm.pkl
echo "Finished."
