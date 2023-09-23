echo "Creating cost models for GPT and T5 11B, using 4 different tensor parallel degrees (1, 2, 4, 8 GPUs)." 
echo "Run profile from scratch or use the provided profile results?"
echo "1) Run profile from scratch (est. 24+ hrs)  2) Use provided results"
read -p "Your choice (please enter number): " choice
case $choice in
    1) echo "Generating from scratch."
    python3 run_cost_model_benchmarks.py --out_dir ./cost_model_gpt_6.7b --model_config ./experiment_configs/model_configs/gpt_6.7b_16l.json
    python3 run_cost_model_benchmarks.py --out_dir ./cost_model_t5_11b --model_config ./experiment_configs/model_configs/gpt_6.7b_16l.json
    mkdir -p ./cost_models
    python3 gen_cost_model_from_profile.py --profile_dir ./cost_model_gpt_6.7b --out_path ./cost_models/gpt_6.7b_cm.pkl
    python3 gen_cost_model_from_profile.py --profile_dir ./cost_model_t5_11b --out_path ./cost_models/t5_11b_cm.pkl
    ;;
    2) echo "Using provided one."
    mkdir -p ./cost_models
    cp -r /root/intermediate_results/cost_models/* ./cost_models
    ;;
    *) echo "Invalid choice. Exiting."
    exit 1
    ;;
esac
echo "Finished."
