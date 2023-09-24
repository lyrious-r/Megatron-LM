import os
import jsonlines
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", type=str, help="input config file")
parser.add_argument("-d", "--best_config_dir", type=str, help="best config dir")
parser.add_argument('-o', "--output", type=str, help="output config file")

args = parser.parse_args()

if args.input is None:
    assert args.best_config_dir is not None
    assert args.output is None
    inputs = []
    for fn in os.listdir(args.best_config_dir):
        if fn.endswith(".jsonl"):
            inputs.append(os.path.join(args.best_config_dir, fn))
    args.input = inputs
    
else:
    args.input = [args.input]

output_json_template = {
    "encoder_seq_length": None,
    "decoder_seq_length": None,
    "seq_length": None,
    "tokens_per_global_batch": None,
    "tensor_parallel_size": None,
    "pipeline_parallel_size": None,
    "micro_batch_size": None,
    "recompute_level": None,
    "enable_deepspeed": None,
    "deepspeed_zero_stage": None,
    "dynapipe_enable_packing": False,
}

for input_fn in args.input:
    assert input_fn.endswith(".jsonl")
    with jsonlines.open(input_fn) as reader:
        if args.output is None:
            output_fn = os.path.join("./experiment_configs/best_configs", os.path.basename(input_fn).rsplit('.', 1)[0] + '.jsonl')
        else:
            output_fn = args.output
        with jsonlines.open(output_fn, mode='w') as writer:
            for obj in reader:
                if obj["Global Batch Size"] == 65536 or obj["Encoder SeqLen"] == 2048:
                    output_json = output_json_template.copy()
                    output_json["encoder_seq_length"] = obj["Encoder SeqLen"]
                    output_json["decoder_seq_length"] = obj["Decoder SeqLen"]
                    output_json["seq_length"] = obj["Encoder SeqLen"]
                    output_json["tokens_per_global_batch"] = obj["Global Batch Size"]
                    best_config = obj["Best Config"]
                    output_json["tensor_parallel_size"] = best_config["tp_size"]
                    output_json["pipeline_parallel_size"] = best_config["pp_size"]
                    output_json["micro_batch_size"] = best_config["mbs"]
                    output_json["recompute_level"] = best_config["rc"]
                    output_json["enable_deepspeed"] = (best_config["ds_level"] >= 1)
                    output_json["deepspeed_zero_stage"] = best_config["ds_level"]
                    writer.write(output_json)
