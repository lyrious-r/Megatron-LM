import sys
import os
import argparse
import subprocess
from pathlib import Path

MASTER_PORT = 8000
MLM_DIR = Path(__file__).parent
VOCAB_FILE = MLM_DIR / "vocabs" / "t5-base-vocab.txt"
DISTRIBUTED_ARGS = "--nproc_per_node {} --nnodes 1 --node_rank 0 --master_addr localhost --master_port {} --use-env"

CMD_TEMPLATE = """
CUDA_VISIBLE_DEVICES={} python3 -m torch.distributed.launch {} \
       microbenchmark_t5.py \
       --tensor-model-parallel-size {} \
       --pipeline-model-parallel-size 1 \
       --encoder-num-layers {} \
       --decoder-num-layers {} \
       --hidden-size {} \
       --num-attention-heads {} \
       --kv-channels {} \
       --ffn-hidden-size {} \
       --encoder-seq-length {} \
       --decoder-seq-length {} \
       --micro-batch-size {} \
       --global-batch-size 4096 \
       --max-position-embeddings {} \
       --no-async-tensor-model-parallel-allreduce \
       --train-iters {} \
       --train-epochs 1 \
       --lr-decay-iters 100 \
""" \
+ """  --vocab-file {} \
""".format(VOCAB_FILE) \
+ """  --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 50 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 5 \
       --fp16 \
        --vocab-extra-ids 100 \
       --num-workers 2 \
       --dataloader-type ordered \
       --microbenchmark-save-dir {} \
       --tokens-per-global-batch 16384"""


def parse_args():
    parser = argparse.ArgumentParser("Run single GPU benchmark for T5.")
    parser.add_argument(
        "-tp",
        "--tensor-model-parallel-size",
        type=int,
        required=True,
        help="Tensor model parallel size",
    )
    parser.add_argument(
        "-e",
        "--encoder-seq-length",
        type=int,
        required=True,
        help="Encoder sequence length",
    )
    parser.add_argument(
        "-d",
        "--decoder-seq-length",
        type=int,
        required=True,
        help="Decoder sequence length",
    )
    parser.add_argument(
        "-b",
        "--micro-batch-size",
        type=int,
        required=True,
        help="Micro batch size",
    )
    parser.add_argument(
        "-el",
        "--encoder-num-layers",
        type=int,
        required=True,
        help="Number of encoder layers",
    )
    parser.add_argument(
        "-dl",
        "--decoder-num-layers",
        type=int,
        required=True,
        help="Number of decoder layers",
    )
    parser.add_argument(
        "-rc",
        "--recompute-type",
        choices=["None", "Selective", "Full"],
        default="None",
        help="Enable recomputation",
    )
    parser.add_argument(
        "-i",
        "--benchmark-iters",
        type=int,
        default=50,
        help="Number of iterations to benchmark",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="GPU to run benchmark on.",
    )
    # default model configuration corresponds to T5-11B
    parser.add_argument(
        "--hidden-size", type=int, default=1024, help="Model hidden size"
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=128,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--kv-channels", type=int, default=128, help="Number of KV Channels"
    )
    parser.add_argument(
        "--ffn-hidden-size", type=int, default=65536, help="FFN hidden size"
    )
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Use flash attention.",
    )
    args = parser.parse_args()
    args.devices = [int(d) for d in args.devices.split(",")]
    return args

def get_microbenchmark_name(tp_size, hidden_size, num_attention_heads,
                            kv_channels, ffn_hidden_size, encoder_seq_length,
                            decoder_seq_length, micro_batch_size, recompute_type):
    name = "tp{}_hs{}_ah{}_kv{}_ffhs{}_encsl{}_decsl{}_mbs{}".format(
        tp_size,
        hidden_size,
        num_attention_heads,
        kv_channels,
        ffn_hidden_size,
        encoder_seq_length,
        decoder_seq_length,
        micro_batch_size,
    )
    # add recomputation settings if exist
    if recompute_type != "None":
        name += "_rc_{}".format(recompute_type.lower())
        if recompute_type == "Full":
            name += "_{}".format("uniform")
    return name

def run_benchmark(
    tp_size,
    enc_seqlen,
    dec_seqlen,
    microbatch_size,
    encoder_num_layers,
    decoder_num_layers,
    output_dir,
    devices,
    benchmark_iters=50,
    hidden_size=1024,
    n_attn_heads=128,
    kv_channels=128,
    ffn_hidden_size=65536,
    recompute_type="None",
    use_flash_attn=False,
    log_file=None,
):
    assert len(devices) >= 1, "Must have at least one device"
    output_fn = "microbench_" + get_microbenchmark_name(
        tp_size,
        hidden_size,
        n_attn_heads,
        kv_channels,
        ffn_hidden_size,
        enc_seqlen,
        dec_seqlen,
        microbatch_size,
        recompute_type,
    ) + ".txt"
    output_path = os.path.join(output_dir, output_fn)
    if os.path.exists(output_path):
        # skip if already exists
        return 0

    distributed_args = DISTRIBUTED_ARGS.format(tp_size, MASTER_PORT + devices[0])
    cmd = CMD_TEMPLATE.format(
        ",".join([str(d) for d in devices]),
        distributed_args,
        tp_size,
        encoder_num_layers,
        decoder_num_layers,
        hidden_size,
        n_attn_heads,
        kv_channels,
        ffn_hidden_size,
        enc_seqlen,
        dec_seqlen,
        microbatch_size,
        max(enc_seqlen, dec_seqlen),
        benchmark_iters,
        output_dir,
    )
    if recompute_type != "None":
        if recompute_type == "Selective":
            cmd += " --recompute-activations"
        elif recompute_type == "Full":
            cmd += " --recompute-granularity full --recompute-method uniform"
        else:
            raise ValueError(f"Unknown recompute type {recompute_type}")
    if use_flash_attn:
        cmd += " --use-flash-attn"

    if log_file:
        with open(log_file, "a") as f:
            p = subprocess.run(cmd, shell=True, stderr=f, stdout=f)
    else:
        p = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return p.returncode


if __name__ == "__main__":
    args = parse_args()
    retval = run_benchmark(
        args.tensor_model_parallel_size,
        args.encoder_seq_length,
        args.decoder_seq_length,
        args.micro_batch_size,
        args.encoder_num_layers,
        args.decoder_num_layers,
        args.output_dir,
        args.devices,
        args.benchmark_iters,
        args.hidden_size,
        args.num_attention_heads,
        args.kv_channels,
        args.ffn_hidden_size,
        args.recompute_type,
        args.use_flash_attn,
    )
    sys.exit(retval)
