import sys
import argparse
import subprocess
from pathlib import Path

MASTER_PORT = 8000
MLM_DIR = Path(__file__).parent
VOCAB_FILE = MLM_DIR / "vocabs" / "gpt2-vocab.json"
MERGE_FILE = MLM_DIR / "vocabs" / "gpt2-merges.txt"
DISTRIBUTED_ARGS = "--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port {} --use-env"

CMD_TEMPLATE = """
CUDA_VISIBLE_DEVICES={} python3 -m torch.distributed.launch {} \
       microbenchmark_gpt.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers {} \
       --hidden-size {} \
       --num-attention-heads {} \
       --kv-channels {} \
       --ffn-hidden-size {} \
       --seq-length {} \
       --micro-batch-size {} \
       --global-batch-size 4096 \
       --max-position-embeddings 65536 \
       --no-async-tensor-model-parallel-allreduce \
       --train-iters {} \
       --train-epochs 1 \
       --lr-decay-iters 100 \
""" \
+ """  --vocab-file {} \
       --merge-file {} \
""".format(VOCAB_FILE, MERGE_FILE) \
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
    parser = argparse.ArgumentParser("Run single GPU benchmark")
    parser.add_argument(
        "-s",
        "--seq-length",
        type=int,
        required=True,
        help="Sequence length",
    )
    parser.add_argument(
        "-b",
        "--micro-batch-size",
        type=int,
        required=True,
        help="Micro batch size",
    )
    parser.add_argument(
        "-n",
        "--num-layers",
        type=int,
        required=True,
        help="Number of layers",
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
        "--device",
        type=int,
        default=0,
        help="GPU to run benchmark on.",
    )
    # default model configuration corresponds to GPT-3 6.7B
    parser.add_argument(
        "--hidden-size", type=int, default=4096, help="Model hidden size"
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=32,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--kv-channels", type=int, default=128, help="Number of KV Channels"
    )
    parser.add_argument(
        "--ffn-hidden-size", type=int, default=16384, help="FFN hidden size"
    )
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Use flash attention.",
    )

    args = parser.parse_args()
    return args


def run_benchmark(
    seqlen,
    microbatch_size,
    n_layers,
    output_dir,
    device=0,
    benchmark_iters=50,
    hidden_size=4096,
    n_attn_heads=32,
    kv_channels=128,
    ffn_hidden_size=16384,
    recompute_type="None",
    use_flash_attn=False,
    log_file=None,
):
    distributed_args = DISTRIBUTED_ARGS.format(MASTER_PORT + device)
    cmd = CMD_TEMPLATE.format(
        device,
        distributed_args,
        n_layers,
        hidden_size,
        n_attn_heads,
        kv_channels,
        ffn_hidden_size,
        seqlen,
        microbatch_size,
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
        p = subprocess.run(cmd, shell=True) #, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return p.returncode


if __name__ == "__main__":
    args = parse_args()
    retval = run_benchmark(
        args.seq_length,
        args.micro_batch_size,
        args.num_layers,
        args.output_dir,
        args.device,
        args.benchmark_iters,
        args.hidden_size,
        args.num_attention_heads,
        args.kv_channels,
        args.ffn_hidden_size,
        args.recompute_type,
        args.use_flash_attn,
    )
    sys.exit(retval)
