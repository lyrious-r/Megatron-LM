import argparse
import json
from plopt.data_opt.optimizer import DataAssignmentOptimizer

parser = argparse.ArgumentParser("Generate microbatches worker process.")
parser.add_argument("profile_path", type=str)
parser.add_argument("input_seqlens", type=str)
parser.add_argument("target_seqlens", type=str)
parser.add_argument("output_path", type=str)
parser.add_argument("--uniform-partition", action="store_true")
parser.add_argument("--uniform-partition-batch-size", type=int, default=1)
parser.add_argument("--min-compute-eff", type=float, default=0.9)
parser.add_argument("--satisfy-seqlen-percentile", type=float, default=0.75)
parser.add_argument("--enable-packing", type=bool, default=True)
parser.add_argument("--uniform-cost", type=bool, default=False)
parser.add_argument("--interleaved", action="store_true")

args = parser.parse_args()

# hardcode model spec for now
dataopt = DataAssignmentOptimizer(args.profile_path, 4, 3, 36000, 1024, 32, 16384, 128)

input_seqlens = [int(x) for x in args.input_seqlens.split(",")]
target_seqlens = [int(x) for x in args.target_seqlens.split(",")]

if args.uniform_partition:
    partition_method = "uniform"
else:
    partition_method = "dp"

_, indices = dataopt.generate_microbatches(
    input_seqlens,
    decoder_sample_sequence_lengths=target_seqlens,
    partition_method=partition_method,
    bottleneck_tsp=False,
    min_compute_efficiency=args.min_compute_eff,
    uniform_partition_batch_size=args.uniform_partition_batch_size,
    satisfy_seqlen_percentile=args.satisfy_seqlen_percentile,
    enable_packing=args.enable_packing,
    interleaved=args.interleaved,
    uniform_cost=args.uniform_cost,
)

with open(args.output_path, "w") as f:
    json.dump(indices, f)
