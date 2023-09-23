import argparse

from dynapipe.data_opt.cost_models import ProfileBasedCostModelWithRC

parser = argparse.ArgumentParser("Generate cost model from profile")
parser.add_argument(
    "--profile_dir",
    type=str,
    required=True,
    help="Path to profile file",
)
parser.add_argument(
    "--out_path",
    type=str,
    required=True,
    help="Output path for cost model",
)

args = parser.parse_args()

cm = ProfileBasedCostModelWithRC(args.profile_dir)
cm.save(args.out_path)
