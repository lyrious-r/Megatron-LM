from sys import platlibdir
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os
import json
import argparse

parser = argparse.ArgumentParser(description='Plot dataset seq len distribution')
parser.add_argument('--dataset-path', type=str, required=True, help='path to the dataset json.')
parser.add_argument('--output-path', type=str, help='output directory')
parser.add_argument('--bin-range', type=int, default=512, help='outliner threshold')

args = parser.parse_args()

if args.output_path is None:
    args.output_path = os.path.basename(args.dataset_path).rsplit(".", 1)[0] + ".pdf"

with open(args.dataset_path, "r") as f:
    dataset_json = json.load(f)

def is_en_fr(sample):
    if 'inputs' not in sample:
        return False
    if sample['inputs'].startswith("translate English to French"):
        return True
    else:
        return False

def is_en_de(sample):
    if 'inputs' not in sample:
        return False
    if sample['inputs'].startswith("translate English to German"):
        return True
    else:
        return False

def is_cnndm(sample):
    if 'inputs' not in sample:
        return False
    if sample['inputs'].startswith("summarize:"):
        return True
    else:
        return False

def is_unsupervised(sample):
    if 'text' not in sample:
        return False
    else:
        return True

dataframe_input = []
dataframe_target = []
for sample in dataset_json:
    if is_en_fr(sample):
        dataframe_input.append(("EnFr Input", int(sample['input_seq_len'])))
        dataframe_target.append(("EnFr Target", int(sample['target_seq_len'])))
    elif is_en_de(sample):
        dataframe_input.append(("EnDe Input", int(sample['input_seq_len'])))
        dataframe_target.append(("EnDe Target", int(sample['target_seq_len'])))
    elif is_cnndm(sample):
        dataframe_input.append(("CNNDM Input", int(sample['input_seq_len'])))
        dataframe_target.append(("CNNDM Target", int(sample['target_seq_len'])))
    else:
        raise ValueError("Unknown dataset")

df_input = pd.DataFrame(dataframe_input, columns=["Data", "Sequence Length"])
df_target = pd.DataFrame(dataframe_target, columns=["Data", "Sequence Length"])

df_input['Sequence Length'] = df_input['Sequence Length'].clip(upper=args.bin_range)
df_target['Sequence Length'] = df_target['Sequence Length'].clip(upper=args.bin_range)

bin_edges_input = np.histogram_bin_edges(df_input["Sequence Length"], bins=50, range=(0, args.bin_range))
bin_edges_target = np.histogram_bin_edges(df_target["Sequence Length"], bins=50, range=(0, args.bin_range))

print("Input: Max sequence length:", df_input["Sequence Length"].max(), "tokens, Min sequence length:", df_input["Sequence Length"].min(),
      "tokens, Mean sequence length:", df_input["Sequence Length"].mean(), "tokens.")
print("Target: Max sequence length:", df_target["Sequence Length"].max(), "tokens, Min sequence length:", df_target["Sequence Length"].min(),
        "tokens, Mean sequence length:", df_target["Sequence Length"].mean(), "tokens.")
print("Input CNNDM avg seq len:", df_input[df_input["Data"] == "CNNDM Input"]["Sequence Length"].mean())
print("Target CNNDM avg seq len:", df_target[df_target["Data"] == "CNNDM Target"]["Sequence Length"].mean())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(df_input, x="Sequence Length", hue="Data", kde=False, stat="count", bins=bin_edges_input, palette="tab10", multiple="stack", ax=ax1)
ax1.set(xlabel="Sequence Length", ylabel="Count", title="Input Sequence Length Distribution")
sns.histplot(df_target, x="Sequence Length", hue="Data", kde=False, stat="count", bins=bin_edges_input, palette="tab10", multiple="stack", ax=ax2)
ax2.set(xlabel="Sequence Length", ylabel="Count", title="Target Sequence Length Distribution")
fig.suptitle("Sequence Length Distribution")
# plt.show()
plt.tight_layout()
plt.savefig(args.output_path, dpi=300)
print("Saved to", args.output_path)
