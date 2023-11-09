import argparse

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from exp_name_utils import augment_df

parser = argparse.ArgumentParser()
parser.add_argument('--data_prefix', type=str, required=True,
                    help="Path prefix to the files containing the estimated/actual iteration time "
                    "data, generated by collect_iter_time_estimation_accuracy.py")
parser.add_argument('--out_dir', type=str, required=True,
                    help="Path to the directory where the output plots "
                    "will be saved")

args = parser.parse_args()

estimated_iter_time = pd.read_csv(args.data_prefix + "_iter_time_estimated.csv", low_memory=False)
measured_iter_time = pd.read_csv(args.data_prefix + "_iter_time_actual.csv", low_memory=False)

# for each iteration, get max time across dpg
estimated_iter_time_maxed = estimated_iter_time.groupby(["exp_name", "spec_name", "iteration"]).agg({'estimated_time':'max'}).reset_index()
# average across iterations
estimated_iter_time_avg = estimated_iter_time_maxed.groupby(["exp_name", "spec_name"]).agg({'estimated_time':'mean'}).reset_index()

measured_iter_time_avg = measured_iter_time.groupby(["exp_name", "spec_name"]).agg({'measured_time':'mean'}).reset_index()
joined = pd.merge(estimated_iter_time_avg, measured_iter_time_avg, on=["exp_name", "spec_name"], how="inner", validate="one_to_one")

joined = augment_df(joined)
joined["Model"] = joined["model"]

t5_joined = joined[joined["Model"] == "T5"]
gpt_joined = joined[joined["Model"] == "GPT"]

t5_measured_time = t5_joined["measured_time"].to_numpy()
t5_estimated_time = t5_joined["estimated_time"].to_numpy()
gpt_measured_time = gpt_joined["measured_time"].to_numpy()
gpt_estimated_time = gpt_joined["estimated_time"].to_numpy()

t5_avg_error = (abs(t5_measured_time - t5_estimated_time) / t5_measured_time)
gpt_avg_error = (abs(gpt_measured_time - gpt_estimated_time) / gpt_measured_time)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

ax = sns.lineplot(x=range(0, 17000), y=range(0, 17000), color="red", ax=ax, alpha=0.2)
ax = sns.scatterplot(data=joined, x="measured_time", y="estimated_time", hue="Model", alpha=0.9, s=8, linewidth=0.1)
ax.set_xlim(0, 16000)
ax.set_ylim(0, 16000)

for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(11)
for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
    item.set_fontsize(12)
ax.set_xlabel("Measured Iteration Time (ms)")
ax.set_ylabel("Estimated Iteration Time (ms)")
ax.text(500, 15000, f"Mean Percentage Error:", fontsize=12)
ax.text(500, 14000, f"T5: {t5_avg_error.mean() * 100:.2f}%", fontsize=12)
ax.text(500, 13000, f"GPT: {gpt_avg_error.mean() * 100:.2f}%", fontsize=12)
ax.legend(loc="lower right", fontsize=12, title="Model")

plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='12') # for legend title

fig.savefig(os.path.join(args.out_dir, "fig18_a.pdf"), bbox_inches="tight")