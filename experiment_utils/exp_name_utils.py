# Description: This file contains functions to extract information 
# from experiment and spec names.

import pandas as pd

def get_model(row):
    if "gpt" in row["exp_name"]:
        return "GPT"
    else:
        return "T5"

def get_dp_degree(row):
    return int(row["spec_name"].split("_")[0][2:])

def get_tp_degree(row):
    return int(row["spec_name"].split("_")[1][2:])

def get_pp_degree(row):
    return int(row["spec_name"].split("_")[2][2:])

def get_model_size(row):
    if "gpt" in row["exp_name"]:
        model_type = "gpt"
    else:
        model_type = "t5"
    if "small" in row["exp_name"]:
        model_size = "small"
    elif "mid" in row["exp_name"]:
        model_size = "mid"
    elif "large" in row["exp_name"] and "xlarge" not in row["exp_name"]:
        model_size = "large"
    else:
        model_size = "xlarge"
    return model_type + "_" + model_size

def get_zero_degree(row):
    if "zero1" in row["spec_name"]:
        return 1
    elif "zero2" in row["spec_name"]:
        return 2
    else:
        return 0

def get_seqlen(row):
    if "t5" in row["exp_name"]:
        return int(row["spec_name"].split("_")[3][5:])
    else:
        return int(row["spec_name"].split("_")[3][2:])

def get_global_batch_size(row):
    if "gpt" in row["exp_name"]:
        return int(row["spec_name"].split("_")[4][3:])
    else:
        return int(row["spec_name"].split("_")[5][3:])

def get_framework(row):
    if "control" in row["exp_name"]:
        return "baseline (c)"
    if "dynapipe" in row["exp_name"]:
        return "dynapipe"
    return "baseline"

def get_cluster(row):
    if "small" in row["exp_name"]:
        return "small"
    elif "mid" in row["exp_name"]:
        return "mid"
    elif "large" in row["exp_name"] and "xlarge" not in row["exp_name"]:
        return "large"
    else:
        return "xlarge"

def get_rc(row):
    if "rc" in row["spec_name"]:
        rc_type = row["spec_name"].split("rc")[1].split("_")[0]
        return rc_type
    else:
        return "dynamic"

def get_zero_degree(row):
    if "zero1" in row["spec_name"]:
        return 1
    elif "zero2" in row["spec_name"]:
        return 2
    else:
        return 0

def augment_df(df: pd.DataFrame):
    df["model"] = df.apply(get_model, axis=1)
    df["dp_degree"] = df.apply(get_dp_degree, axis=1)
    df["tp_degree"] = df.apply(get_tp_degree, axis=1)
    df["pp_degree"] = df.apply(get_pp_degree, axis=1)
    df["model_size"] = df.apply(get_model_size, axis=1)
    df["zero_degree"] = df.apply(get_zero_degree, axis=1)
    df["seqlen"] = df.apply(get_seqlen, axis=1)
    df["global_batch_size"] = df.apply(get_global_batch_size, axis=1)
    df["framework"] = df.apply(get_framework, axis=1)
    df["cluster"] = df.apply(get_cluster, axis=1)
    df["rc"] = df.apply(get_rc, axis=1)
    df["zero_degree"] = df.apply(get_zero_degree, axis=1)
    return df