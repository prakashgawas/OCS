#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:57:02 2025

@author: prakashgawas
"""

import os
import glob
import pandas as pd

def concat_all_csvs(folder: str, out_csv: str, pattern: str = "*.csv", identifier: bool = False):
    """
    Concatenate all CSV files in `folder` (same columns) into `out_csv`.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {folder} (pattern={pattern})")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    dfs = []
    for p in paths:
        df = pd.read_csv(p)

        if identifier:
            df["i"] = p.split("/")[-1].split("_")[1]
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"Merged {len(paths)} files -> {out_csv}")

name = "Evaluation_N100_s8_P4_I8_L20_1_4_1_200_50_0_10"
concat_all_csvs(f"../data/Data_Dagger_0.8_flat_0/{name}", f"../data/Data_Dagger_0.8_flat_0/{name}/merged_{name}.csv")

#name = "Instances_N100_s8_P4_I8_L20_1_4_1_200_50_0_10/SimInstances"
#concat_all_csvs(f"../data/Data_Dagger_0.8_flat/{name}", f"../data/Data_Dagger_0.8_flat/{name}/merged_instances.csv", "*patients.csv", True)

def parse_number(s):
    m = re.search(r'\d+(?:\.\d+)?', s)
    if not m:
        return None
    val = m.group()
    return int(val) if val.isdigit() else float(val)


from pathlib import Path
import re

# Root directory containing the folders
root_dir = Path("../data/Data_Dagger_0.8_flat_0/Instances_N100_s8_P4_I8_L20_1_4_1_200_50_0_10")

# Name of the CSV file inside each folder
csv_name = "Train_stats/Train_time_all.csv"

# Read and collect all CSVs
dfs = []
for csv_path in root_dir.glob(f"*/{csv_name}"):
    temp = pd.read_csv(csv_path)
    parts = csv_path.parts[4]
    parts = parts.split("_")
    numbers = [parse_number(s) for s in parts if parse_number(s) is not None]
    temp["Total_Iteration"] = numbers[0]
    temp["runs"] = numbers[1]
    temp["new"] = numbers[3]
    temp["stoch"] = numbers[4]
    temp["scenarios"] = numbers[5]
    temp['tl'] =  numbers[8]
    temp['mg'] =  numbers[9]
    dfs.append(temp)

# Concatenate
combined_df = pd.concat(dfs, ignore_index=True)

# Save combined CSV
combined_df.to_csv("../data/Data_Dagger_0.8_flat_0/" + name + "/All_train_times.csv", index=False)
