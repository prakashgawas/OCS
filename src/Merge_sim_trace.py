#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:26:10 2025

@author: prakashgawas
"""

import os
import pandas as pd
import argparse
import numpy as np
from datetime import datetime
import json
parser = argparse.ArgumentParser(description="OCS")
parser.add_argument('--s', type=int, default=8, help='std deviation')
parser.add_argument('--kmin', type=int, default=1, help='non pref percent')
parser.add_argument('--pw', type=int, default= 1, help='physician weights given')
parser.add_argument('--learn_iter', type=int, default=200, help="Number of learning iterations")
parser.add_argument('--runs', type=int, default=10, help="Parallel simulation runs")
parser.add_argument('--time_limit', type=int, default=30, help="Solver time limit")
parser.add_argument('--mipgap', type=float, default=0.02, help="Solver allowed mipgap")
parser.add_argument('--scenarios', type=int, default=5, help="Number of stochastic scenarios")
parser.add_argument('--stoch', type=int, default=0, help="Enable stochastic runs")
parser.add_argument('--resume', type=int, default=0, help="Resume from last model")
parser.add_argument('--vanilla', type=int, default=1, help="Vanilla DAgger mode")
parser.add_argument('--gated', type=int, default=0, help="Enable gated model")
parser.add_argument('--um', type=int, default=1, help="Update model frequency")
parser.add_argument('--new', type=int, default=1, help="Use new scenario")
parser.add_argument('--lm', type=float, default=0.8, help="Lambda value for learning")
parser.add_argument('--use_weight', type=int, default=0, help="Use weighted loss")
parser.add_argument('--train', type = int,  default=1, help="Number of simulations")
parser.add_argument('--model', type = int,  default=-1, help="Which model to run")
parser.add_argument('--sims', type = int,  default=1000, help="Number of simulations")
parser.add_argument('--seed', type = int,  default=0, help="seed")
parser.add_argument('--store_sim', type = int,  default=0, help="store sime trace")
parser.add_argument('--adapter', type=str, default="flat", help="NN type")
parser.add_argument('--evolve_expert', type=int, default=0, help="whether to evolve expert from in terms of scenario")
args = parser.parse_args()

def merge_files(folder: str, output_file: str, files:str = "OCS", remain:str='schedule',  start_id=0, end_id=1000, scen = None):
    dfs = []
    
    jsfilename = os.path.join(folder, "Train_stats/Train_time.json")
    hours = 0
    if os.path.isfile(jsfilename):
        with open(jsfilename, "r") as f:
            data = json.load(f)
            diff =  datetime.strptime(data['end'], "%Y-%m-%dT%H:%M:%S.%f") - datetime.strptime(data['start'], "%Y-%m-%dT%H:%M:%S.%f")
            hours = np.round(diff.total_seconds() / 3600, 3)

    filename = os.path.join(folder, "All_data.csv")
    df = pd.read_csv(filename)
    gap = [df['gap'].mean(), df['gap'].max(), df['gap'].quantile(0.99)]
    sol_time = [df['sol_time'].mean(), df['sol_time'].max(), df['sol_time'].quantile(0.99)]

    for i in range(start_id, end_id + 1):
        if remain =="":
            name = f"{files}_{i}.csv"
        else:
            name = f"{files}_{i}_{remain}.csv"
        filename = os.path.join(folder + "Evaluation/", name)
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            
            df['instance_id'] = i  # optional: keep track of which file it came from
            if scen is not None:
                df['scenarios'] = scen
                df['stoch'] = args.stoch
                df['runs'] = args.runs
                df['learn_iter'] = args.learn_iter
                df['new'] = args.new
                df['tl'] = args.time_limit
                df['mg'] = args.mipgap
                df['train_time'] = hours
                df['gap_mean'] = np.round(gap[0], 3)
                df['gap_max'] = np.round(gap[1], 3)
                df['gap_99p'] = np.round(gap[2], 3)
                df['sol_time_mean'] = np.round(sol_time[0], 3)
                df['sol_time_max'] = np.round(sol_time[1], 3)
                df['sol_time_99p'] = np.round(sol_time[2], 3)
                
            dfs.append(df)
        else:
            print(f"File not found: {filename}")

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Merged {len(dfs)} files into {output_file}")
    else:
        print("No files found to merge.")


if __name__ == "__main__":
    
    base_path = f"../data/Data_Dagger_{args.lm}_{args.adapter}_{args.evolve_expert}/Instances_N100_s{args.s}_P4_I8_L20_{args.kmin}_4_{args.pw}_200_50_0_10/NN_iter{args.learn_iter}_runs{args.runs}_um1_new{args.new}_stoch{args.stoch}_sc{args.scenarios}_van1_gat0_tl{args.time_limit}_mg{args.mipgap}_nc5/"
    # folder_path = base_path + "Sim_stats/"  # or the folder where the files are located
    # output_path = folder_path + "merged_sim_trace.csv"
    # merge_files(folder_path, output_path, files="Sim_stats_0", remain="sim_trace", end_id  = 200 )

    folder_path = base_path   # or the folder where the files are located
    output_path = f"../data/Data_Dagger_{args.lm}_{args.adapter}_{args.evolve_expert}/Evaluation_N100_s{args.s}_P4_I8_L20_{args.kmin}_4_{args.pw}_200_50_0_10/"  # or the folder where the files are located
    os.makedirs(output_path, exist_ok=True)
    output_path = output_path + f"merged_sim_stats_NN_iter{args.learn_iter}_runs{args.runs}_um1_new{args.new}_stoch{args.stoch}_sc{args.scenarios}_van1_gat0_tl{args.time_limit}_mg{args.mipgap}_nc5.csv"
    merge_files(folder_path, output_path, files="Sim_stats", remain="", end_id  = args.learn_iter, scen = args.scenarios )



