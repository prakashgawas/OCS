#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:39:26 2025

@author: prakashgawas
"""

import argparse
from dataclasses import dataclass
import sys
import logging
import os

@dataclass
class Config:
    policy: str = "NN"
    train: bool = True
    store_sim_stats: bool = True
    lambda_: float = 0.7
    scenarios: int = 15
    runs: int = 1
    vanilla: int = 1
    gated: int = 0
    update_model: int = 1
    time_limit: int = 10
    mipgap:float = 0
    learn_iter: int = 2
    stoch: int = 0
    use_weight: int = 0
    task: str = "full"  # "full" or "bin"
    action_type: str = "max"  # "max" or "random"
    last_model: int = 0
    ocs_param: dict = None
    num_class: int = 0
    new: int = 1
    seed: int = 1
    train:bool = True
    model:int = 0
    epochs:int = 100
    adapter:str = "per_phy"
    sims:int = 10
    target:str = "expert_distribution"
    target_mode:str = "soft"
    static_data:int = 0
    evolve_expert:int = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Bin Packing DAgger Trainer")

    # Learning settings
    parser.add_argument('--learn_iter', type=int, default=2, help="Number of learning iterations")
    parser.add_argument('--runs', type=int, default=2, help="Parallel simulation runs")
    parser.add_argument('--lm', type=float, default=0.8, help="Lambda value for learning")
    parser.add_argument('--time_limit', type=int, default=30, help="Solver time limit")
    parser.add_argument('--mipgap', type=float, default=0.001, help="Solver allowed mipgap")
    parser.add_argument('--scenarios', type=int, default=30, help="Number of stochastic scenarios")
    parser.add_argument('--stoch', type=int, default=1, help="Enable stochastic runs")
    parser.add_argument('--resume', type=int, default=0, help="Resume from last model")
    parser.add_argument('--vanilla', type=int, default=1, help="Vanilla DAgger mode")
    parser.add_argument('--gated', type=int, default=0, help="Enable gated model")
    parser.add_argument('--um', type=int, default=1, help="Update model frequency")
    parser.add_argument('--new', type=int, default=1, help="Use new scenario")
    parser.add_argument('--use_weight', type=int, default=0, help="Use weighted loss")
    parser.add_argument('--train', type = int,  default=1, help="Number of simulations")
    parser.add_argument('--model', type = int,  default=-1, help="Which model to run")
    parser.add_argument('--sims', type = int,  default=1, help="Number of simulations")
    parser.add_argument('--seed', type = int,  default=0, help="seed")
    parser.add_argument('--store_sim', type = int,  default=0, help="store sime trace")
    parser.add_argument('--static_data', type = int,  default=0, help="use static data")

    # Problem settings
    parser.add_argument('--N', type=int, default=100, help='number of patients, mean')
    parser.add_argument('--s', type=int, default=8, help='std deviation')
    parser.add_argument('--P', type=int, default=4, help='number of physicians')
    parser.add_argument('--L', type=int, default=20, help='patient cap of physicians')
    parser.add_argument('--I', type=int, default=8, help='number of slots (used only for session_time = I*Delta)')
    # --- Costs for the no-slot model ---
    parser.add_argument('--c_miss_1', type=int, default=200, help='rejection cost for priority-1')
    parser.add_argument('--c_miss_2', type=int, default=50, help='rejection cost for priority-2')
    parser.add_argument('--co', type=int, default=0, help='overtime cost per minute')
    parser.add_argument('--cnp', type=int, default= 10, help='non pref percent')
    parser.add_argument('--Rmax', type=int, default=None, help='optional total acceptance cap')
    parser.add_argument('--kmin', type=int, default= 1, help='non pref percent')
    parser.add_argument('--kmax', type=int, default= 4, help='non pref percent')
    parser.add_argument('--pw', type=int, default= 1, help='physician weights given')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--task', type=str, default="full", choices=['full', 'bin'], help="Task type")
    parser.add_argument('--adapter', type=str, default="flat", help="NN type")
    parser.add_argument('--evolve_expert', type=int, default=0, help="whether to evolve expert from in terms of scenario")
    parser.add_argument('--action_type', type=str, default="random", choices=['max', 'random'], help="Action selection type")

    return parser.parse_args()

def build_config(args) -> Config:
    ocs_param = dict(N=args.N, sigma= args.s, P=args.P, I=args.I, L=args.L, k_min = args.kmin, k_max = args.kmax, physician_weights = args.pw, c_miss_1=args.c_miss_1, c_miss_2=args.c_miss_2, co=args.co, cnp= args.cnp, Rmax=args.Rmax)
    num_class = 1 if args.task == "bin" else args.P + 1
    new = args.new if args.stoch == 0 else 1
    scenarios = 1 if new == 0 else args.scenarios

    return Config(
        lambda_=args.lm,
        scenarios=scenarios,
        runs=args.runs,
        vanilla=args.vanilla,
        store_sim_stats=args.store_sim>0,
        gated=args.gated,
        update_model=args.um,
        time_limit=args.time_limit,
        mipgap=args.mipgap,
        learn_iter=args.learn_iter,
        stoch=args.stoch,
        use_weight=args.use_weight,
        task=args.task,
        action_type=args.action_type,
        last_model=args.resume,
        ocs_param=ocs_param,
        num_class=num_class,
        new=new,
        seed=args.seed,
        train=args.train,
        model=args.model,
        epochs=args.epochs,
        adapter=args.adapter,
        sims=args.sims,
        static_data=args.static_data,
        evolve_expert=args.evolve_expert
    )

def setup_logging(output_dir: str):
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"Logging initialized. Logs will be saved to {log_file}")
