#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 15:56:32 2025

@author: prakashgawas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Bin Packing DAgger Training Script
Refactored for readability, maintainability, and performance.

Author: Prakash 
"""

import os
import json
import logging
from datetime import datetime
from dataclasses import  asdict
from tqdm import tqdm
import pandas as pd

from OCS import AppointmentScheduler
from Learn_Module import IO_learning
from NN import SupervisedNN
from utility import parse_args, build_config, setup_logging, Config


# ============================================================
# 1. Simulation Runner
# ============================================================

def run_instance(n, k, sim, learner, folder, policy, config: Config):
    logging.info(f"Start run {n} {k} at  {datetime.now()}")

    #sim.save_instance_csv(f"{n}_{k}", sim.patients, "SimInstances")
    learner.iteration = k
    states, actions,feasibles, rewards = [], [], [], []
    state, done = sim.reset_simulator()

    while not done:
        x = learner.get_feature_vector(state)
        action, pred = learner.get_action(state, k, policy, x)
        #if state['time'] == 47:
        #     pass
        if config.num_class == 2 and action:
            action = sim.select_physician(state)
        learner.add_new_data_avg(x + [action], k = k)
        next_state, reward, feasible, done = sim.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        feasibles.append(feasible)
        state = next_state
        #logging.info("Assign - ", state["current_item"], "to -", action)
        #if state['time'] == 1:
        #     break
    states.append(state)
    stats = sim.collect_stats(states, actions, rewards, feasibles)
    stats['k'] = k

    if config.store_sim_stats:
       sim.save_sim_trace_csv(tag=f"Sim_stats_{n}_{k}" ,
       states=states,
       actions=actions,
       rewards=rewards,
       feasibles=feasibles,
       include_vectors = True,
       folder = f"Sim_stats_{config.model}/")
    return stats

# ============================================================
# 4. Main Training Logic
# ============================================================

def main():
    args = parse_args()
    config = build_config(args)
    config.train = 0

    # Folder structure
    components = [
        f"iter{config.learn_iter}",
        f"runs{config.runs}",
        f"um{config.update_model}",
        f"new{config.new}",
        f"stoch{config.stoch}",
        f"sc{config.scenarios}",
        f"van{config.vanilla}",
        f"gat{config.gated}",
        f"tl{config.time_limit}",
        f"mg{config.mipgap}",
        f"nc{config.num_class}"
    ]
    name = "_".join(components)
    folder = f"{config.policy}_{name}/"
    train_folder = f"../data/Data_Dagger_{config.lambda_}_{config.adapter}_{config.evolve_expert}/Instances_N{config.ocs_param['N']}_s{config.ocs_param['sigma']}_P{config.ocs_param['P']}_I{config.ocs_param['I']}_L{config.ocs_param['L']}_{config.ocs_param['k_min']}_{config.ocs_param['k_max']}_{config.ocs_param['physician_weights']}_{config.ocs_param['c_miss_1']}_{config.ocs_param['c_miss_2']}_{config.ocs_param['co']}_{config.ocs_param['cnp']}"
    output_dir = os.path.join(train_folder, folder)
    os.makedirs(output_dir, exist_ok=True)

    # Logging & seed
    setup_logging(output_dir)

    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=4)

    # Initialise simulators & learners
    sim = AppointmentScheduler(
        n_patients=config.ocs_param['N'],
        sigma = config.ocs_param['sigma'],
        n_phys=config.ocs_param['P'],
        n_slots=config.ocs_param['I'],
        k_max=config.ocs_param['k_max'],
        k_min=config.ocs_param['k_min'],
        # --- costs wired to your class fields used in build_model ---
        c_miss_1=config.ocs_param['c_miss_1'],
        c_miss_2=config.ocs_param['c_miss_2'],
        c_overtime=config.ocs_param['co'],
        c_np_per=config.ocs_param['cnp'],
        physician_weights=config.ocs_param['physician_weights'],
        release_cap=config.ocs_param['Rmax'],
        cap_per_phys=config.ocs_param['L'],
        time_limit=config.time_limit,
        seed=config.seed,
        out_dir=output_dir,
        suffix=False
    )

    learner = IO_learning(sim, asdict(config), folder=output_dir)

    # Initialise policy
    policy = SupervisedNN( learner.feature_names, asdict(config), output_dir)


    # Training loop
    sim_stats_all = []
    all_data = pd.DataFrame()
    
    for n in tqdm(range(config.sims), desc="Simulating Iterations", unit="iter"):
        #if n ==25:
        #    print("here")
        sim.set_seed(n+1)
        sim.generate_instance()
        stats = run_instance(0, n, sim, learner, output_dir, policy, config)        
        sim_stats_all.append(stats)
        
    all_data = pd.DataFrame.from_dict(learner.data, orient='index')
    all_data.columns = learner.feature_names + ["action", "gap", "obj" , "sol_time", "sim"]

    output_dir = os.path.join(output_dir, "Evaluation")
    os.makedirs(output_dir, exist_ok=True)
    #all_data.to_csv(os.path.join(output_dir, f"Feature_sim_data_{config.model}.csv"), index=False)
    #learner.pred_history.to_csv(os.path.join(output_dir, f"Predict_data_{config.model}.csv"), index=False)
    df = pd.DataFrame(sim_stats_all)
    df.to_csv(os.path.join(output_dir, f"Sim_stats_{config.model}.csv"), index=False)
    
    print("Average Cost = ", df.total_cost.mean())

if __name__ == '__main__':
    main()
