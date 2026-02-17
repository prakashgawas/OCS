#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Bin Packing DAgger Training Script
Refactored for readability, maintainability, and performance.

Author: Prakash 
"""

import os
import time
import json
import logging
from datetime import datetime
from multiprocessing import get_context
from dataclasses import  asdict
from tqdm import tqdm
import pandas as pd

from OCS import AppointmentScheduler
from Learn_Module import IO_learning
from NN import SupervisedNN
from utility import parse_args, build_config, setup_logging, Config

import psutil, threading

def track_memory(interval=0.5):
    proc = psutil.Process(os.getpid())
    peak = 0
    while not stop_flag:
        mem = proc.memory_info().rss
        peak = max(peak, mem)
        time.sleep(interval)
    print(f"[MEMORY] Peak usage: {peak / (1024**3):.2f} GB")

# ============================================================
# 1. Utility Functions
# ============================================================

def get_last_model(folder: str) -> int:
    model = 0
    while os.path.isfile(os.path.join(folder, f'NN_{model}.pth')):
        model += 1
    return model - 1

# ============================================================
# 2. Simulation Runner
# ============================================================

def run_instance(n, k, sim, learner, folder, policy, config: Config):
    
    log = {}
    log['start'] = datetime.now().isoformat()
    
    print(f"Start run {n} {k} at {log['start']}.")
    name = f"{n}_{k}"
    if hasattr(sim, "patients"):
        sim.save_instance_csv(f"{n}_{k}", sim.patients, "Instances")
    elif hasattr(sim, "scenarios"):
        sim.save_instance_csv(f"{n}_{k}", sim.scenarios, "Instances")
    sim.build_model(sim.patients)
    status, termination, gap, sol_time = sim.solve(tee=True)
    print(f"Solver status = {status}, termination = {termination}, MIP gap = {gap}, Sol_time = {sol_time}")
    
    sol, obj_val = sim.extract_solution(sim.patients)
    sim.save_solution_files(f"Instance{n}_{k}", sol, obj_val, status, termination, gap, sol_time)
    learner.iteration = k
    state, done = sim.reset_simulator()
    print("SIMULATING....")
    states, actions, feasibles, rewards = [], [], [], []
    while not done:
        #print("State - ", state)
        expert_actions, gap, obj, sol_time = learner.get_expert_action(state, name)
        #print(sol_time)
        feature, action, decision_rule = learner.get_decision_to_take(state, k, policy, expert_actions)
        if config.num_class == 2 and not action:
            action = sim.select_physician(state)
        learner.add_new_data_avg(feature, gap, obj, sol_time, k)
        if config.evolve_expert:
            if sol_time <= 5:
                learner.update_scenarios(1)
        #print("Assign Patient - ", state["t"], " to physician ", action)
        states.append(state)
        next_state, reward, feasible, done = sim.step(action)
        actions.append(action)
        rewards.append(reward)
        feasibles.append(feasible)
        state = next_state
        #if state['time'] == 8:
        #     break
    states.append(state)
    stats = sim.collect_stats(states, actions, rewards, feasibles)
    print("Stats - ", stats)
    log['end'] = datetime.now().isoformat()
    print(f"End run {n} {k} at {log['end']}.")

    log = pd.DataFrame([log])
    log['run'] = k
    log['iteration'] = n
    log['step'] = 'collect'
    if config.store_sim_stats:
        sim.save_sim_trace_csv(tag=f"Train_stats_{n}_{k}" ,
        states=states,
        actions=actions,
        rewards=rewards,
        feasibles=feasibles,
        include_vectors = True,
        folder = "Train_stats/")
        #learner.pred_history.to_csv(os.path.join(sim_folder, f"Prediction_history_{n}_{k}.csv"), index=False)

    return learner.data, stats, log

# ============================================================
# 3. Main Training Logic
# ============================================================

def main():
    train_time = {}
    train_time["start"] =  datetime.now().isoformat()
    args = parse_args()
    config = build_config(args)

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
    instance_name = f"Instances_N{config.ocs_param['N']}_s{config.ocs_param['sigma']}_P{config.ocs_param['P']}_I{config.ocs_param['I']}_L{config.ocs_param['L']}_{config.ocs_param['k_min']}_{config.ocs_param['k_max']}_{config.ocs_param['physician_weights']}_{config.ocs_param['c_miss_1']}_{config.ocs_param['c_miss_2']}_{config.ocs_param['co']}_{config.ocs_param['cnp']}"
    train_folder = f"../data/Data_Dagger_{config.lambda_}_{config.adapter}_{config.evolve_expert}/{instance_name}"
    output_dir = os.path.join(train_folder, folder)
    os.makedirs(output_dir, exist_ok=True)

    # Logging & seed
    setup_logging(output_dir)

    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=4)

    # Initialise simulators & learners
    sim = [AppointmentScheduler(
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
        mipgap=config.mipgap,
        seed=config.seed + _,
        out_dir=output_dir,
        suffix=False
    ) for _ in range(config.runs)]
    
    for s in sim:
        s.setup_solver()

    learner = [IO_learning(sim[i], asdict(config), folder=output_dir) for i in range(config.runs)]
    if config.scenarios > 1 and config.stoch == 0:
        config.target_mode = "soft"
        config.target = "expert_distribution"
        print("Learning soft distribution")
    else:
        config.target_mode = "hard"
        config.target = "expert_action"
        
    if config.stoch == 0:
        config.evolve_expert = 0

        
    # Initialise policy
    policy = SupervisedNN( learner[0].feature_names, asdict(config), output_dir)

    # Resume from last model
    start = 0
    if config.last_model:
        model = get_last_model(output_dir)
        if model >= 0:
            config.model = model
            start = model + 1
            policy.load_model(output_dir, asdict(config))
            for i in range(config.runs):
                sim[i].set_seed(110 + i)
                learner[i].beta_update(start)
                learner[i].set_action_fn()

    # Training loop
    start_time = time.time()
    log_all = pd.DataFrame()
    sim_stats_all = []
    if config.static_data and config.scenarios == 1:
        fit_data = pd.read_csv(f"../data/Deterministic/{instance_name}/instances_solution/offline_features_data.csv")
        fit_data = fit_data[fit_data.instance_id < 200]
    else:
        fit_data = pd.DataFrame()
    
    for n in tqdm(range(start, config.learn_iter), desc="Learning Iterations", unit="iter"):
        logging.info(f"Iteration {n} | Beta = {learner[0].beta:.3f}")
    
        for i in range(config.runs):
            sim[i].generate_instance()
            learner[i].set_scenarios(1000 + n * 100 + i, asdict(config))
    
        with get_context("spawn").Pool(min(config.runs, 10)) as pool:
            # Add progress bar for pool mapping
            results = list(tqdm(
                pool.starmap(
                    run_instance,
                    [(n, i, sim[i], learner[i], output_dir, policy, config) for i in range(config.runs)]
                ),
                total=config.runs,
                desc=f"Runs (Iter {n})",
                leave=False
            ))
    
        all_data = pd.DataFrame()
        for k, (learner_data, stats, log) in enumerate(results):
            temp = pd.DataFrame.from_dict(learner_data, orient='index')
            temp.columns = learner[0].column_names
            temp['iteration'] = n
            all_data = pd.concat([all_data, temp])
            sim_stats_all.append(stats)
            log_all = pd.concat([log_all, log])
    
        all_data.to_csv(os.path.join(output_dir, f"Saved_data_{n}.csv"), index=False)
        folder = os.path.join(output_dir, "Train_stats")
        os.makedirs(folder, exist_ok=True)
        
        fit_data = pd.concat([fit_data, all_data])
        train_start = datetime.now().isoformat()
        if (n + 1) % config.update_model == 0:
            policy.update_model(fit_data, output_dir, n)
            for l in learner:
                l.beta_update()
                l.set_action_fn()
            fit_data = pd.DataFrame()
        logging.info(f"Iteration {n} completed in {time.time() - start_time:.2f} seconds")
        log = {'start':train_start, 'end': datetime.now().isoformat(), 'iteration': n, 'run':-1, 'step':'train'}
        log_all = pd.concat([log_all, pd.DataFrame([log])])
        filepath = os.path.join(folder, f"Sim_stats_{n}.json")
        with open(filepath, 'w') as f:
            json.dump(sim_stats_all, f, indent=4)
            
    train_time["end"] =  datetime.now().isoformat()
    filepath = os.path.join(folder, "Train_time.json")
    with open(filepath, 'w') as f:
         json.dump(train_time, f, indent=4)
    log_all.to_csv(os.path.join(folder, "Train_time_all.csv"))

if __name__ == '__main__':
    stop_flag = False
    thread = threading.Thread(target=track_memory)
    thread.start()
    main()
    stop_flag = True
    thread.join()
