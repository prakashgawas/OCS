#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:49:29 2023

@author: Prakash
"""

import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import os
from collections import Counter
import json
from datetime import datetime

from OCS import AppointmentScheduler
from SOCS import SAppointmentScheduler


    
class IO_learning():
    def __init__(self, sim, config, folder = None):
        self.sim = sim
        self.policy_type = config['policy']
        self.feature_names = self.get_feature_names(sim)
        self.pred_history = pd.DataFrame(columns=[i for i in range(config["num_class"])] + ['t', 'action', 'run'])
        self.column_names = self.feature_names + ['pred', 'action', 'label', 'expert_action', 'decision_rule', 'time', 'expert_actions', 'expert_distribution', 'num_scenarios', 'gap', 'obj', 'sol_time', 'run']
        self.lambda_ = config['lambda_']
        self.vanilla = config["vanilla"]
        self.gated = config["gated"]
        self.beta = 1 
        self.task = config["task"]
        self.set_action_fn()
        self.iteration = 0
        self.stoch = config['stoch']
        self.all_data = pd.DataFrame()

        self.num_class = config["num_class"]
        self.new = config["new"] 
        self.num_scenarios = config["scenarios"]
        self.data = {}
        self.folder = folder
        self.sim_stats = pd.DataFrame()
        self.ber_randomGen = bernoulli
        self.ber_randomGen.random_state = np.random.Generator(np.random.PCG64(10))
        
       
    def get_feature_names(self, sim) -> list[str]:
        """
        Build flat CSV column names for one decision-step example.
    
        Returns:
          List of column names in the order your row builder should emit values.
        """
        feature_names: list[str] = []
    
        # ---- Context features ----
        feature_names += [
            "t_current",            # <-- added missing comma here
            "remaining_release",
            "priority_is_p1",
            #"arrival_score"
            "duration"
        ]
    
        # ---- Per-physician features ----
        for p in range(sim.P):
            feature_names.append(f"phys{p}_is_preferred")
            feature_names.append(f"phys{p}_cap_remaining_norm")
            feature_names.append(f"phys{p}_acc_p1_norm")
            #feature_names.append(f"phys{p}_acc_p2_norm")
            feature_names.append(f"phys{p}_load_pressure")
    
        # ---- Action mask (as in your original: mask_reject repeated per physician) ----
        for p in range(sim.P):
            feature_names.append(f"mask_phys{p}")
        feature_names.append("mask_reject")
        return feature_names

    def set_action_fn(self):
        if self.policy_type == 'NN':    
            self.get_action = self.get_action_NN
            print("Action Maker - NN")
    
    def set_scenarios(self, seed, config):
        if config['stoch'] == 0:
            self.scenario = []
            for i in range(config['scenarios']):
                self.scenario.append( AppointmentScheduler(
                                                n_patients=config["ocs_param"]['N'],
                                                sigma =config["ocs_param"]['sigma'],
                                                n_phys=config["ocs_param"]['P'],
                                                n_slots=config["ocs_param"]['I'],
                                                k_max=config["ocs_param"]['k_max'],
                                                k_min=config["ocs_param"]['k_min'],
                                                # --- costs wired to your class fields used in build_model ---
                                                c_miss_1=config["ocs_param"]['c_miss_1'],
                                                c_miss_2=config["ocs_param"]['c_miss_2'],
                                                c_overtime=config["ocs_param"]['co'],
                                                c_np_per=config["ocs_param"]['cnp'],
                                                physician_weights=config["ocs_param"]['physician_weights'],
                                                release_cap=config["ocs_param"]['Rmax'],
                                                cap_per_phys=config["ocs_param"]['L'],
                                                time_limit=config["time_limit"],
                                                mipgap=config["mipgap"],
                                                seed=seed + i,
                                                out_dir=self.folder,
                                                suffix=False))
                self.scenario[i].setup_solver()
        else:
            self.scenario = []
            self.scenario.append(SAppointmentScheduler(
                                            n_patients=config["ocs_param"]['N'],
                                            sigma =config["ocs_param"]['sigma'],
                                            n_phys=config["ocs_param"]['P'],
                                            n_slots=config["ocs_param"]['I'],
                                            k_max=config["ocs_param"]['k_max'],
                                            k_min=config["ocs_param"]['k_min'],
                                            # --- costs wired to your class fields used in build_model ---
                                            c_miss_1=config["ocs_param"]['c_miss_1'],
                                            c_miss_2=config["ocs_param"]['c_miss_2'],
                                            c_overtime=config["ocs_param"]['co'],
                                            c_np_per=config["ocs_param"]['cnp'],
                                            physician_weights=config["ocs_param"]['physician_weights'],
                                            release_cap=config["ocs_param"]['Rmax'],
                                            cap_per_phys=config["ocs_param"]['L'],
                                            time_limit=config["time_limit"],
                                            mipgap=config["mipgap"],
                                            seed=seed,
                                            out_dir=self.folder,
                                            suffix=False,
                                            num_scenarios=config['scenarios']))
        
        for scenario in self.scenario:
            scenario.setup_solver()
        expert_action_map = {    0: self.get_avg_expert_action_wt,
                                 1: self.get_avg_stoch_expert_action_wt
                            }
        self.get_expert_action = expert_action_map.get(config['stoch'])
        
    def update_scenarios(self, num):
        for scenario in self.scenario:
            self.num_scenarios = scenario.update_scenarios(num)  
        
            
    def get_avg_expert_action_wt(self, state, name):
        actions = []
        total_gap = 0
        total_time = 0
        total_obj = 0
        for i in range(len(self.scenario)):
            self._prepare_scenario(self.scenario[i], state, f"{name}_{i}", use_scenarios=(self.stoch != 0))
            status, termination, gap, sol_time = self.scenario[i].solve(tee=False)
            #self._log_solver_result(state, status, termination, gap, sol_time, f"{name}_{i}")
            sol, obj_val = self.scenario[i].extract_solution(self.scenario[i].regenerated_patients)
            total_gap += gap
            total_obj += obj_val
            total_time += sol_time
            #self.scenario[i].save_solution_files(f"regen_{state['t']}_{name}_{i}", sol, obj_val, status, termination, gap, sol_time, "Regen")
            actions.append(self.scenario[i].get_current_action(0))
        return actions, total_gap/len(self.scenario), total_obj/len(self.scenario), total_time
    
    def get_avg_stoch_expert_action_wt(self, state, name):
        actions = []
        gap = None
        for i in range(len(self.scenario)):
            self._prepare_scenario(self.scenario[i], state, f"{name}_{i}", use_scenarios=(self.stoch != 0))
            status, termination, gap, sol_time = self.scenario[i].solve(tee=False)
            #self._log_solver_result(state, status, termination, gap, sol_time, f"{name}_{i}")
            sol, obj_val = self.scenario[i].extract_solution()
            #self.scenario[i].save_solution_files(f"regen_{state['t']}_{name}_{i}", sol["scenarios"], status, termination, gap, sol_time, "Regen")
            actions.append(self.scenario[i].get_current_action(0))
        return actions, gap, obj_val, sol_time
    
    def _prepare_scenario(self, scenario, state, name, use_scenarios=False):
        if self.new:
            scenario.regenerate_scenario(state)
        else:
            scenario.regenerate_scenario(state, self.sim.patients, self.new)
        patients = scenario.scenarios if use_scenarios else scenario.regenerated_patients
        #scenario.save_instance_csv(f"regen_{state['t']}_{name}", patients, "Regen")
        if use_scenarios:
            info = scenario.build_model(state = state, new=self.new)
        else:
            info = scenario.build_model(patients, state, new=self.new)
            
        #out_dir = os.path.join(scenario.out_dir, "Regen")
        #os.makedirs(out_dir, exist_ok=True)
        #info_json_path = os.path.join(out_dir, f"{name}_{state['t']}_info.json")
    
        #with open(info_json_path, "w") as f:
        #    json.dump(info, f, indent=2)

    def _log_solver_result(self, state, status, termination, gap, sol_time, name):
        print(f"Time = {state['t']}, Instance = {name}| Solver status = {status}, "
              f"termination = {termination}, MIP gap = {gap}, Sol_time = {sol_time}") #Iteration = {n} Run = {k} 
    
    def get_decision_to_take(self, state, k, policy, expert_actions):
        decision_rule = self.get_decision_rule()
        x = self.get_feature_vector(state)
        label, dist = self.sampled_vote(expert_actions, policy)
        label = int(np.round(label))
        if not decision_rule:
            action, pred = self.get_action(state, k, policy, x)
        #    self.x["gate"] = 0
        else:
        #    self.x["gate"] = 0  
            if self.vanilla:
                action = label
                pred = -1
            # elif self.gated > 0:
            #     action =  self.aggregator(wait_set)
            #     action, self.x["gate"]  = self.second_expert_check(action, state)
            
        if self.task == "bin":
            expert_action = label > 0
        else:
            expert_action = label

        x = x + [pred, action, label, expert_action, decision_rule, state["t"], expert_actions, list(dist.values()), self.num_scenarios]
        return x, action, decision_rule
    
    def get_decision_rule(self):
        return self.ber_randomGen.rvs(self.beta)
    
    def beta_update(self, initial = 1):
        self.beta = np.power(self.lambda_, initial) * self.beta 
    
    def get_feature_vector(self, state: dict) :
        """
        Convert a simulator `state` into a flat vector aligned with get_feature_names(sim).
    
        Order (must match your header):
          Context (normalized):
            [t_current = t/N, remaining_release = remaining_release/Rmax, priority_is_p1]
          Per-physician p = 0..P-1:
            [phys{p}_is_preferred,
             phys{p}_cap_remaining_norm = cap_remaining[p]/cap_per_phys (or 1.0 if no cap),
             phys{p}_load_pressure      = work_per_phys[p]/session_time]
          Masks (repeated per physician, as in your header):
            for p = 0..P-1:
              [mask_phys{p}, mask_reject]   # mask_reject is always 0 here
        """
        if state.get("done", False):
            raise ValueError("Terminal state: no current_patient to featurize.")
        cur = state.get("current_patient")
        if cur is None:
            raise ValueError("State missing 'current_patient'.")
    
        P = state.get("P")
        session = self.sim.session_time
        eps = 1e-9
        MAX_OVERTIME = self.sim.MAX_OVERTIME
        # ---------- Context (normalized) ----------
        t = int(state.get("t", 0))
        N = int(state.get("N", max(1, getattr(self, "N", 1))))
        t_current_norm = t / max(1, N)
    
        remaining_release_raw = state.get("remaining_release", None)
        Rmax = state.get("Rmax", self.sim.Rmax)
        if (remaining_release_raw is not None) and (Rmax is not None) and (float(Rmax) > 0):
            remaining_release_norm = float(remaining_release_raw) / max(eps, float(Rmax))
        else:
            remaining_release_norm = 0.0
    
        priority_is_p1 = 1 if int(cur.get("priority", 2)) == 1 else 0
    
        vec: list[float | int] = [t_current_norm, remaining_release_norm, priority_is_p1, cur.get("duration")/60]
    
        # ---------- Per-physician features ----------
        preferred_phys = int(cur.get("preferred_phys", 0))
        # Availability vector at decision time (capacity-based). Fallback: all ones.
        available = cur.get("available_phys")
        eligible = cur.get("eligible_phys")
        if available is None:
            available = [1] * P
        else:
            available = list(available)
        if eligible is None:
            eligible = [1] * P
        else:
            eligible = list(eligible)
        if len(available) != len(eligible):
            raise ValueError("Lists must have the same length.")
        mask =  [x * y for x, y in zip(available, eligible)]

    
        work_per_phys = [float(x) for x in state.get("work_per_phys", [0.0] * P)]
    
        # cap_remaining: prefer provided; else derive from assigned_per_phys if we have a cap
        cap_remaining = state.get("cap_remaining")
        accepted_priority_phy  = state.get("accepted_priority_phy")
        
        if cap_remaining is None:
            if self.cap_per_phys is not None:
                assigned = state.get("assigned_per_phys", [0] * P)
                cap_remaining = [max(0, int(self.cap_per_phys) - int(a)) for a in assigned]
            else:
                cap_remaining = [1_000_000] * P  # effectively infinite when no cap
    
        # Normalized capacity & load
        if (self.sim.cap_per_phys is not None) and (self.sim.cap_per_phys > 0):
            cap_remaining_norm = [cr / max(eps, float(self.sim.cap_per_phys)) for cr in cap_remaining]
        else:
            cap_remaining_norm = [1.0] * P  # neutral if no cap enforced
    
        load_pressure = [w / max(eps, session) for w in work_per_phys]
    
        for p in range(P):
            is_pref = 1 if p == preferred_phys else 0
            vec.extend([
                is_pref,
                cap_remaining_norm[p],
                accepted_priority_phy[p][1],
                #accepted_priority_phy[p][2]
                load_pressure[p],
            ])
    
        # ---------- Masks (availability ∧ capacity), with mask_reject repeated per-phys ----------
        # availability already encodes capacity in your design → use it directly
        for p in range(P):
            mask_phys_p = 1 if int(mask[p]) == 0 else 0
            vec.append(mask_phys_p)
        vec.append(0)  # mask_reject
        return vec
    
    def majority_vote(self, values, policy):
        """
        Returns the label with the highest count (majority vote).
        If there's a tie, returns one of the most common labels arbitrarily.
        Args:
            predictions (list): List of class labels (e.g., [0, 1, 1, 2])
        Returns:
            int or str: Majority-voted label
        """
        if not values:
            raise ValueError("Predictions list is empty.")
    
        counter = Counter(values)
        majority_label = counter.most_common(1)[0][0]
        return majority_label
    
    def sampled_vote(self, values, policy):
        dist = self.get_dist(values)
        return policy.random_action(list(dist.values())), dist
    
    def get_dist(self, values):
        counter = Counter(values)
        total = sum(counter.values())
        for i in range(self.num_class):
            if i not in counter:
                counter[i] = 0
        dist = {k: v / total for k, v in counter.items()}
        dist = dict(sorted(dist.items(), key=lambda kv: kv[0]))
        return dist
        
    
    def get_action_NN(self, state, k, policy, x):
        action, pred = policy.get_action(x)
        pred = np.round(pred, 3)
        self.pred_history.loc[len(self.pred_history)] =  list(pred) + [state['t'], action, k]
        return action, pred
        
    def add_new_data_avg(self, x, gap = 0, obj = 0, sol_time=0, k = 0):
        self.data[len(self.data)] = x + [ gap, obj, sol_time, k]

    def save_model(self, folder):
        self.policy.save_model(folder)
        
