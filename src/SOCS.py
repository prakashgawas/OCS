#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 17:49:05 2025

@author: prakashgawas
"""

import os
import csv
import math
import json
import copy
import time
from typing import List, Tuple, Dict
from pyomo.environ import ConcreteModel, Constraint, Var, Objective, Set
from pyomo.environ import Binary, minimize, NonNegativeReals, value
from pyomo.opt import SolverFactory
from typing import Optional
from OCS import AppointmentScheduler, Patient
from utility import parse_args, build_config
import psutil, threading


def track_memory(interval=0.5):
    proc = psutil.Process(os.getpid())
    peak = 0
    while not stop_flag:
        mem = proc.memory_info().rss
        peak = max(peak, mem)
        time.sleep(interval)
    print(f"[MEMORY] Peak usage: {peak / (1024**3):.2f} GB")

class SAppointmentScheduler():
    def __init__( self,
                 n_patients: int = 100,
                 sigma: int = 8,
                 n_phys: int = 5,
                 n_slots: int = 8,
                 slot_minutes: int = 30,
                 show_prob: float = 1,
                 accept_prob: float = 1,
                 c_wait: int = 1,           # deprecated (unused), kept for API compatibility
                 c_miss_1: int = 80,       # rejection cost for P1
                 c_miss_2: int = 50,        # rejection cost for P2
                 c_overtime: int = 4,       # overtime cost per minute
                 c_np_per:int=10,
                 release_cap: Optional[int] = None,
                 cap_per_phys: int = 25,
                 k_max:Optional[int] = None,
                 k_min:Optional[int] = None,
                 physician_weights: bool = False,
                 seed: int = 123,
                 time_limit: int = 30,
                 mipgap:int = 0,
                 out_dir: str = "../data/Stochastic/",
                 suffix: bool= True,
                 num_scenarios:int = 5):
        
        # Sizes / time structure
        self.N = n_patients
        self.sigma = sigma
        self.P = n_phys
        self.I = n_slots
        self.Delta = slot_minutes
        self.session_time = self.I * self.Delta  # regular minutes per physician

        # Behavior parameters
        self.accept_prob = accept_prob
        self.show_prob = show_prob
        self.p1_fraction = 0.25  # fraction of priority-1 patients

        # Costs
        self.c_miss_1 = c_miss_1
        self.c_miss_2 = c_miss_2
        self.c_o = c_overtime
        self.c_np_per = c_np_per
        self.c_nonpref_1 = c_np_per/100 * c_miss_1
        self.c_nonpref_2 = c_np_per/100 * c_miss_2
        self.MAX_OVERTIME = 60   # Hard cap on overtime (minutes)
        # note: c_wait is intentionally not stored (deprecated / unused)

        # Optional constraints / I/O / reproducibility
        self.cap_per_phys = cap_per_phys
        self.Rmax = release_cap if release_cap is not None else (self.cap_per_phys) * self.P
        self.seed = seed
        self.time_limit = time_limit
        self.mipgap = mipgap
        self.out_dir = out_dir + f"Instances_N{n_patients}_s{sigma}_P{n_phys}_I{n_slots}_L{cap_per_phys}_{k_min}_{k_max}_{physician_weights}_{c_miss_1}_{c_miss_2}_{c_overtime}_{c_np_per}" if suffix else out_dir
        self.eligibility_mode = "dirichlet_topk"
        if  self.eligibility_mode == "dirichlet_topk":
            
            self.k_min = max(1, k_min) if k_min is not None else n_phys
            self.k_max = min(k_max, n_phys) if k_max is not None else n_phys
            if not physician_weights:
                self.physician_weights = [1.0] * n_phys
                self.alpha_dirichlet = float(1)
            else:
                self.alpha_dirichlet = float(25)
                if n_phys == 4:
                    self.physician_weights = [0.4, 0.3, 0.15, 0.15]
                else:
                    self.physician_weights = [0.35, 0.25, 0.2, 0.1, 0.1]

        os.makedirs(self.out_dir, exist_ok=True)

        # Independent RNGs
        self.num_scenarios = num_scenarios

        # Populated by generate_instance()
        self.patients: List[Patient] = []

        self.model: Optional[ConcreteModel] = None
        self.scenarios = []
        
        for i in range(self.num_scenarios):
           bp = AppointmentScheduler(n_patients=n_patients,
                           sigma = sigma,
                           n_phys=n_phys,
                           n_slots=n_slots,
                           k_max=k_max,
                           k_min=k_min,
                           # --- costs wired to your class fields used in build_model ---
                           c_miss_1=c_miss_1,
                           c_miss_2=c_miss_2,
                           c_overtime=c_overtime,
                           c_np_per=c_np_per,
                           physician_weights=physician_weights,
                           release_cap=release_cap,
                           cap_per_phys=cap_per_phys,
                           seed=seed + i)
           self.scenarios.append(bp)
           
    def update_scenarios(self, num:int = 0):
        num = min(num, 40 - self.num_scenarios)
        for i in range(num):
           self.scenarios.append(copy.deepcopy( self.scenarios[-1]))
           self.scenarios[-1].seed = self.seed + i + self.num_scenarios
           self.scenarios[-1]._init_rngs(self.scenarios[-1].seed)
        self.num_scenarios += num
        return self.num_scenarios
        #print("New Scenarios - ", self.num_scenarios)


    def save_instance_csv(
        self,
        tag: str,
        scenarios:list,
        folder: str = "instances",
        sort_by_id: bool = True,
    ) -> Dict[str, str]:
        """
        Save ALL scenarios into two CSVs (single files):
          - '{tag}_patients_all.csv' with columns:
              [scenario, pid, n, priority, duration, preferred_phys, preferred_slot, arrival_score]
          - '{tag}_AYE_all.csv' with columns:
              [scenario, n, p, A_np, Y_np, E_np]
    
        Notes
        -----
        - Requires each scenario to have `patients` populated.
        - Handles legacy slot-level A[(p,i)] by collapsing max over slots.
        - Uses P/I from each scenario (can vary across scenarios).
    
        Returns
        -------
        dict with file paths:
          {"patients_csv": <path>, "aye_csv": <path>}
        """
        if not hasattr(self, "scenarios") or not self.scenarios:
            raise ValueError("No scenarios found on this scheduler (self.scenarios is empty).")
    
        out_dir = self.out_dir if not folder else os.path.join(self.out_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
    
        patients_path = os.path.join(out_dir, f"{tag}_patients_all.csv")
        aye_path      = os.path.join(out_dir, f"{tag}_AYE_all.csv")
    
        # ---------- open both writers once ----------
        with open(patients_path, "w", newline="") as fp_pat, open(aye_path, "w", newline="") as fp_aye:
            w_pat = csv.writer(fp_pat)
            w_aye = csv.writer(fp_aye)
    
            # headers
            w_pat.writerow([
                "scenario", "pid", "n", "priority", "duration",
                "preferred_phys", "preferred_slot", "arrival_score"
            ])
            w_aye.writerow(["scenario", "n", "p", "A_np", "Y_np", "E_np"])
    
            for s_idx, scen in enumerate(scenarios):
                patients = scen.regenerated_patients
                    
                if not patients:
                    raise ValueError(f"Scenario {s_idx} has no patients. Generate instances first.")
    
                # Choose P/I for this scenario
                P = getattr(scen, "P", getattr(self, "P", None))
                I = getattr(scen, "I", getattr(self, "I", None))
                if P is None or I is None:
                    raise ValueError(f"Scenario {s_idx} missing P/I attributes.")
    
                # order cohort
                cohort = list(patients)
                if sort_by_id:
                    cohort.sort(key=lambda q: int(getattr(q, "_id", 0)))
    
                # ---------- patients rows ----------
                for p in cohort:
                    w_pat.writerow([
                        s_idx,
                        getattr(p, "pid", f"pat_{getattr(p, '_id', 0)}"),
                        int(getattr(p, "_id", 0)),
                        int(getattr(p, "priority", 2)),
                        float(getattr(p, "duration", 0.0)),
                        int(getattr(p, "preferred_phys", 0)),
                        int(getattr(p, "preferred_slot", -1)),
                        float(getattr(p, "score", getattr(p, "arrival_score", 0.0))),
                    ])
    
                # helper to detect slot-level A
                def has_slot_A(pat) -> bool:
                    A = getattr(pat, "A", None)
                    return isinstance(A, dict) and any(isinstance(k, tuple) and len(k) == 2 for k in A.keys())
    
                any_slot = any(has_slot_A(p) for p in cohort)
    
                # ---------- A/Y/E rows ----------
                for p in cohort:
                    n = int(getattr(p, "_id", 0))
                    pref_p = int(getattr(p, "preferred_phys", 0))
                    A_dict = getattr(p, "A", {}) or {}
                    Y_dict = getattr(p, "Y", {}) or {}
                    E_dict = getattr(p, "eligible", {}) or {}
    
                    for phys in range(P):
                        # A_np (collapse over slots if needed)
                        if any_slot and has_slot_A(p):
                            a_np = 0
                            for i in range(I):
                                a_np = max(a_np, int(A_dict.get((phys, i), 0)))
                        else:
                            a_np = int(A_dict.get(phys, 0))
    
                        # Y_np (use stored Y; else fallback consistent with your rules)
                        if phys in Y_dict:
                            y_np = float(Y_dict[phys])
                        else:
                            prio = int(getattr(p, "priority", 2))
                            if prio == 2:
                                y_np = 1.0
                            else:
                                y_np = 1.0 if (phys == pref_p) else float(getattr(self, "show_prob", 1.0))
    
                        e_np = int(E_dict.get(phys, 0))
                        w_aye.writerow([s_idx, n, phys, int(a_np), int(y_np), int(e_np)])

                    
    def regenerate_scenario(self, state):
        for i in range(self.num_scenarios):
            self.scenarios[i].regenerate_scenario(state)

    def _sorted_future_from_state(self, patients: List["Patient"], state: Optional[Dict], P: int, I: int):
        """
        Return:
          - future: list[Patient] in increasing _id (full list if state is None;
                    else only patients with _id >= current._id)
          - current_global_id: int or None
          - base_work: list[float] length P
          - cap_limit: Optional[list[int]]   remaining per-phys capacity (or None)
          - release_limit: Optional[int]     remaining accepts allowed (or None)
        """
        if state is None:
            future = sorted(patients, key=lambda q: int(getattr(q, "_id", 0)))
            current_global_id = None
            base_work = [0.0] * P
            cap_limit = None if getattr(self, "cap_per_phys", None) is None else [self.cap_per_phys] * P
            release_limit = None if getattr(self, "Rmax", None) is None else int(self.Rmax)
            return future, current_global_id, base_work, cap_limit, release_limit
    
        # state-aware

        cur = state.get("current_patient", {}) or {}
        current_global_id = int(cur.get("n"))  # prefer explicit _id
    
        # baseline work and capacity
        base_work = list(state.get("work_per_phys", [0.0] * P))
        used_cap = list(state.get("assigned_per_phys", [0] * P))
    
        if len(base_work) != P or len(used_cap) != P:
            raise ValueError("state.work_per_phys / state.assigned_per_phys must be length P.")
    
        if getattr(self, "cap_per_phys", None) is not None:
            cap_limit = [max(0, int(self.cap_per_phys) - int(c)) for c in used_cap]
        else:
            cap_limit = None
    
        if getattr(self, "Rmax", None) is not None:
            rr = state.get("remaining_release")
            if rr is None:
                rr = max(0, int(self.Rmax) - int(state.get("accepted_count", 0)))
            release_limit = int(rr)
        else:
            release_limit = None
    
        # Future cohort: include the current patient and all later ones (by _id)
        full_sorted = sorted(patients, key=lambda q: int(getattr(q, "_id", 0)))

        return full_sorted, current_global_id, base_work, cap_limit, release_limit
    

    def build_model(self,
                    state: Optional[Dict] = None,
                    new: int = 1,
                    name: str = "mip") -> Dict:
        """
        Two-stage stochastic MIP over self.scenarios:
    
        Stage 1 (here-and-now):
          - decide for the next/current patient once: z[p] (assign to p) or r (reject)
        Stage 2 (wait-and-see, scenario-specific):
          - decide for the rest of the patients per scenario: x[s,f,p], u[s,f]
          - overtime B_over[s,p]
    
        Non-anticipativity:
          - For each scenario s, the local row corresponding to the current patient is linked:
                x[s,f0_s,p] == z[p],  u[s,f0_s] == r
        """
        if not hasattr(self, "scenarios") or not self.scenarios:
            raise ValueError("No scenarios found. Ensure self.scenarios is populated.")
    
        S_idx = list(range((self.num_scenarios)))
        P = int(self.P)
        Pset = list(range(P))
    
        # ---- Build per-scenario "future" lists aligned to the same 'state' ----
        future_by_s: Dict[int, List["Patient"]] = {}
        current_f_by_s: Dict[int, Optional[int]] = {}
        base_work = None
        cap_limit = None
        release_limit = None
    
        for s in S_idx:
            sc = self.scenarios[s]
            patients = getattr(sc, "regenerated_patients", None)
                
            fut_s, current_gid_s, base_work_s, cap_limit_s, release_limit_s = self._sorted_future_from_state(
                patients, state, P, sc.I
            )
           
            future_by_s[s] = fut_s
            current_f_by_s[s] = 0
    
            # sanity-sync shared baseline vectors/caps/releases across scenarios
            if base_work is None:
                base_work = base_work_s
            else:
                if base_work != base_work_s:
                    # small tolerance on float equality if desired
                    pass
    
            if cap_limit is None:
                cap_limit = cap_limit_s
            else:
                if (cap_limit is None) != (cap_limit_s is None):
                    raise ValueError("Inconsistent cap_limit presence across scenarios.")
                if cap_limit is not None and cap_limit != cap_limit_s:
                    pass  # we accept identical values; if different, decide policy (min? assert equal?)
    
            if release_limit is None:
                release_limit = release_limit_s
            else:
                if (release_limit is None) != (release_limit_s is None):
                    raise ValueError("Inconsistent release_limit presence across scenarios.")
                if release_limit is not None and release_limit != release_limit_s:
                    pass
    
        # fallbacks
        if base_work is None:
            base_work = [0.0] * P
    
        # ---- Flatten (s,f) pairs into one index UF to keep Pyomo simple ----
        UF = []
        sf2idx: Dict[Tuple[int, int], int] = {}
        for s in S_idx:
            for f in range(len(future_by_s[s])):
                sf2idx[(s, f)] = len(UF)
                UF.append((s, f))
    
        # Helper accessors for data
        def _prio(s, f) -> int:
            return int(getattr(future_by_s[s][f], "priority", 2))
    
        def _pref(s, f) -> int:
            return int(getattr(future_by_s[s][f], "preferred_phys", 0))
    
        def _dur(s, f) -> float:
            return float(getattr(future_by_s[s][f], "duration", 0.0))
    
        def _A(s, f, p) -> int:
            A = getattr(future_by_s[s][f], "A", {}) or {}
            return int(A.get(p, 0))
    
        def _E(s, f, p) -> int:
            E = getattr(future_by_s[s][f], "eligible", {}) or {}
            return int(E.get(p, 0))
    
        def _Y(s, f, p) -> float:
            Y = getattr(future_by_s[s][f], "Y", {}) or {}
            return float(Y.get(p, 1.0))
    
        # ---- Build model ----
        m = ConcreteModel(name=name)
    
        # Sets
        m.S = Set(initialize=S_idx, ordered=True)
        m.P = Set(initialize=Pset, ordered=True)
        m.UF = Set(initialize=list(range(len(UF))), ordered=True)
    
        # Back maps for convenience inside rules
        m._UF_to_sp = {u: UF[u] for u in range(len(UF))}
        m._current_f_by_s = current_f_by_s
    
        # First-stage (here-and-now) variables for current patient
        # NOTE: Only meaningful if state is provided; otherwise we still create them,
        # and link them to the first row per scenario (or you can skip non-anticipativity if state is None).
        m.z = Var(m.P, domain=Binary)        # assign to p
        m.r = Var(domain=Binary)             # reject here-and-now
        m.HereNow = Constraint(expr=sum(m.z[p] for p in m.P) + m.r == 1)
    
        # Second-stage variables per (s,f) and p
        m.x = Var(m.UF, m.P, domain=Binary)        # assign scenario-future row to p
        m.u = Var(m.UF, domain=Binary)             # reject scenario-future row
        m.B_over = Var(m.S, m.P, domain=NonNegativeReals)
    
        # Assign-or-reject for each (s,f):
        def _assign_or_reject_rule(m, u):
            s, f = m._UF_to_sp[u]
            return sum(m.x[u, p] for p in m.P) + m.u[u] == 1
        m.AssignOrReject = Constraint(m.UF, rule=_assign_or_reject_rule)
    
        # Acceptability & eligibility: x <= A*E for each (s,f,p)
        def _accept_rule(m, u, p):
            s, f = m._UF_to_sp[u]
            return m.x[u, p] <= _A(s, f, p) * _E(s, f, p)
        m.Acceptability = Constraint(m.UF, m.P, rule=_accept_rule)
    
        # Capacity per scenario: sum_f x[s,f,p] <= remaining cap
        if cap_limit is not None:
            def _cap_rule(m, s, p):
                # sum over all rows belonging to scenario s
                rows = [u for u in m.UF if m._UF_to_sp[u][0] == s]
                return sum(m.x[u, p] for u in rows) <= int(cap_limit[p])
            m.Capacity = Constraint(m.S, m.P, rule=_cap_rule)
    
        # Release cap per scenario: sum_f (1 - u) <= remaining release
        if release_limit is not None:
            def _release_rule(m, s):
                rows = [u for u in m.UF if m._UF_to_sp[u][0] == s]
                return sum(1 - m.u[u] for u in rows) <= int(release_limit)
            m.ReleaseCap = Constraint(m.S, rule=_release_rule)
    
        # Overtime per scenario:
        def _overtime_rule(m, s, p):
            rows = [u for u in m.UF if m._UF_to_sp[u][0] == s]
            future_work = sum(
                _dur(s, m._UF_to_sp[u][1]) * _Y(s, m._UF_to_sp[u][1], p) * m.x[u, p]
                for u in rows
            )
            return m.B_over[s, p] >= (float(base_work[p]) + future_work) - float(self.session_time)
        m.Overtime = Constraint(m.S, m.P, rule=_overtime_rule)
    
        # Optional overtime cap to avoid runaway
        def _overtime_cap_rule(m, s, p):
            return m.B_over[s, p] <= float(self.MAX_OVERTIME)
        m.OvertimeCap = Constraint(m.S, m.P, rule=_overtime_cap_rule)
    
        # Non-anticipativity (link the "current" row per scenario to first-stage z/r)
        if state is not None:
            def _link_x_rule(m, s, p):
                f0 = m._current_f_by_s[s]
                # map to UF index
                u = sf2idx[(s, f0)]
                return m.x[u, p] == m.z[p]
            m.LinkX = Constraint(m.S, m.P, rule=_link_x_rule)
    
            def _link_u_rule(m, s):
                f0 = m._current_f_by_s[s]
                u = sf2idx[(s, f0)]
                return m.u[u] == m.r
            m.LinkU = Constraint(m.S, rule=_link_u_rule)
    
        # Objective: average over scenarios (use weights if you have them; here uniform)
        def _rej_cost(s, f):
            return self.c_miss_1 if _prio(s, f) == 1 else self.c_miss_2
    
        def _nonpref_cost(s, f, p):
            pr = _prio(s, f)
            pref = _pref(s, f)
            if pr == 1:
                return (self.c_nonpref_1 if p != pref else 0.0)
            else:
                return (self.c_nonpref_2 if p != pref else 0.0)
    
        # Build components
        def _reject_pen_s(s):
            rows = [u for u in range(len(UF)) if m._UF_to_sp[u][0] == s]
            return sum(_rej_cost(s, m._UF_to_sp[u][1]) * m.u[u] for u in rows)
    
        def _nonpref_pen_s(s):
            rows = [u for u in range(len(UF)) if m._UF_to_sp[u][0] == s]
            return sum(_nonpref_cost(s, m._UF_to_sp[u][1], p) * m.x[u, p] for u in rows for p in m.P)
    
        def _overtime_pen_s(s):
            return self.c_o * sum(m.B_over[s, p] for p in m.P)
    
        scen_terms = []
        for s in S_idx:
            scen_terms.append(_reject_pen_s(s) + _nonpref_pen_s(s) + _overtime_pen_s(s))
    
        m.Obj = Objective(expr=(1.0 / max(1, len(S_idx))) * sum(scen_terms), sense=minimize)
    
        # ---- Output info ----
        info = {
            "mode": ("state" if state is not None else "full"),
            "name": name,
            "t": (None if state is None else int(state["t"])),
            "num_scenarios": len(S_idx),
            "future_len_per_scenario": [len(future_by_s[s]) for s in S_idx],
            "remaining_release": release_limit,
            "remaining_cap_per_phys": cap_limit,
            "baseline_work_per_phys": base_work
        }
    
        self.model = m
        return info

        
    def setup_solver(self, name: str = "gurobi"):
        """Create solver and set time limit (you can add MIPGap, OutputFlag, etc.)."""
        self.solver_name = name
        self.solver = SolverFactory(name, manage_env=True)
        if self.solver is None:
            raise RuntimeError(f"Solver {name} not available")
        self.solver.options["TimeLimit"] = self.time_limit
        self.solver.options["Threads"] = 2   
        self.solver.options["MIPGap"] = self.mipgap 


    def solve(self, tee: bool = False):
        """Return (status, termination_condition, mip_gap_pct, wall_time_sec)."""
        assert self.model is not None
        #start = time.time()
        res = self.solver.solve(self.model, tee=tee)
        #duration = time.time() - start
        duration = res.solver.time
        status = str(res.solver.status)
        term = str(res.solver.termination_condition)
        # Try to compute MIP gap
        try:
            problem_data = res.Problem._list[0]
            LB = float(problem_data.lower_bound)
            UB = float(problem_data.upper_bound)
            if math.isfinite(UB) and math.isfinite(LB):
                gap = abs(UB - LB) * 100 / max(1e-5, abs(UB))
            else:
                gap = None
        except (AttributeError, IndexError, ZeroDivisionError, TypeError):
            gap = None
        return status, term, gap, duration


    def extract_solution(self) -> Dict:
        """
        Extract solution per scenario for the stochastic two-stage model.
    
        Returns:
          {
            "scenarios": [
               {
                 "s": <scenario index>,
                 "assignments": [...],
                 "rejected": [...],
                 "overtime": [...],
                 "totals": {...},
                 "objective": <float>
               }, ...
            ],
            "overall": {
               "mean_objective": <float>,
               "mean_reject_pen": <float>,
               "mean_nonpref_pen": <float>,
               "mean_overtime_pen": <float>,
               "num_scenarios": <int>
            }
          }
        """
        assert self.model is not None, "Call after solving build_model(...)"
        m = self.model
    
        # ---------- helpers & fallbacks ----------
        # Build patient lookup per scenario:
        # Prefer model-attached maps created in build_model; else reconstruct here.
        # Expect either:
        #   m._nmap_by_s: {s: {f_local: global_id}}
        # and optionally
        #   m._id2pat_by_s: {s: {global_id: Patient}}
        nmap_by_s = getattr(m, "_nmap_by_s", None)
        id2pat_by_s = getattr(m, "_id2pat_by_s", None)
    
        if nmap_by_s is None:
            # Reconstruct f->global_id from the scenarios' patients (sorted by _id).
            nmap_by_s = {}
            for s in list(m.S):
                pats = sorted(self.scenarios[s].regenerated_patients, key=lambda q: int(getattr(q, "_id", 0)))
                nmap_by_s[s] = {f: int(pats[f]._id) for f in range(len(pats))}
            # Attach for reuse
            m._nmap_by_s = nmap_by_s
    
        if id2pat_by_s is None:
            id2pat_by_s = {}
            for s in list(m.S):
                pats = self.scenarios[s].regenerated_patients
                id2pat_by_s[s] = {int(getattr(p, "_id", 0)): p for p in pats}
            m._id2pat_by_s = id2pat_by_s
    
        # Helper to fetch Patient for a (s,f)
        def _get_patient(s: int, f: int):
            gid = nmap_by_s[s][f]
            return id2pat_by_s[s][gid]
    
        Pset = list(m.P)
    
        # ---------- per-scenario extraction ----------
        scen_outputs: List[Dict] = []
        for s in list(m.S):
            # rows (u) that belong to scenario s
            rows_u = [u for u in list(m.UF) if m._UF_to_sp[u][0] == s]
    
            sol_s = {
                "s": int(s),
                "assignments": [],
                "rejected": [],
                "overtime": [],
                "totals": {
                    "reject_pen": 0.0,
                    "overtime_pen": 0.0,
                    "nonpref_pen": 0.0,
                    "objective": 0.0
                }
            }
    
            # patients loop
            for u in rows_u:
                s_u, f_u = m._UF_to_sp[u]
                pat = _get_patient(s_u, f_u)
                chosen = [p for p in Pset if value(m.x[u, p]) > 0.5]
    
                if not chosen and value(m.u[u]) > 0.5:
                    pen = float(self.c_miss_1 if int(pat.priority) == 1 else self.c_miss_2)
                    sol_s["rejected"].append({
                        "kind": "reject",
                        "pid": getattr(pat, "pid", f"pat_{f_u}"),
                        "n": int(getattr(pat, "_id", f_u)),
                        "p": -1,
                        "priority": int(pat.priority),
                        "score": getattr(pat, "score", None),
                        "duration": float(pat.duration),
                        "preferred_phys": int(getattr(pat, "preferred_phys", 0)),
                        "eligible": list(getattr(pat, "eligible", {}).values()),
                        "penalty": pen,
                    })
                    sol_s["totals"]["reject_pen"] += pen
    
                for p in chosen:
                    Y_np = float(getattr(pat, "Y", {}).get(p, 1.0))
                    nonpref = int(p != int(getattr(pat, "preferred_phys", 0)))
                    nonpref_cost = (
                        self.c_nonpref_1 if int(pat.priority) == 1 else self.c_nonpref_2
                    ) * nonpref
    
                    work_min = float(pat.duration) * Y_np
    
                    sol_s["assignments"].append({
                        "kind": "assign",
                        "pid": getattr(pat, "pid", f"pat_{f_u}"),
                        "n": int(getattr(pat, "_id", f_u)),
                        "p": int(p),
                        "priority": int(pat.priority),
                        "score": getattr(pat, "score", None),
                        "duration": float(pat.duration),
                        "preferred_phys": int(getattr(pat, "preferred_phys", 0)),
                        "eligible": list(getattr(pat, "eligible", {}).values()),
                        "nonpref_flag": nonpref,
                        "nonpref_cost": float(nonpref_cost),
                        "Y_np": float(Y_np),
                        "work_minutes": float(work_min),
                    })
                    sol_s["totals"]["nonpref_pen"] += float(nonpref_cost)
    
            # overtime for this scenario
            over_total = 0.0
            for p in Pset:
                Bover = float(value(m.B_over[s, p]))
                sol_s["overtime"].append({"p": int(p), "B_over": Bover})
                over_total += Bover
            sol_s["totals"]["overtime_pen"] = float(self.c_o) * over_total
    
            # scenario objective contribution = what your model actually averages
            # If m.Obj is averaged, Pyomo gives global objective; we recompute the scenario term:
            #   reject_pen + nonpref_pen + c_o * sum B_over[s, p]
            scen_obj = (
                sol_s["totals"]["reject_pen"]
                + sol_s["totals"]["nonpref_pen"]
                + sol_s["totals"]["overtime_pen"]
            )
            sol_s["totals"]["objective"] = float(scen_obj)
            sol_s["objective"] = float(scen_obj)
    
            scen_outputs.append(sol_s)
    
        # ---------- overall summary ----------
        S = max(1, len(scen_outputs))
        mean_obj = sum(s["objective"] for s in scen_outputs) / S
        mean_rej = sum(s["totals"]["reject_pen"] for s in scen_outputs) / S
        mean_np  = sum(s["totals"]["nonpref_pen"] for s in scen_outputs) / S
        mean_ot  = sum(s["totals"]["overtime_pen"] for s in scen_outputs) / S
    
        sol = {
            "scenarios": scen_outputs,
            "overall": {
                "mean_objective": float(mean_obj),
                "mean_reject_pen": float(mean_rej),
                "mean_nonpref_pen": float(mean_np),
                "mean_overtime_pen": float(mean_ot),
                "num_scenarios": int(S),
            }
        }
        return sol, mean_obj
    
    def get_current_action(self, i):
        if self.model.r.value > 0.5:
            action = self.P
        else:
            for p in range(self.P):
                if self.model.z[p].value > 0.5:
                    action = p
                    break
        return action
    
    
    def save_solution_files(
        self,
        tag: str,
        sol: List[Dict],
        status: str,
        term: str,
        gap: float,
        solve_time: float,
        folder: str = "instances_solution",
    ):
        """
        Save one CSV and one JSON for ALL scenarios.
    
        CSV (one file): '{tag}_scenarios_schedule.csv'
          Columns include per-scenario rows for assign/reject and overtime,
          sorted by (scenario s, n, kind), with per-scenario cumulative counters.
    
        JSON (one file): '{tag}_scenarios_solution.json'
          Packs per-scenario totals and an overall summary (mean penalties/objective).
    
        Args
        ----
        scenarios : list of dicts
            Exactly out["scenarios"] produced by extract_solution_all_scenarios().
            Each item must have keys: 's', 'assignments', 'rejected', 'overtime', 'totals', 'objective'.
        """
        out_dir = self.out_dir if not folder else os.path.join(self.out_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
    
        # ---------- CSV (per-scenario, chronological by patient n) ----------
        csv_path = os.path.join(out_dir, f"{tag}_scenarios_schedule.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "scenario", "kind", "pid", "n", "p", "priority", "score",
                "preferred_phys", "eligible", "duration", "Y_np", "work_minutes",
                "penalty", "B_over", "row_cost",
                "cum_accept_total", "cum_accept_p1", "cum_accept_p2",
                "cum_reject_total", "cum_reject_p1", "cum_reject_p2"
            ])
    
            # For each scenario, build chronological event stream and write
            kind_order = {"assign": 0, "reject": 1}
            for scen in sol:
                s = int(scen["s"])
                # Build events for this scenario
                events = []
                for rec in scen.get("assignments", []):
                    events.append({
                        "kind": "assign",
                        "pid": rec.get("pid", ""),
                        "n": int(rec["n"]),
                        "p": int(rec["p"]),
                        "priority": int(rec["priority"]),
                        "score": rec.get("score", None),
                        "preferred_phys": int(rec.get("preferred_phys", -1)),
                        "eligible": rec.get("eligible", []),
                        "duration": float(rec.get("duration", 0.0)),
                        "Y_np": float(rec.get("Y_np", 0.0)),
                        "work_minutes": float(rec.get("work_minutes", 0.0)),
                        "penalty": 0.0,
                        "B_over": 0.0,
                        "row_cost": 0.0,
                    })
                for rec in scen.get("rejected", []):
                    pen = float(rec.get("penalty", 0.0))
                    events.append({
                        "kind": "reject",
                        "pid": rec.get("pid", ""),
                        "n": int(rec["n"]),
                        "p": -1,
                        "priority": int(rec["priority"]),
                        "score": rec.get("score", None),
                        "preferred_phys": int(rec.get("preferred_phys", -1)),
                        "eligible": rec.get("eligible", []),
                        "duration": float(rec.get("duration", 0.0)),
                        "Y_np": 0.0,
                        "work_minutes": 0.0,
                        "penalty": pen,
                        "B_over": 0.0,
                        "row_cost": pen,
                    })
    
                # Sort by (n, kind), then emit with per-scenario cumulative counters
                events.sort(key=lambda r: (r["n"], kind_order.get(r["kind"], 99)))
    
                cum_total = 0
                cum_p1 = 0
                cum_p2 = 0
                cum_reject = 0
                cum_reject_p1 = 0
                cum_reject_p2 = 0
    
                for ev in events:
                    if ev["kind"] == "assign":
                        cum_total += 1
                        if ev["priority"] == 1:
                            cum_p1 += 1
                        elif ev["priority"] == 2:
                            cum_p2 += 1
                    else:  # reject
                        cum_reject += 1
                        if ev["priority"] == 1:
                            cum_reject_p1 += 1
                        elif ev["priority"] == 2:
                            cum_reject_p2 += 1
    
                    w.writerow([
                        s, ev["kind"], ev["pid"], ev["n"], ev["p"], ev["priority"],
                        ev["score"], ev["preferred_phys"], ev["eligible"],
                        ev["duration"], ev["Y_np"], ev["work_minutes"],
                        ev["penalty"], ev["B_over"], ev["row_cost"],
                        cum_total, cum_p1, cum_p2, cum_reject, cum_reject_p1, cum_reject_p2
                    ])
    
                # Overtime rows for this scenario (do not change cumulative counters)
                for o in scen.get("overtime", []):
                    p = int(o["p"]); b = float(o.get("B_over", 0.0))
                    w.writerow([
                        s, "overtime", "", -1, p, -1,
                        -1, -1, [],
                        0.0, 0.0, 0.0,
                        0.0, b, float(self.c_o) * b,
                        cum_total, cum_p1, cum_p2, cum_reject, cum_reject_p1, cum_reject_p2
                    ])
    
        # ---------- JSON meta / totals (per-scenario + overall) ----------
        # per-scenario totals + objectives
        scen_totals = []
        for scen in sol:
            s = int(scen["s"])
            t = scen.get("totals", {})
            scen_totals.append({
                "scenario": s,
                "reject_pen": float(t.get("reject_pen", 0.0)),
                "overtime_pen": float(t.get("overtime_pen", 0.0)),
                "nonpref_pen": float(t.get("nonpref_pen", 0.0)),
                "objective": float(scen.get("objective", 0.0)),
            })
    
        # overall (means)
        S = max(1, len(sol))
        mean_obj = sum(x["objective"] for x in scen_totals) / S
        mean_rej = sum(x["reject_pen"] for x in scen_totals) / S
        mean_np  = sum(x["nonpref_pen"] for x in scen_totals) / S
        mean_ot  = sum(x["overtime_pen"] for x in scen_totals) / S
    
        meta = {
            "status": status,
            "termination": term,
            "gap": gap,
            "solve_time": solve_time,
            "num_scenarios": S,
            "overall": {
                "mean_objective": float(mean_obj),
                "mean_reject_pen": float(mean_rej),
                "mean_nonpref_pen": float(mean_np),
                "mean_overtime_pen": float(mean_ot),
            },
            "params": {
                "N": getattr(self, "N", None),
                "P": getattr(self, "P", None),
                "session_time": getattr(self, "session_time", None),
                "Delta": getattr(self, "Delta", None),
                "I": getattr(self, "I", None),
                "c_miss_1": getattr(self, "c_miss_1", None),
                "c_miss_2": getattr(self, "c_miss_2", None),
                "c_o": getattr(self, "c_o", None),
                "Rmax": getattr(self, "Rmax", None),
                "cap_per_phys": getattr(self, "cap_per_phys", None),
            },
        }
    
        json_payload = {
            "meta": meta,
            "scenarios": scen_totals,
        }
        json_path = os.path.join(out_dir, f"{tag}_scenarios_solution.json")
        with open(json_path, "w") as f:
            json.dump(json_payload, f, indent=2)


if __name__ == "__main__":
    stop_flag = False
    thread = threading.Thread(target=track_memory)
    thread.start()
    
    args = parse_args()
    config = build_config(args)
    
    sched = SAppointmentScheduler(
        n_patients=config.ocs_param['N'],
        sigma=config.ocs_param['sigma'],
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
        num_scenarios=config.scenarios
    )
    
    
    tag = f"OCS_{config.seed}"

    sched.setup_solver()
    patient = sched.scenarios[0].generate_patient(0)
    state =   {'t': 0, 'N': 100, 'N_total': 97, 'P': 4, 'Rmax': 80,
               'session_time': 240, 'patients_arrived': 1,
               'current_patient': {'n': 0, 'pid': patient.pid, 'priority': patient.priority,
                                   'score': patient.score, 'duration': patient.duration,
                                   'preferred_phys': patient.preferred_phys, 'available_phys': [1, 1, 1, 1],
                                   'eligible_phys': patient.eligible}, 'work_per_phys': [0.0, 0.0, 0.0, 0.0],
               'assigned_per_phys': [0, 0, 0, 0], 'cap_remaining': [20, 20, 20, 20], 'accepted_count': 0,
               'accepted_priority_phy': {0: {1: 0, 2: 0}, 1: {1: 0, 2: 0}, 2: {1: 0, 2: 0}, 3: {1: 0, 2: 0}},
               'remaining_release': 80, 'done': False}
    #sched.update_scenarios(1)
    sched.regenerate_scenario(state)
    sched.build_model(state=state)
    status, termination, gap, sol_time = sched.solve(tee=True)
    sol, obj = sched.extract_solution()
    print(f"   Solver status = {status}, termination = {termination}, MIP gap = {gap}, Time = {sol_time}")
    sched.save_instance_csv(tag, sched.scenarios)
    sched.save_solution_files(tag,  sol["scenarios"], status, termination, gap, sol_time)

    print("Current Action - ", sched.get_current_action(0))