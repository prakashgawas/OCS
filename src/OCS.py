#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 10:33:12 2025

@author: prakashgawas
"""

import os
import csv
import math
import time
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pyomo.environ import ConcreteModel, Constraint, Var, Objective
from pyomo.environ import Binary, NonNegativeIntegers, maximize, minimize, value, NonNegativeReals
from pyomo.opt import SolverFactory
from utility import parse_args, build_config
from typing import Optional
import psutil, threading


def track_memory(interval=0.5):
    proc = psutil.Process(os.getpid())
    peak = 0
    while not stop_flag:
        mem = proc.memory_info().rss
        peak = max(peak, mem)
        time.sleep(interval)
    print(f"[MEMORY] Peak usage: {peak / (1024**3):.2f} GB")

@dataclass
class Patient:
    pid: str            # external id
    _id: int            # internal id (0..N-1)
    duration: float     # service duration in minutes (if shows)
    preferred_phys: int # index of preferred physician (0..P-1)
    preferred_slot:int
    priority: int      
    A: Dict[Tuple[int, int], int]  # acceptability for (p,i)
    Y: Dict[int, int]        # pre-sampled show indicator by physician p
    score:float
    eligible: Dict[int, int]  
# ------------------------------------------------------------
# Appointment Scheduler (Full-Information via Pre-Sampled Shows)
# ------------------------------------------------------------

class AppointmentScheduler:
    """
    Offline physician-assignment MIP (no slot dimension).

    Show/no-show model:
      - Priority-1 (P1): preferred physician => show with certainty (1);
                         non-preferred => Bernoulli(show_prob).
      - Priority-2 (P2): always show (1) on any physician.

    We pre-sample per-patient attendance indicators Y[n,p] once
    and treat them as constants to approximate expected workload/overtime.
    """
    def __init__(
        self,
        n_patients: int = 100,
        sigma: int = 8,
        n_phys: int = 5,
        n_slots: int = 8,
        slot_minutes: int = 30,
        show_prob: float = 1,
        accept_prob: float = 1,
        c_wait: int = 1,           # deprecated (unused), kept for API compatibility
        c_miss_1: int = 80,        # rejection cost for P1
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
        out_dir: str = "../data/Deterministic/",
        suffix: bool= True
    ):
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
        self._init_rngs(seed)

        # Populated by generate_instance()
        self.patients: List[Patient] = []

        self.model: Optional[ConcreteModel] = None

    def _init_rngs(self, seed: int):
        """
        Keep separate RNGs but ensure independence using SeedSequence splits.
        """
        ss = np.random.SeedSequence(seed)
        streams = ss.spawn(12)  # make plenty for future use
        self.duration_generator = np.random.default_rng(streams[0])
        self.phys_generator = np.random.default_rng(streams[1])
        self.show_generator = np.random.default_rng(streams[2])
        self.rank_generator = np.random.default_rng(streams[3])     # arrival order score
        #self.jitter_generator = np.random.default_rng(streams[4])   # tie-break jitter
        self.rejection_generator = np.random.default_rng(streams[5])# acceptability A
        # Reserved / presently unused but preserved for API symmetry:
        self.slot_generator = np.random.default_rng(streams[6])
        self.priority_generator = np.random.default_rng(streams[7])
        self.score_generator = np.random.default_rng(streams[8])
        self.total_p_generator = np.random.default_rng(streams[9])
        self.eligibile_phys_generator = np.random.default_rng(streams[10])
        self.K_generator = np.random.default_rng(streams[11])

    def set_seed(self, seed: int):
        """Public reseed API (keeps multiple independent streams)."""
        self._init_rngs(seed)

    # ------------------------------------------------------------
    # Synthetic instance generation
    # ------------------------------------------------------------

    def generate_instance(self):
        """
        Create N patients and then impose an arrival order by sorting on `score`
        (beta-distributed, P1 skew late; P2 uniform) with small jitter for stability.
        After sorting, overwrite IDs to reflect arrival order.
        """
        self.patients.clear()
        self.N_total = int(round(self.total_p_generator.normal(loc=self.N, scale=self.sigma)))
        for n in range(self.N_total):
            self.patients.append(self.generate_patient(n))

        # jitter to break ties deterministically
        #jitter = self.jitter_generator.uniform(0, 1e-9, size=len(self.patients))
        ordered = sorted(self.patients, key=lambda p: (p.score ))#+ jitter[p._id]

        # Overwrite IDs to arrival order
        for rank, p in enumerate(ordered):
            p._id = rank
            p.pid = f"pat_{p.priority}_{rank}"  # optional: encode priority+order
        self.patients = ordered
        
    def _eligible_dirichlet_topk(self) -> tuple[dict[int, int], np.ndarray]:
        """
        Sample an eligible set E first using patient-specific tastes:
          - taste ~ Dirichlet(alpha * weights)
          - K ~ Uniform[k_min, k_max]
          - Eligible = Top-K by taste
        Returns:
          E: dict p -> {0,1}
          taste: np.ndarray shape (P,) for downstream use (e.g., to choose preferred)
        """
        K = int(self.K_generator.integers(self.k_min, self.k_max + 1))
    
        w = np.asarray(self.physician_weights, dtype=float)
        w = np.clip(w, 1e-9, None)
        w = w / w.sum()
    
        alpha = self.alpha_dirichlet * w
        taste = self.eligibile_phys_generator.dirichlet(alpha)  # patient taste over phys
        taste = taste + 1e-9 * np.random.standard_normal(self.P)  # tie-break noise
    
        # Top-K eligible
        top_idx = np.argsort(taste)[-K:]
        E = {p: 0 for p in range(self.P)}  # NOTE: self.P (not self.PP)
        for p in top_idx:
            E[int(p)] = 1
    
        # Safety: ensure ≥1 eligible (shouldn’t trigger, but guard anyway)
        if sum(E.values()) == 0:
            E[int(np.argmax(taste))] = 1
        return E, taste

    def sample_duration(self, priority: int) -> float:
        """
        Truncated lognormal service time (minutes), shorter for P1 on average.
        """
        if priority == 2:
            mu, sigma = 2.3, 0.3
        else:
            mu, sigma = 3, 0.8
        val = self.duration_generator.lognormal(mean=mu, sigma=sigma)
        return int(np.round(np.clip(val, 4, 60), 0))  # truncate to [4, 60]
    
    def _sample_eligible_set(self) -> tuple[dict[int, int], np.ndarray]:
        """
        Unified eligibility sampler (no preferred input):
          - If mode == "dirichlet_topk": use Dirichlet taste + Top-K to build E, return (E, taste)
          - Else: Bernoulli(eligibility_prob) per physician to build E, return (E, weights_for_pick)
                  (weights_for_pick defaults to normalized global physician_weights)
        """
        mode = str(getattr(self, "eligibility_mode", "dirichlet_topk")).lower()
        if mode == "dirichlet_topk":
            return self._eligible_dirichlet_topk()
    
    def generate_patient(self, n: int) -> Patient:
        """
        Generate a single patient:
          - priority in {1,2} with P(P1)=p1_fraction
          - duration ~ truncated lognormal
          - sample eligible set E first
          - choose preferred_phys from eligible (Dirichlet-proportional if available)
          - acceptability A[p]: 0 if not eligible; if eligible → 1 for preferred, else Bernoulli(accept_prob)
          - attendance Y[p]: P2 shows on all; P1 shows on preferred, else Bernoulli(show_prob)
        """
        # Priority & duration
        prio = 1 if self.priority_generator.uniform() < self.p1_fraction else 2
        duration = self.sample_duration(prio)
    
        # Arrival score (P1 skew late; P2 uniform)
        a, b = (3.0, 1.0) if prio == 1 else (1.0, 1.0)
        score = float(self.rank_generator.beta(a, b))
    
        # --- Eligible set first ---
        E_map, weights = self._sample_eligible_set()  # weights is taste (Dirichlet) or global w
        elig_idx = [p for p, e in E_map.items() if e == 1]
    
        # choose preferred_phys from eligible (proportional to weights restricted to eligible)
        w_elig = np.asarray([weights[p] for p in elig_idx], dtype=float)
        w_elig = np.clip(w_elig, 1e-12, None)
        w_elig = w_elig / w_elig.sum()
        preferred_phys = int(self.phys_generator.choice(elig_idx, p=w_elig))
    
        # --- Acceptability map A[p] ---
        A_map: dict[int, int] = {}
        for p in range(self.P):
            if E_map.get(p, 0) == 0:
                A_map[p] = 0
            else:
                A_map[p] = 1 if p == preferred_phys else int(self.rejection_generator.uniform() < self.accept_prob)
    
        # --- Attendance Y[p] ---
        Y_map: dict[int, float] = {}
        if prio == 2:
            for p in range(self.P):
                Y_map[p] = 1
        else:
            for p in range(self.P):
                if p == preferred_phys:
                    Y_map[p] = 1
                else:
                    Y_map[p] = int(self.show_generator.uniform() < self.show_prob)
    
        return Patient(
            pid=f"pat_{n}",
            _id=n,
            duration=float(duration),
            preferred_phys=int(preferred_phys),
            preferred_slot=-1,   # no-slot model
            priority=int(prio),
            A=A_map,
            Y=Y_map,
            score=score,
            eligible=E_map,      # keep the eligibility snapshot if useful downstream
        )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    def build_model(
        self,
        patients: List[Patient],
        state: Optional[Dict] = None,
        new: int = 1,
        name: str = "mip",
    ):
        """
        Unified MIP builder over an explicit patient list.
    
        Behavior
        --------
        - state is None  -> Full model over *all* `patients`.
        - state provided -> State-aware model over regenerated horizon,
                           **including the current patient** (all with _id >= current._id).
    
        Returns
        -------
        model : ConcreteModel
        info  : dict with diagnostics:
            {
              "mode": "full" | "state",
              "name": name,
              "t": Optional[int],
              "ref_id": Optional[int],
              "future_len": int,
              "remaining_release": Optional[int],
              "remaining_cap_per_phys": Optional[List[int]],
              "baseline_work_per_phys": List[float],
              "nmap": {local_index -> global _id}
            }
    
        -----------------------
        State fields read (when state is provided)
        -----------------------
        - state["t"]: int
        - state["current_patient"]["_id"] (preferred; else fallback via t)
        - state["work_per_phys"]: List[float] length P (baseline overtime minutes)
        - state["assigned_per_phys"]: List[int] length P (used to compute remaining cap)
        - state.get("remaining_release"): Optional[int]; else derived from Rmax - accepted_count
        - state.get("accepted_count"): int (only if remaining_release missing)
        """
        # ---------- Determine cohort & parameters ----------
        if state is None:
            mode = "full"
            future = list(patients)  # use exactly what was passed
            t_val = None
    
            # Baseline vectors for unified rules
            base_work = [0.0] * self.P
            # Capacity: use original per-phys cap or disable if None
            cap_limit = [self.cap_per_phys] * self.P if self.cap_per_phys is not None else None
            # Release: use original Rmax or disable if None
            release_limit = int(self.Rmax) if self.Rmax is not None else None
    
        else:
            mode = "state"
            t_val = int(state["t"])
            # Baselines
            base_work = list(state.get("work_per_phys", [0.0] * self.P))
                
            used_cap = list(state.get("assigned_per_phys", [0] * self.P))
            if len(base_work) != self.P or len(used_cap) != self.P:
                raise ValueError("state.work_per_phys / state.assigned_per_phys must be length P.", base_work)

            future = sorted(patients, key=lambda q: int(q._id))

            # Capacity limit = remaining per-phys capacity
            if self.cap_per_phys is not None:
                cap_limit = [max(0, int(self.cap_per_phys) - int(c)) for c in used_cap]
            else:
                cap_limit = None
    
            # Release limit = remaining release
            if self.Rmax is not None:
                rr = state.get("remaining_release")
                if rr is None:
                    rr = max(0, int(self.Rmax) - int(state.get("accepted_count", 0)))
                release_limit = int(rr)
            else:
                release_limit = None
    
    
        # ---------- Build common model (no duplicated rules) ----------
        m = ConcreteModel()
        F = list(range(len(future)))
        Pset = list(range(self.P))
        nmap = {f: int(future[f]._id) for f in F}

        # Variables
        m.x = Var(F, Pset, domain=Binary)
        m.u = Var(F, domain=Binary)
        m.B_over = Var(Pset, domain=NonNegativeReals)
    
        # (1) assign or reject
        def _assign_or_reject(m, f):
            return sum(m.x[f, p] for p in Pset) + m.u[f] == 1
        m.AssignOrReject = Constraint(F, rule=_assign_or_reject)
    
        # (2) acceptability
        def _accept(m, f, p):
            acc = int(future[f].A.get(p, 0))
            elig = int(getattr(future[f], "eligible", {}).get(p, 0))
            return m.x[f, p] <= acc * elig
        m.Acceptability = Constraint(F, Pset, rule=_accept)
        

        def _overtime_cap(m, p):
            return m.B_over[p] <= self.MAX_OVERTIME
        m.OvertimeCap = Constraint(Pset, rule=_overtime_cap)

        # (3) capacity (unified)
        if cap_limit is not None:
            def _cap(m, p):
                return sum(m.x[f, p] for f in F) <= int(cap_limit[p])
            m.Capacity = Constraint(Pset, rule=_cap)
            
        #if self.c_o > 0:
            # (4) overtime (unified)
        def _overtime(m, p):
            future_work = sum(
                float(future[f].duration) * float(future[f].Y.get(p, 1.0)) * m.x[f, p]
                for f in F
            )
            return m.B_over[p] >= (float(base_work[p]) + future_work) - float(self.session_time)
        m.Overtime = Constraint(Pset, rule=_overtime)
        overtime_pen = self.c_o * sum(m.B_over[p] for p in Pset)
        #else:
        #    overtime_pen = 0
    
        # (5) release (unified)
        if release_limit is not None:
            def _release(m):
                return sum(1 - m.u[f] for f in F) <= int(release_limit)
            m.ReleaseCap = Constraint(rule=_release)
    
        # Objective (same for both modes)
        def _rej_cost(f):
            return self.c_miss_1 if int(future[f].priority) == 1 else self.c_miss_2
    
        pref = {f: int(future[f].preferred_phys) for f in F}
        def _nonpref_pen(f, p):
            return (self.c_nonpref_1 if int(future[f].priority) == 1 else self.c_nonpref_2) * (1 if p != pref[f] else 0)
    
        reject_pen  = sum(_rej_cost(f) * m.u[f] for f in F)

        nonpref_pen  = sum(_nonpref_pen(f, p) * m.x[f, p] for f in F for p in Pset)
    
        m.Obj = Objective(expr=reject_pen + overtime_pen + nonpref_pen, sense=minimize)
    
        self.model = m
        info = {
            "mode": mode,
            "name": name,
            "t": t_val,
            "future_len": len(future),
            "remaining_release": release_limit,
            "remaining_cap_per_phys": cap_limit,
            "baseline_work_per_phys": base_work,
            "nmap": nmap,
        }
        return info
    
    # ------------------------------------------------------------
    # Solve utilities
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Extraction & I/O
    # ------------------------------------------------------------

    def extract_solution(self, patients):
        """
        Build dicts for assignments, rejections, overtime, and totals (including
        non-preferred penalties and objective).
        """
        assert self.model is not None
        m = self.model
        Nset = range(len(patients))
        Pset = range(self.P)

        sol = {
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

        for n in Nset:
            chosen = [p for p in Pset if value(m.x[n, p]) > 0.5]
            pat = patients[n]

            if not chosen and value(m.u[n]) > 0.5:
                pen = float(self.c_miss_1 if pat.priority == 1 else self.c_miss_2)
                sol["rejected"].append({
                    "kind": "reject",
                    "pid": getattr(pat, "pid", f"pat_{n}"),
                    "n": n,
                    "p": -1,
                    "priority": int(pat.priority),
                    "score": pat.score,
                    "duration": float(pat.duration),
                    "preferred_phys": pat.preferred_phys,
                    "eligible": list(pat.eligible.values()),
                    "penalty": pen
                    
                })
                sol["totals"]["reject_pen"] += pen

            for p in chosen:
                Y_np = getattr(pat, "Y", {}).get(p)
                nonpref = int(p != pat.preferred_phys)
                nonpref_cost = (
                    self.c_nonpref_1 if pat.priority == 1 else self.c_nonpref_2
                ) * nonpref

                work_min = float(pat.duration) * Y_np

                sol["assignments"].append({
                    "kind": "assign",
                    "pid": getattr(pat, "pid", f"pat_{n}"),
                    "n": n,
                    "p": p,
                    "priority": int(pat.priority),
                    "score": pat.score,
                    "duration": float(pat.duration),
                    "preferred_phys": pat.preferred_phys,
                    "eligible": list(pat.eligible.values()),
                    "nonpref_flag": nonpref,
                    "nonpref_cost": float(nonpref_cost),
                    "Y_np": Y_np,
                    "work_minutes": work_min
                })
                sol["totals"]["nonpref_pen"] += float(nonpref_cost)

        # overtime
        over_total = 0.0
        for p in Pset:
            Bover = float(value(m.B_over[p]))
            sol["overtime"].append({"p": p, "B_over": Bover})
            over_total += Bover
        sol["totals"]["overtime_pen"] = float(self.c_o) * over_total

        # objective
        obj = float(value(m.Obj))
        sol["totals"]["objective"] = obj

        return sol, obj
    
    def get_current_action(self, i):
        if self.model.u[i].value > 0.5:
            action = self.P
        else:
            for p in range(self.P):
                if self.model.x[i,p].value > 0.5:
                    action = p
                    break
        return action

    def save_solution_files(
        self,
        tag: str,
        sol: Dict,
        obj: float,
        status: str,
        term: str,
        gap: float,
        solve_time: float,
        folder: str = "instances_solution",
    ):
        """
        Save:
          - CSV '{tag}_schedule.csv' (assign/reject/overtime rows, chronological, with cumulative accepted counts)
          - JSON '{tag}_solution.json' (meta, totals, raw solution)
        """
        out_dir = self.out_dir if not folder else os.path.join(self.out_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
    
        # ---------- CSV (chronological by patient n) ----------
        csv_path = os.path.join(out_dir, f"{tag}_schedule.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            # added cumulative columns
            w.writerow([
                "kind", "pid", "n", "p", "priority", "score", "preferred_phys", "eligible",
                "duration", "Y_np", "work_minutes", "penalty", "B_over", "row_cost",
                "cum_accept_total", "cum_accept_p1", "cum_accept_p2",
                "cum_reject_total", "cum_reject_p1", "cum_reject_p2"
            ])
    
            # build chronological stream of patient events
            events = []
            for rec in sol.get("assignments", []):
                events.append({
                    "kind": "assign",
                    "pid": rec.get("pid", ""),
                    "n": int(rec["n"]),
                    "p": int(rec["p"]),
                    "priority": int(rec["priority"]),
                    "score": rec["score"],
                    "preferred_phys": int(rec.get("preferred_phys", -1)),
                    "eligible": rec.get("eligible", ""),
                    "duration": float(rec.get("duration", 0.0)),
                    "Y_np": float(rec.get("Y_np", 0.0)),
                    "work_minutes": float(rec.get("work_minutes", 0.0)),
                    "penalty": 0.0,
                    "B_over": 0.0,
                    "row_cost": 0.0,
                })
            for rec in sol.get("rejected", []):
                pen = float(rec.get("penalty", 0.0))
                events.append({
                    "kind": "reject",
                    "pid": rec.get("pid", ""),
                    "n": int(rec["n"]),
                    "p": -1,
                    "priority": int(rec["priority"]),
                    "score": rec["score"],
                    "preferred_phys": int(rec.get("preferred_phys", -1)),
                    "eligible": rec.get("eligible", ""),
                    "duration": float(rec.get("duration", 0.0)),
                    "Y_np": 0.0,
                    "work_minutes": 0.0,
                    "penalty": pen,
                    "B_over": 0.0,
                    "row_cost": pen,
                })
    
            kind_order = {"assign": 0, "reject": 1}
            events.sort(key=lambda r: (r["n"], kind_order.get(r["kind"], 99)))
    
            # running cumulative counters (accepted only)
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
                elif  ev["kind"] == "reject":
                    cum_reject += 1
                    if ev["priority"] == 1:
                        cum_reject_p1 += 1
                    elif ev["priority"] == 2:
                        cum_reject_p2 += 1
    
                w.writerow([
                    ev["kind"], ev["pid"], ev["n"], ev["p"], ev["priority"],ev["score"], ev["preferred_phys"],
                    ev["eligible"], ev["duration"], ev["Y_np"], ev["work_minutes"], ev["penalty"],
                    ev["B_over"], ev["row_cost"],
                    cum_total, cum_p1, cum_p2, cum_reject, cum_reject_p1, cum_reject_p2
                ])
    
            # overtime rows (do not affect cumulative accepted)
            for o in sol.get("overtime", []):
                p = int(o["p"]); b = float(o.get("B_over", 0.0))
                w.writerow([
                    "overtime", "", -1, p, -1, -1, -1,[],
                    0.0, 0.0, 0.0, 0.0,
                    b, float(self.c_o) * b,
                    cum_total, cum_p1, cum_p2, cum_reject, cum_reject_p1, cum_reject_p2
                ])
    
        # ---------- JSON meta / totals (unchanged except nonpref included) ----------
        reject_pen_total   = float(sol["totals"].get("reject_pen", 0.0))
        overtime_pen_total = float(sol["totals"].get("overtime_pen", 0.0))
        nonpref_pen_total  = float(sol["totals"].get("nonpref_pen", 0.0))
        objective_total    = float(sol["totals"].get("objective", obj))
    
        meta = {
            "objective": obj,
            "status": status,
            "termination": term,
            "gap": gap,
            "N_total": getattr(self, "N_total", getattr(self, "N", None)),
            "solve_time": solve_time,
            "breakdown": {
                "reject_pen_total": reject_pen_total,
                "overtime_pen_total": overtime_pen_total,
                "nonpref_pen_total": nonpref_pen_total,
                "objective_total": objective_total,
                "objective_check": reject_pen_total + overtime_pen_total + nonpref_pen_total
            },
            "params": {
                "N": self.N,
                "N_total": getattr(self, "N_total", getattr(self, "N", None)),
                "P": self.P,
                "session_time": self.session_time,
                "Delta": self.Delta,
                "I": self.I,
                "c_miss_1": self.c_miss_1,
                "c_miss_2": self.c_miss_2,
                "c_o": self.c_o,
                "Rmax": self.Rmax,
                "cap_per_phys": self.cap_per_phys,
            },
        }
    
        json_path = os.path.join(out_dir, f"{tag}_solution.json")
        with open(json_path, "w") as f:
            json.dump({"meta": meta, "solution": sol}, f, indent=2)
    

    def save_instance_csv(
        self,
        tag: str,
        patients: List["Patient"],
        folder: str = "instances",
        sort_by_id: bool = True,
    ) -> Dict[str, str]:
        """
        Persist a patient cohort (any list) to CSV:
          - '{tag}_patients.csv': pid, n(_id), priority, duration, preferred_phys, preferred_slot, arrival_score
          - '{tag}_AY.csv'     : (n, p, A_np, Y_np) per physician
    
        Args
        ----
        tag : str
            File stem for the two CSVs (e.g., 'exp1', 'regen_42').
        patients : List[Patient]
            Cohort to persist (e.g., self.patients or self.regenerated_patients).
        folder : str
            Subfolder under self.out_dir. Use "" to write directly to self.out_dir.
        sort_by_id : bool
            If True, rows are ordered by patient._id.
    
        Returns
        -------
        dict with paths: {"patients_csv": <path>, "ay_csv": <path>}
        """
        # Resolve target dir
        out_dir = self.out_dir if not folder else os.path.join(self.out_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
    
        # Order cohort (arrival order by _id)
        cohort = list(patients)
        if sort_by_id:
            cohort.sort(key=lambda q: int(getattr(q, "_id", 0)))
    
        # ---------- Patients ----------
        patients_path = os.path.join(out_dir, f"{tag}_patients.csv")
        with open(patients_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "pid", "n", "priority", "duration",
                "preferred_phys", "preferred_slot", "arrival_score"
            ])
            for p in cohort:
                w.writerow([
                    getattr(p, "pid", f"pat_{getattr(p, '_id', 0)}"),
                    int(getattr(p, "_id", 0)),
                    int(getattr(p, "priority", 2)),
                    float(getattr(p, "duration", 0.0)),
                    int(getattr(p, "preferred_phys", 0)),
                    int(getattr(p, "preferred_slot", -1)),
                    float(getattr(p, "score", getattr(p, "arrival_score", 0.0))),
                ])
    
        # ---------- A and Y per (n,p) ----------
        ay_path = os.path.join(out_dir, f"{tag}_AYE.csv")
        with open(ay_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "p", "A_np", "Y_np", "E_np"])
    
            # Detect if any patient in cohort uses legacy slot-level A[(p,i)]
            def has_slot_A(pat) -> bool:
                A = getattr(pat, "A", None)
                return isinstance(A, dict) and any(isinstance(k, tuple) and len(k) == 2 for k in A.keys())
    
            any_slot = any(has_slot_A(p) for p in cohort)
            for p in cohort:
                n = int(getattr(p, "_id", 0))
                pref_p = int(getattr(p, "preferred_phys", 0))
                A_dict = getattr(p, "A", {}) or {}
                Y_dict = getattr(p, "Y", {}) or {}
                E_dict = getattr(p, "eligible", {}) or {}
    
                for phys in range(self.P):
                    # A_np: collapse slot-level if needed, else read physician-level
                    if any_slot and has_slot_A(p):
                        a_np = 0
                        for i in range(self.I):
                            a_np = max(a_np, int(A_dict.get((phys, i), 0)))
                    else:
                        a_np = int(A_dict.get(phys, 0))
    
                    # Y_np: prefer stored Y; else fallback to rule
                    if phys in Y_dict:
                        y_np = float(Y_dict[phys])
                    else:
                        prio = int(getattr(p, "priority", 2))
                        if prio == 2:
                            y_np = 1.0
                        else:
                            y_np = 1.0 if (phys == pref_p) else float(self.show_prob)
                    e_np = int(E_dict.get(phys, 0))   
    
                    w.writerow([n, phys, int(a_np), int(y_np), int(e_np)])
    
    # ------------------------------------------------------------
    # End-to-end run
    # ------------------------------------------------------------

     # ============================
    # Simulation (step-based API)
    # ============================

    def init_sim(self,
                 enforce_cap_per_phys: bool = True,
                 respect_release_cap: bool = True,
                 reject_on_infeasible: bool = True,
                 reward_scale: float = 1.0):
        """
        Configure the built-in simulator.
        - enforce_cap_per_phys: respect self.cap_per_phys if not None
        - respect_release_cap  : stop once accepted == self.Rmax (if not None)
        - reject_on_infeasible : infeasible assign -> reject (apply rejection cost)
        - reward_scale         : multiply rewards by this (use -1.0 for positive utility)
        """
        self._sim_cfg = dict(
            enforce_cap_per_phys=enforce_cap_per_phys,
            respect_release_cap=respect_release_cap,
            reject_on_infeasible=reject_on_infeasible,
            reward_scale=float(reward_scale),
        )
        return self._sim_cfg
    
    # --- add these two small helpers inside your class ---
    
    def _sim_no_capacity_left(self) -> bool:
        """Return True iff per-phys caps are enforced and every physician is at capacity."""
        if not self._sim_cfg.get("enforce_cap_per_phys", False):
            return False
        if self.cap_per_phys is None:
            return False
        # all physicians have assigned >= cap
        return all(int(c) >= int(self.cap_per_phys) for c in self._sim_assigned_per_phys)
    
    def _sim_terminal_forced_reject_reward(self) -> float:
        """
        Sum the rejection penalties for all remaining patients (from current index self._sim_t to N-1).
        Negative value (since we use penalties as negative rewards).
        """
        pen = 0.0
        for idx in range(self._sim_t, self.N_total):
            prio = int(getattr(self.patients[idx], "priority", 2))
            pen += (self.c_miss_1 if prio == 1 else self.c_miss_2)
        return -pen


    def reset_simulator(self):
        """
        Reset the simulator episode. Requires self.patients to be populated
        (call generate_instance() first). Returns (state, done).
        """
        if not getattr(self, "patients", None):
            raise ValueError("Call generate_instance() before reset_sim().")

        self._sim_t = 0  # current patient index
        self._sim_patient_arrived=0
        self._sim_done = False
        self._sim_assigned_per_phys = [0] * self.P
        self._sim_work_per_phys = [0.0] * self.P  # sum of duration * Y[n,p]
        self._sim_accepted_count = 0
        self._sim_accepted_priority_phy = {i :{1:0, 2:0} for i in range(self.P)}

        # default config if not set
        if not hasattr(self, "_sim_cfg"):
            self.init_sim()

        # early-terminate if release cap is 0
        if self._sim_cfg["respect_release_cap"] and (self.Rmax is not None) and (self.Rmax <= 0):
            self._sim_done = True

        return self._sim_state(), self._sim_done
    
    def step(self, action: int):
        """
        One step in the simulator.
    
        Action space:
            0..P-1  => assign current patient to physician p
            P       => reject current patient
    
        Returns:
            (next_state, reward, done)
    
        Notes:
            - Rewards are NEGATIVE penalties (higher is better).
            - Overtime penalty is applied ONLY once, at the terminal step.
        """
        cfg = self._sim_cfg  # simulation configuration flags (set by init_sim)
    
        # If the episode is already finished, return a no-op transition.
        if self._sim_done:
            return self._sim_state(), 0.0 * cfg["reward_scale"], True
    
        # --- Early termination: global release cap exhausted BEFORE consuming the next patient ---
        # If a release cap is enforced and we've already accepted Rmax patients, end now.
        if cfg["respect_release_cap"] and (self.Rmax is not None) and (self._sim_accepted_count >= self.Rmax):
            self._sim_done = True
            term_reward = self._sim_terminal_overtime_reward() + self._sim_terminal_forced_reject_reward()  # add terminal overtime once
            return self._sim_state(), term_reward * cfg["reward_scale"], True
        
        if self._sim_no_capacity_left():
          self._sim_done = True
          term_reward = self._sim_terminal_overtime_reward() + self._sim_terminal_forced_reject_reward()
          return self._sim_state(), term_reward * cfg["reward_scale"], True

    
        # --- Early termination: no more patients to process ---
        if self._sim_t >= self.N_total:
            self._sim_done = True
            term_reward = self._sim_terminal_overtime_reward() 
            return self._sim_state(), term_reward * cfg["reward_scale"], True
    
        # Fetch the current patient (indexed by arrival order _sim_t)
        pat = self.patients[self._sim_t]
    
        # Interpret the action:
        #  - if action == P      -> reject
        #  - else (0..P-1)       -> assign to physician p_sel
        is_reject = (action == self.P)
        p_sel = None if is_reject else int(action)
    
        # --- Check feasibility WHEN attempting an assignment ---
        feasible = True
        if not is_reject:
            # Eligibility must be 1
            if int(getattr(pat, "eligible", {}).get(p_sel, 0)) == 0:
                feasible = False
        
            # Acceptability must be 1
            if feasible and pat.A.get(p_sel, 0) == 0:
                feasible = False
        
            # Capacity limit (existing)
            if feasible and cfg["enforce_cap_per_phys"] and (self.cap_per_phys is not None):
                if self._sim_assigned_per_phys[p_sel] >= self.cap_per_phys:
                    feasible = False
                    
            if feasible and (getattr(self, "session_time", None) is not None):
                # minutes that would be added by assigning this patient to p_sel
                y_np = float(pat.Y.get(p_sel, 1.0))
                add_work = float(pat.duration) * y_np
    
                # current accumulated future work for this physician
                cur_work = float(self._sim_work_per_phys[p_sel])
    
                # projected overtime after this assignment
                sess = float(self.session_time)
                projected_overtime = max(0.0, (cur_work + add_work) - sess)
    
                if projected_overtime > self.MAX_OVERTIME:
                    feasible = False
    
        
        # Initialize per-step reward (we accumulate negative penalties)
        reward = 0.0
    
        # --- Apply the action outcome ---
        if is_reject or (not feasible and cfg["reject_on_infeasible"]):
            # Case 1: Explicit reject OR infeasible assignment auto-converted to reject
            rej_cost = self.c_miss_1 if pat.priority == 1 else self.c_miss_2
            reward -= rej_cost  # penalties are negative rewards
            self._sim_t += 1    # move to the next patient
            
    
        elif not feasible and not cfg["reject_on_infeasible"]:
            # Case 2: Infeasible assignment but we choose NOT to convert to reject.
            # This is a "no-op" step by default; you could add a small shaping penalty if desired.
            reward += 0.0
    
        else:
            # Case 3: Valid assignment to physician p_sel
            # Penalize non-preferred assignment (priority-dependent)
            nonpref = int(p_sel != pat.preferred_phys)
            nonpref_cost = (self.c_nonpref_1 if pat.priority == 1 else self.c_nonpref_2) * nonpref
            reward -= nonpref_cost
    
            # Add realized workload to that physician:
            #  - duration (minutes) times attendance indicator Y[p_sel]
            y_np = float(pat.Y.get(p_sel, 1.0))
            self._sim_work_per_phys[p_sel] += float(pat.duration) * y_np
    
            # Update counters: physician assignment count and global accepted count
            self._sim_assigned_per_phys[p_sel] += 1
            self._sim_accepted_count += 1
            self._sim_accepted_priority_phy[p_sel][pat.priority] += 1
    
            # Advance to the next patient
            self._sim_t += 1
        
        # --- Check terminal conditions AFTER consuming this step ---
        # End the episode if:
        #   - we processed the last patient, OR
        #   - the release cap (if enforced) has just been reached.
        done_now = False
        if (self._sim_t >= self.N_total):
            done_now = True
        if cfg["respect_release_cap"] and (self.Rmax is not None) and (self._sim_accepted_count >= self.Rmax):
            done_now = True
        if self._sim_no_capacity_left():
            done_now = True
    
        if done_now:
            self._sim_done = True
            # Overtime + forced rejections for any *remaining* patients (if any)
            reward += self._sim_terminal_overtime_reward()
            reward += self._sim_terminal_forced_reject_reward()
    
        return self._sim_state(), reward * cfg["reward_scale"], feasible, self._sim_done
    
    def simulate_policy(self, policy_fn):
        """
        Convenience runner: simulate a full episode using policy_fn(state)->action.
        Returns a summary dict.
        """
        states, actions, feasibles, rewards = [], [], [], []
        state, done = self.reset_simulator()
        while not done:
            action = policy_fn(state)
            next_state, reward, feasible, done = self.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            feasibles.append(feasible)
            state = next_state

        # Add final state
        states.append(state)
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "feasibles": feasibles,
            "total_reward": float(sum(rewards)),
            "overtime_minutes_per_phys": self._sim_overtime_vector(),
            "overtime_penalty": -self._sim_terminal_overtime_reward()  # positive cost
        }

    # -----------------------
    # Internal helpers
    # -----------------------
    def _sim_overtime_vector(self):
        """Per-physician overtime minutes based on accumulated realized work."""
        return [max(0.0, w - self.session_time) for w in self._sim_work_per_phys]

    def _sim_terminal_overtime_reward(self):
        """Negative overtime penalty, realized only at terminal step."""
        over = self._sim_overtime_vector()
        return -self.c_o * float(sum(over))


    def _sim_state(self):
        """
        Build current state dict:
          - t, N, P, Rmax
          - current_patient (or None if done):
              priority, preferred_phys, available_phys (capacity-based availability)
          - work_per_phys (minutes), assigned_per_phys (counts)
          - cap_remaining (if enforcing caps), accepted_count, remaining_release (if applicable)
          - done, action_space_n
        """
        # --- capacity remaining per physician (if enforced) ---
        cap_remaining = None
        if self._sim_cfg.get("enforce_cap_per_phys", False) and (self.cap_per_phys is not None):
            cap_remaining = [max(0, int(self.cap_per_phys) - int(c)) for c in self._sim_assigned_per_phys]
    
        # --- global remaining release (if enforced) ---
        remaining_release = None
        if self._sim_cfg.get("respect_release_cap", False) and (self.Rmax is not None):
            remaining_release = max(0, int(self.Rmax) - int(self._sim_accepted_count))
    
        # --- current patient payload ---
        if self._sim_t >= self.N_total:
            cur = None
        else:
            pat = self.patients[self._sim_t]
            session = float(self.session_time) if self.session_time is not None else None

    
            # Capacity-based availability vector:
            #  - if caps enforced: 1 iff that physician still has remaining capacity
            #  - if caps not enforced: all physicians are available (all ones)
            if cap_remaining is not None:
                cap_ok = [cap_remaining[p] > 0 for p in range(self.P)]
            else:
                cap_ok = [True] * self.P  # all ok if no per-phys cap

            # Overtime-feasibility for assigning THIS patient to each physician
            # proj_OT_p = max(0, base_work_p + cur_work_p + pat.duration * Y[p] - session_time)
            ot_ok = [True] * self.P
            if  (session is not None) and session > 0:
                # baseline work if you track it; else default to 0
                base_work = getattr(self, "_sim_base_work_per_phys", [0.0] * self.P)
                cur_work = self._sim_work_per_phys
                cur_dur = float(getattr(pat, "duration", 0.0))
                Y = getattr(pat, "Y", {}) or {}
                for p in range(self.P):
                    y_np = float(Y.get(p, 1.0))
                    projected_overtime = max(
                        0.0,
                        float(base_work[p]) + float(cur_work[p]) + cur_dur * y_np - session
                    )
                    if projected_overtime > self.MAX_OVERTIME:
                        ot_ok[p] = False

            available_phys = [1 if (cap_ok[p] and ot_ok[p]) else 0 for p in range(self.P)]
            eligible_phys = [int(getattr(pat, "eligible", {}).get(p, 0)) for p in range(self.P)]
            cur = {
                "n": self._sim_t,
                "pid": getattr(pat, "pid", f"pat_{self._sim_t}"),
                "priority": int(pat.priority),
                "score": pat.score,
                "duration": pat.duration,
                "preferred_phys": int(pat.preferred_phys),
                "available_phys": available_phys,   # capacity availability
                "eligible_phys": eligible_phys,     # NEW eligibility mask
            }

    
        return {
            "t": self._sim_t,
            "N": self.N,
            "N_total": self.N_total,
            "P": self.P,
            "Rmax": self.Rmax,
            "session_time": self.session_time,
            "patients_arrived": self._sim_t + 1,
            "current_patient": cur,
            "work_per_phys": list(self._sim_work_per_phys),
            "assigned_per_phys": list(self._sim_assigned_per_phys),
            "cap_remaining": cap_remaining,          # None if caps not enforced
            "accepted_count": self._sim_accepted_count,
            "accepted_priority_phy": self._sim_accepted_priority_phy,
            "remaining_release": remaining_release,  # None if release cap not enforced
            "done": self._sim_done,
        }
    
    def collect_stats(
        self,
        states: List[Dict],
        actions: List[int],
        rewards: List[float],
        feasibles: List[bool]
    ) :
        """
        Return a *flat* dict of end-of-episode stats: {key: value}.
        Vectors are expanded into indexed keys, e.g., accepted_by_phys_0, ..., P-1.
        """
        if not states or len(states) != (len(actions) + 1) or len(actions) != len(rewards):
            raise ValueError("Inconsistent inputs: expect len(states)=len(actions)+1=len(rewards)+1.")
    
        P = self.P
        #N = states[0].get("N", getattr(self, "N", len(actions)))
        session_time = float(self.session_time)
    
        # ---- reconstruct assigns/rejects across steps ----
        accepted_by_phys = [0] * P
        accepted_by_phys_p1 = [0] * P
        accepted_by_phys_p2 = [0] * P
        nonpref_by_phys  = [0] * P
        nonpref_by_priority = [0] * 2
        rejected_p1 = 0
        rejected_p2 = 0
        total_arrivals = {1:0, 2:0}
    
        for k in range(len(actions)):
            s_before = states[k]
            s_after  = states[k + 1]
    
            assigned_before = s_before.get("assigned_per_phys") or [0]*P
            assigned_after  = s_after.get("assigned_per_phys") or [0]*P
            deltas = [assigned_after[i] - assigned_before[i] for i in range(P)]
    
            if sum(1 for d in deltas if d == 1) == 1 and all(d in (0,1) for d in deltas):
                p_exec = deltas.index(1)
                accepted_by_phys[p_exec] += 1
                cur = s_before.get("current_patient") or {}
                prio = int(cur.get("priority", -1))
                pref = int(cur.get("preferred_phys", -1))
                if 0 <= pref < P and p_exec != pref:
                    nonpref_by_phys[p_exec] += 1
                    nonpref_by_priority[prio-1] += 1
                if prio == 1:
                    accepted_by_phys_p1[p_exec] += 1
                elif prio == 2:
                    accepted_by_phys_p2[p_exec] += 1
                
            else:
                cur = s_before.get("current_patient") or {}
                prio = int(cur.get("priority", -1))
                if prio == 1:
                    rejected_p1 += 1
                elif prio == 2:
                    rejected_p2 += 1
                    
            total_arrivals[prio] +=1

        accepted_total = sum(accepted_by_phys)
        rejected_total = rejected_p1 + rejected_p2
        
        final_state = states[-1]
        t_final = int(final_state.get("t", len(actions)))  # index of NEXT patient to process
        N_total = int(final_state.get("N_total"))
    
        # Detect early termination condition:
        #   - Remaining release == 0 (when Rmax used up), OR
        #   - All per-phys capacities are zero (cap_remaining sums to 0).
        forced_rejects_p1 = 0
        forced_rejects_p2 = 0
    
        remaining_release = final_state.get("remaining_release", None)
        cap_remaining_vec = final_state.get("cap_remaining", None)
    
        release_exhausted = (remaining_release is not None and int(remaining_release) == 0 and getattr(self, "Rmax", None) is not None)
        cap_exhausted = (isinstance(cap_remaining_vec, list) and sum(int(max(0, c)) for c in cap_remaining_vec) == 0)
    
        if t_final < N_total and (release_exhausted or cap_exhausted):
            # All *unseen* patients must be rejected due to hard constraints
            # We can count their priorities from self.patients (authoritative list)
            for idx in range(t_final, N_total):
                #try:
                prio = int(getattr(self.patients[idx], "priority", 2))
                total_arrivals[prio] +=1
                #except:
                #    print(idx, self.N, self.N_total)
                if prio == 1:
                    forced_rejects_p1 += 1
                else:
                    forced_rejects_p2 += 1
    
            # Add to explicit rejections
            rejected_p1 += forced_rejects_p1
            rejected_p2 += forced_rejects_p2
            rejected_total = rejected_p1 + rejected_p2
    
        # ---- realized work & overtime from final state ----
        final_state = states[-1]
        work_per_phys = [float(x) for x in final_state.get("work_per_phys", [0.0]*P)]
        overtime_per_phys = [max(0.0, w - session_time) for w in work_per_phys]
        utilization_per_phys = [min(1.0, (w / session_time) if session_time > 0 else 0.0) for w in work_per_phys]
        overtime_total = sum(overtime_per_phys)
    
        # ---- penalties mirroring objective ----
        reject_penalty = rejected_p1 * float(self.c_miss_1) + rejected_p2 * float(self.c_miss_2)
    
        nonpref_penalty = 0.0
        for k in range(len(actions)):
            s_before = states[k]; s_after = states[k+1]
            deltas = [
                (s_after.get("assigned_per_phys", [0]*P)[i] -
                 s_before.get("assigned_per_phys", [0]*P)[i]) for i in range(P)
            ]
            if sum(1 for d in deltas if d == 1) == 1 and all(d in (0,1) for d in deltas):
                p_exec = deltas.index(1)
                cur = s_before.get("current_patient") or {}
                prio = int(cur.get("priority", 2))
                pref = int(cur.get("preferred_phys", -1))
                if 0 <= pref < P and p_exec != pref:
                    nonpref_penalty += float(self.c_nonpref_1 if prio == 1 else self.c_nonpref_2)
    
        overtime_penalty = float(self.c_o) * overtime_total
        total_cost = reject_penalty + nonpref_penalty + overtime_penalty
        total_reward = float(sum(rewards))
        total_feasibles = sum(feasibles)
    
        # ---- release usage ----
        flat: Dict[str, float | int] = {}
        Rmax = getattr(self, "Rmax", None)
        if Rmax is not None:
            flat["release_Rmax"] = int(Rmax)
            flat["release_fraction_used"] = (float(accepted_total) / max(1, int(Rmax)))
    
        # ---- scalar fields ----
        flat.update({
            "episode_length": int(len(actions)),
            "total_reward": total_reward,
            "total_feasibles": total_feasibles,
            "total_arrivals": self.N_total,
            "total_arrivals_p1":total_arrivals[1],
            "total_arrivals_p2":total_arrivals[2],
            "accepted_total": int(accepted_total),
            "rejected_total": int(rejected_total),
            "rejected_p1": int(rejected_p1),
            "rejected_p2": int(rejected_p2),
            "forced_rejects_p1": int(forced_rejects_p1),
            "forced_rejects_p2": int(forced_rejects_p2),
            "forced_rejects_total": int(forced_rejects_p1 + forced_rejects_p2),
            "nonpref_assignments_total": int(sum(nonpref_by_phys)),
            "nonpref_assignments_p1": int(nonpref_by_priority[0]),
            "nonpref_assignments_p2": int(nonpref_by_priority[1]),
            "overtime_minutes_total": float(overtime_total),
            "reject_penalty": float(reject_penalty),
            "nonpref_penalty": float(nonpref_penalty),
            "overtime_penalty": float(overtime_penalty),
            "total_cost": float(total_cost),
        })
    
        # ---- expand vectors into indexed keys ----
        for i in range(P):
            flat[f"accepted_by_phys_{i}"]            = int(accepted_by_phys[i])
            flat[f"accepted_by_phys_{i}_p1"]         = int(accepted_by_phys_p1[i])
            flat[f"accepted_by_phys_{i}_p2"]         = int(accepted_by_phys_p2[i])
            flat[f"nonpref_assignments_by_phys_{i}"] = int(nonpref_by_phys[i])
            flat[f"realized_work_per_phys_{i}"]      = float(work_per_phys[i])
            flat[f"overtime_minutes_per_phys_{i}"]   = float(overtime_per_phys[i])
            flat[f"utilization_per_phys_{i}"]        = float(utilization_per_phys[i])
        return flat
            
    # ============================
    # Regenerate future patients
    # ============================
    
    def _synthesize_patient_from_state_cur(self, cur_state: dict) -> "Patient":
        """
        Build a full Patient from a minimal current-patient dict in state:
        expected keys present in `cur_state`: pid, n (or _id), preferred_phys, preferred_slot, priority.
        We generate: duration, A[p], Y[p], score using the scheduler's RNGs and rules.
        """
        # Required fields (with safe defaults)
        pid = str(cur_state.get("pid", f"pat_{cur_state.get('n', 0)}"))
        n   = int(cur_state.get("n", cur_state.get("_id", 0)))
        preferred_phys  = int(cur_state.get("preferred_phys", 0))
        preferred_slot  = int(cur_state.get("preferred_slot", -1))
        priority        = int(cur_state.get("priority", 2))
    
        # --- Generate the missing bits using your existing distributions/rules ---
    
        # Duration from your truncated lognormal per priority
        duration = int(cur_state.get("duration"))
        score = 0
        E_map = cur_state.get("eligible_phys")
        E_map = {i: E_map[i] for i in range(len(E_map)) }
        # Physician-level acceptability A[p]: preferred = 1; others with accept_prob
        
        # Mask A by E
        A_map: dict[int, int] = {}
        for p in range(self.P):
            if E_map.get(p, 0) == 0:
                A_map[p] = 0
            else:
                A_map[p] = 1 if p == preferred_phys else int(self.rejection_generator.uniform() < self.accept_prob)
    
        # --- Attendance Y[p] ---
        Y_map: dict[int, float] = {}
        if priority == 2:
            for p in range(self.P):
                Y_map[p] = 1.0
        else:
            for p in range(self.P):
                if p == preferred_phys:
                    Y_map[p] = 1.0
                else:
                    Y_map[p] = float(int(self.show_generator.uniform() < self.show_prob))
    
        # Construct Patient (assumes your Patient dataclass/ctor matches these fields)
        return Patient(
            pid=pid,
            _id=n,
            duration=duration,
            preferred_phys=preferred_phys,
            preferred_slot=preferred_slot,
            priority=priority,
            A=A_map,
            Y=Y_map,
            score=score,
            eligible=E_map
        )
    
    def select_physician(self, state: dict, allow_reject_if_none: bool = True) -> int:
        cur = state.get("current_patient")
        if not cur:
            return self.P if allow_reject_if_none else -1
        P = self.P
        preferred = int(cur.get("preferred_phys", 0))
        avail = list(cur.get("available_phys") or [1] * P)
        elig  = list(cur.get("eligible_phys")  or [1] * P)
    
        # Preferred if both available and eligible
        if 0 <= preferred < P and (avail[preferred] == 1) and (elig[preferred] == 1):
            return preferred
    
        # Random among available & eligible
        candidates = [p for p in range(P) if (avail[p] == 1 and elig[p] == 1)]
        if candidates:
            return int(self.phys_generator.choice(candidates))
        return self.P if allow_reject_if_none else -1

    
    def _sample_total_N_given_arrivals(self,
        n_already: int,
        max_tries: int = 1000,
    ) -> int:
        """
        Sample an integer total N ~ Normal(mean, std^2) conditional on N >= n_already.
        Uses rejection sampling, rounds to nearest int, and enforces bounds.
    
        Parameters
        ----------
        n_already : int
            Number that have already arrived (lower truncation point).
        mean : float
            Mean of the (untruncated) normal prior for total N.
        std : float
            Standard deviation of the (untruncated) normal prior for total N (must be > 0).
        N_min : int, default 1
            Absolute lower bound for N before applying n_already constraint.
        N_max : int | None, default None
            Optional absolute upper bound for N. If None, no upper bound.
        seed : int | None, default None
            RNG seed for reproducibility.
        max_tries : int, default 1000
            Maximum rejection attempts before falling back to a clipped value.
    
        Returns
        -------
        int
            A sampled total N that satisfies N >= n_already and N in [N_min, N_max].
    
        Notes
        -----
        - This is an exact sample from the conditional distribution
          Normal(mean, std^2) given N >= n_already (and <= N_max if provided),
          up to rounding to the nearest integer.
        - If acceptance becomes extremely unlikely (deep tail), the function
          falls back to a clipped rounded mean to ensure progress.
        """
        
        if self.sigma == 0:
            return self.N
    
        if self.sigma < 0:
            raise ValueError("std must be > 0")
        
        for _ in range(max_tries):
            draw = self.total_p_generator.normal(loc=self.N, scale=self.sigma)
            N = int(round(draw))
            if N < n_already:
                continue
            return N
    
        # Fallback if the truncated region is extremely small
        N = int(round(self.N))
        N = max(n_already, N)
        return N


    def regenerate_scenario(self, state: Dict, patients: Optional[List["Patient"]] = None,  new: int = 1):
        """
        Regenerate a 'future scenario' cohort based on the current simulator state.

        Args:
            state: A dict like the one returned by self._sim_state() / reset_sim() / step().
                   It must contain keys: 't' and 'current_patient'.
            patients: Optional full list to reuse when new == 0. If None, falls back to self.patients.
            new: If 1 (default), generate *new* patients for the remaining horizon.
                 If 0, reuse the *same* patients already created (either from `patients` arg,
                 or from `self.patients`).

        Side-effects:
            - Sets self.regenerated_patients to a list containing:
              [ current_patient, future_patient_{t+1}, ... ]
            - Keeps order consistent with arrival order.
        """
        if "t" not in state:
            raise ValueError("State missing 't'. Pass a state returned by reset_sim()/step().")

        t = int(state["t"])

        # Current patient from the authoritative store (preserves A/Y maps, etc.)
        cur_state = state.get("current_patient", None)

        regenerated: List["Patient"] = []

        cur_pat = self._synthesize_patient_from_state_cur(cur_state)
        # Tag as regen view
        cur_pat.pid = f"regen_{cur_pat.pid}"
        regenerated: List["Patient"] = [cur_pat]

        # How many future patients remain after the current index?
        N_curr = state["patients_arrived"]
        self.N_total  = self._sample_total_N_given_arrivals(N_curr)
        remaining = max(0, self.N_total - (t + 1))


        if int(new) == 1:
            # -------- Generate brand-new future patients --------
            # We’ll generate `remaining` patients and assign sequential arrival IDs after `t`.
            for k in range(remaining):
                # Generate a fresh patient using your RNG streams
                p = self.generate_patient(n=t + 1 + k)
                # Tag PID so it's clear these are regenerated
                # Ensure arrival order IDs are sequential after t
                p._id = t + 1 + k
                regenerated.append(p)
                
            #jitter = self.jitter_generator.uniform(0, 1e-9, size=len(regenerated))
            ordered = sorted(regenerated, key=lambda p: (p.score))

            # Overwrite IDs to arrival order
            for rank, p in enumerate(ordered):
                p._id = rank
                p.pid = f"regenpat_{p.priority}_{rank}"  # optional: encode priority+order

            regenerated = ordered

        else:
            # -------- Reuse the same original future cohort --------
            base = patients if (patients is not None) else self.patients
            # Keep those strictly after current patient's arrival order
            future_same = sorted([p for p in base if int(p._id) > int(cur_pat._id)], key=lambda q: int(q._id))
            for p in future_same:
                # Shallow copy & tag; preserve A/Y/priority/duration etc.
                pcopy = type(p)(
                    pid=f"regen_{p.pid}",
                    _id=int(p._id),
                    duration=float(p.duration),
                    preferred_phys=int(p.preferred_phys),
                    preferred_slot=int(getattr(p, "preferred_slot", -1)),
                    priority=int(p.priority),
                    A=dict(getattr(p, "A", {})),
                    Y=dict(getattr(p, "Y", {})),
                    score=float(getattr(p, "score", 0.0)),
                    eligible=dict(getattr(p, "eligible", {}))
                )
                regenerated.append(pcopy)

        # Store for later use
        self.regenerated_patients = regenerated       
    
    # ============================
    # Simulation trace persistence
    # ============================



    def save_sim_trace_csv(
        self,
        tag: str,
        states: list,
        actions: list,
        rewards: list,
        feasibles: list,
        include_vectors: bool = True,
        folder: str = "greedy",
    ):
        """
        Save per-step simulation trace to CSV and, on the terminal step (early stop),
        add forced rejections for all *remaining* patients to the cumulative reject counts.
    
        Notes:
          - Availability uses capacity-driven 'available_phys' (not acceptability A).
          - Duration/Y are read from self.patients[n].
          - 'feasibles' is an external list you pass per step for debugging (0/1 or bool).
        """
        # Basic checks
        if not states or len(states) != (len(actions) + 1) or len(actions) != len(rewards):
            raise ValueError("Inconsistent inputs: expect len(states)=len(actions)+1=len(rewards)+1.")
    
        out_dir = self.out_dir if not folder else os.path.join(self.out_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{tag}_sim_trace.csv")
    
        # Headers
        base_cols = [
            "step",
            "t_before",
            "action_raw",
            "executed",             # "assign" | "reject"
            "executed_phys",        # -1 if reject
            "reward",
            "feasible",
            "cumulative_reward",
            "done_after_step",
    
            # Patient snapshot
            "n", "pid", "priority", "score", "duration",
            "preferred_phys", "eligible",
    
            # Availability hint for preferred
            "available_phys_pref",
        ]
    
        vector_cols = [
            "available_phys_vec",
            "Y_map_vec",
            "work_per_phys_vec",
            "assigned_per_phys_vec",
            "cap_remaining_vec",
        ]
    
        agg_cols = [
            "accepted_count_before",
            "accepted_count_after",
            "accepted_count_after_p1",
            "accepted_count_after_p2",
            "total_rejected",                # includes forced rejections if terminal
            "rejected_count_after_p1",       # includes forced rejections if terminal
            "rejected_count_after_p2",       # includes forced rejections if terminal
            "remaining_release_before",
        ]
    
        overtime_cols = [
            "overtime_minutes_per_phys_after",
        ]
    
        headers = base_cols + agg_cols
        if include_vectors:
            headers += vector_cols
        headers += overtime_cols
    
        # Running tallies (by priority) for accepted / rejected *so far*
        accepted_count_after = {1: 0, 2: 0}
        rejected_count_after = {1: 0, 2: 0}
    
        cum = 0.0
    
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
    
            for k in range(len(actions)):
                s_before = states[k]
                s_after  = states[k + 1]
                a = int(actions[k])
                r = float(rewards[k])
                feas = feasibles[k]
                cum += r
    
                # For remaining computation on terminal step
                t_after = int(s_after.get("t", s_before.get("t", 0) + 1))   # NEW
    
                # Current patient snapshot
                cur = s_before.get("current_patient") or {}
                pid = cur.get("pid", "")
                n   = int(cur.get("n", -1))
                prio = int(cur.get("priority", -1))
                score = cur.get("score", -1)
                pref = int(cur.get("preferred_phys", -1))
                # Your state sometimes has 'eligible_phys', you want to log under 'eligible'
                elig = cur.get("eligible_phys", cur.get("eligible", None))
    
                # Duration & Y from master list
                if 0 <= n < len(self.patients):
                    dur = float(getattr(self.patients[n], "duration", 0.0))
                    y_map_dict = getattr(self.patients[n], "Y", {})
                    y_map_vec = [float(y_map_dict.get(p, 1.0)) for p in range(self.P)]
                else:
                    dur = 0.0
                    y_map_vec = None
    
                # Availability vector at decision time
                avail_vec = cur.get("available_phys", None)
                if avail_vec is None:
                    avail_vec = [1] * self.P
                available_pref = None
                if isinstance(avail_vec, list) and 0 <= pref < len(avail_vec):
                    available_pref = int(avail_vec[pref])
    
                # Determine executed effect by counting assignment deltas
                assigned_before = s_before.get("assigned_per_phys") or [0]*self.P
                assigned_after  = s_after.get("assigned_per_phys") or [0]*self.P
                deltas = [assigned_after[i] - assigned_before[i] for i in range(self.P)]
    
                if sum(1 for d in deltas if d == 1) == 1 and all(d in (0, 1) for d in deltas):
                    executed = "assign"
                    executed_phys = deltas.index(1)
                    accepted_count_after[prio] += 1
                else:
                    executed = "reject"
                    executed_phys = -1
                    if prio in (1, 2):
                        rejected_count_after[prio] += 1
    
                # Overtime minutes after the step
                work_after = s_after.get("work_per_phys") or [0.0]*self.P
                overtime_after = [max(0.0, wmin - self.session_time) for wmin in work_after]
    
                # ---- If terminal now, add FORCED rejections for all remaining patients ----
                done_after = bool(s_after.get("done", False))
                forced_rej_p1 = 0
                forced_rej_p2 = 0
                if done_after:
                    # Count all future patients (indices >= t_after) as forced rejections
                    # because episode ended early (capacity or release cap)
                    for idx in range(t_after, len(self.patients)):
                        pprio = int(getattr(self.patients[idx], "priority", 2))
                        if pprio == 1:
                            forced_rej_p1 += 1
                        else:
                            forced_rej_p2 += 1
                    # Update cumulative rejected counts
                    rejected_count_after[1] += forced_rej_p1
                    rejected_count_after[2] += forced_rej_p2
                # -------------------------------------------------------------------------
    
                row = [
                    k,
                    int(s_before.get("t", -1)),
                    a,
                    executed,
                    executed_phys,
                    r,
                    feas,
                    cum,
                    done_after,
                    n, pid, prio, score, dur, pref, elig,
                    available_pref if available_pref is not None else "",
                    int(s_before.get("accepted_count", 0)),
                    int(s_after.get("accepted_count", 0)),
                    accepted_count_after[1], accepted_count_after[2],
                    (rejected_count_after[1] + rejected_count_after[2]),
                    rejected_count_after[1], rejected_count_after[2],
                    int(s_before.get("remaining_release", 0) if s_before.get("remaining_release") is not None else 0),
                ]
    
                if include_vectors:
                    row += [
                        json.dumps(avail_vec) if avail_vec is not None else "",
                        json.dumps(y_map_vec) if y_map_vec is not None else "",
                        json.dumps(s_before.get("work_per_phys", [])),
                        json.dumps(s_before.get("assigned_per_phys", [])),
                        json.dumps(s_before.get("cap_remaining", [])) if s_before.get("cap_remaining", None) is not None else "",
                    ]
    
                row += [json.dumps(overtime_after)]
                w.writerow(row)


    def run_and_save_sim(self, policy_fn, tag: str, folder:str = "greedy"):
        """
        Convenience wrapper:
          - (optionally) configure the simulator via init_sim(**init_kwargs)
          - run simulate_policy(policy_fn)
          - save the per-step trace to CSV
        Returns: (trajectory_dict, csv_path)
        """

        traj = self.simulate_policy(policy_fn)
        self.save_sim_trace_csv(tag, traj["states"], traj["actions"], traj["rewards"], traj["feasibles"])
        stats = self.collect_stats(traj["states"], traj["actions"], traj["rewards"], traj["feasibles"])
        out_dir = os.path.join(self.out_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"Sim_Stats_{tag}.json")
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        return traj


    # ---------------------------- End-to-end run ----------------------------

    def run(self, tag: str = "sampled", solver: str = "gurobi", tee: bool = False):
        self.generate_instance()
        self.save_instance_csv(tag=tag, patients=self.patients)  # save instance
        _ = self.build_model(self.patients)
        self.setup_solver(solver)
        status, term, gap, t = self.solve(tee=tee)
        sol, obj = self.extract_solution(self.patients)
        self.save_solution_files(tag, sol, obj, status, term, gap, t)
        print(f"Status={status} Termination={term} Obj={obj:.2f} Gap={gap} Time={t:.2f}s")
        return sol, obj
    
    
#-----------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    
    stop_flag = False
    thread = threading.Thread(target=track_memory)
    thread.start()
    
    args = parse_args()
    config = build_config(args)
    
    sched = AppointmentScheduler(
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
    )
    
    
    tag = f"OCS_{config.seed}"
    sched.run(tag=tag, solver='gurobi', tee=True)
    
    sched.init_sim(enforce_cap_per_phys=True, respect_release_cap=True, reward_scale=1.0)
    
    # run it
    tag = f"OCS_greedy_{config.seed}"
    summary = sched.run_and_save_sim(sched.select_physician, tag=tag)
    print("Total reward:", summary["total_reward"])
    print("Overtime per phys:", summary["overtime_minutes_per_phys"])
    print("Overtime penalty:", summary["overtime_penalty"])
    
        
    stop_flag = True
    thread.join()