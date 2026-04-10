#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:12:50 2026

@author: prakashgawas
"""

#!/usr/bin/env python3
"""
Quick experimental check: compare new=0 vs new=1 in terms of
  - MIP size (number of patients in regenerated_patients)
  - Solve time
  - MIP gap
across decision steps t within one simulated episode.

Run from the project root, e.g.:
    python check_new_flag.py

Adjust the CONFIG block below to match your actual parameters.
"""

import time
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # adjust if needed

# ── put your project on the path ──────────────────────────────────────────────
PROJECT_DIR = "."   # <-- change to your project directory if needed
sys.path.insert(0, PROJECT_DIR)

from OCS import AppointmentScheduler

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  –  match your actual experiment parameters
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    N1=25, sigma1=5,
    N2=75, sigma2=12,
    P=4, I=8,
    k_min=1, k_max=4,
    physician_weights=True,
    c_miss_1=200, c_miss_2=50,
    c_overtime=0, c_np_per=10,
    release_cap=80, cap_per_phys=20,
    time_limit=30, mipgap=0.02,
    seed=42,
)

# How many decision steps to sample per episode (None = all steps)
SAMPLE_EVERY = 5      # check every 5th step to keep runtime short
NUM_EPISODES = 3      # run this many episodes

# ══════════════════════════════════════════════════════════════════════════════

def make_sim(seed, out_dir="/tmp/ocs_check"):
    sim = AppointmentScheduler(
        n_patients1=CONFIG["N1"], sigma1=CONFIG["sigma1"],
        n_patients2=CONFIG["N2"], sigma2=CONFIG["sigma2"],
        n_phys=CONFIG["P"], n_slots=CONFIG["I"],
        k_min=CONFIG["k_min"], k_max=CONFIG["k_max"],
        physician_weights=CONFIG["physician_weights"],
        c_miss_1=CONFIG["c_miss_1"], c_miss_2=CONFIG["c_miss_2"],
        c_overtime=CONFIG["c_overtime"], c_np_per=CONFIG["c_np_per"],
        release_cap=CONFIG["release_cap"], cap_per_phys=CONFIG["cap_per_phys"],
        time_limit=CONFIG["time_limit"], mipgap=CONFIG["mipgap"],
        seed=seed, out_dir=out_dir, suffix=False,
    )
    sim.setup_solver()
    return sim


def run_episode_check(episode_idx, new_flag):
    """
    Simulate one episode using greedy (preferred physician) policy.
    At every SAMPLE_EVERY-th step, regenerate + solve and record stats.
    Returns a DataFrame of per-step measurements.
    """
    seed = CONFIG["seed"] + episode_idx
    sim = make_sim(seed)
    sim.set_seed(seed)
    sim.generate_instance()
    sim.init_sim(enforce_cap_per_phys=True, respect_release_cap=True)

    # A second sim object used purely for scenario regeneration / solving
    regen_sim = make_sim(seed + 1000)

    state, done = sim.reset_simulator()
    records = []
    step = 0

    while not done:
        if step % SAMPLE_EVERY == 0:
            # ── regenerate ────────────────────────────────────────────────
            t0_regen = time.perf_counter()
            if new_flag == 1:
                regen_sim.regenerate_scenario(state)
            else:
                regen_sim.regenerate_scenario(state, sim.patients, new=0)
            regen_time = time.perf_counter() - t0_regen

            mip_size = len(regen_sim.regenerated_patients)

            # ── build + solve ─────────────────────────────────────────────
            regen_sim.build_model(regen_sim.regenerated_patients, state=state)
            t0_solve = time.perf_counter()
            status, term, gap, sol_time = regen_sim.solve(tee=False)
            wall_solve = time.perf_counter() - t0_solve

            records.append({
                "episode":    episode_idx,
                "new":        new_flag,
                "step_t":     state["t"],
                "N_total_sim": sim.N_total,              # actual simulated N
                "mip_patients": mip_size,                # patients in MIP
                "regen_wall_s": round(regen_time, 4),
                "solve_wall_s": round(wall_solve, 4),
                "gap_pct":    round(gap, 3) if gap is not None else None,
                "status":     status,
            })

        # greedy action: prefer preferred physician
        action = sim.select_physician(state)
        state, reward, feasible, done = sim.step(action)
        step += 1

    return pd.DataFrame(records)


def main():
    all_records = []

    for ep in range(NUM_EPISODES):
        print(f"\n── Episode {ep} ──────────────────────────────────")
        for new_flag in [1, 0]:
            label = "new=1 (fresh)" if new_flag == 1 else "new=0 (reuse)"
            print(f"  Running {label} ...")
            df = run_episode_check(ep, new_flag)
            all_records.append(df)
            summary = df[["mip_patients", "regen_wall_s", "solve_wall_s", "gap_pct"]].describe().loc[
                ["mean", "min", "max"]
            ].round(4)
            print(f"  {label} summary:")
            print(summary.to_string())

    full = pd.concat(all_records, ignore_index=True)

    print("\n══════════════════════════════════════════════════════")
    print("AGGREGATE COMPARISON  (mean across all episodes & steps)")
    print("══════════════════════════════════════════════════════")
    agg = (
        full.groupby("new")[["mip_patients", "regen_wall_s", "solve_wall_s", "gap_pct"]]
        .agg(["mean", "std", "max"])
        .round(4)
    )
    agg.index = agg.index.map({1: "new=1 (fresh)", 0: "new=0 (reuse)"})
    print(agg.to_string())

    # ── per-step view: how does MIP size evolve with t? ───────────────────────
    print("\n── MIP size by step_t bucket (mean across episodes) ──")
    full["t_bucket"] = (full["step_t"] // 10) * 10
    pivot = full.pivot_table(
        index="t_bucket", columns="new", values="mip_patients", aggfunc="mean"
    ).rename(columns={1: "new=1", 0: "new=0"}).round(1)
    print(pivot.to_string())

    out_csv = os.path.join(PROJECT_DIR, "check_new_flag_results.csv")
    full.to_csv(out_csv, index=False)
    print(f"\nFull results saved to: {out_csv}")


if __name__ == "__main__":
    main()