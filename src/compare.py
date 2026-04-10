#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:32:16 2026

@author: prakashgawas
"""

"""
compare_regen.py
================
1. Compare _sample_N_future vs _sample_N_future_rejection  (N_future distributions)
2. Validate that regenerated horizons match generate_instance() statistics:
     - N1_total, N2_total
     - score distributions per priority
     - duration distributions per priority
     - preferred_phys distributions
     - eligibility counts
"""

import sys, os
sys.path.insert(0, "/mnt/project")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, norm, binom, ks_2samp
from scipy.stats import truncnorm

# ── Paste the two standalone N-future samplers so we can call them without
#    instantiating the full class (avoids Gurobi / pyomo overhead).
# ─────────────────────────────────────────────────────────────────────────────

def _sample_N_future_posterior(mu, sigma, n_arrived, s_mid, beta_a, beta_b, rng):
    """
    Method C  — exact Bayesian posterior  (_sample_N_future in new OCS.py)
    P(N_total | n_arrived) ∝ Binom(n_arrived; N_total, p) * Normal(N_total; mu, sigma)
    where p = P(Beta(a,b) < s_mid).
    Returns N_future = N_total - n_arrived.
    """
    p = beta_dist.cdf(s_mid, beta_a, beta_b)
    likelihood_peak = int(n_arrived / p) if p > 1e-9 else int(mu)
    N_max = int(max(mu + 6 * sigma, likelihood_peak + 6 * sigma))
    N_vals = np.arange(n_arrived, N_max + 1)
    prior      = norm.pdf(N_vals, mu, sigma)
    likelihood = binom.pmf(n_arrived, N_vals, p)
    weights    = prior * likelihood
    total      = weights.sum()
    if total < 1e-300:
        N_total = max(n_arrived, likelihood_peak)
    else:
        weights /= total
        N_total = int(rng.choice(N_vals, p=weights))
    return max(0, N_total - n_arrived)


def _sample_N_future_rejection_sampler(
    mu, sigma, n_arrived, s_mid, beta_a, beta_b, rng, max_tries=100_000
):
    """
    Method D  — rejection sampling  (_sample_N_future_rejection in new OCS.py)
    Draws N_total from Normal prior, simulates N_total scores, accepts if
    exactly n_arrived have score < s_mid.
    Falls back to the posterior if rejection fails.
    """
    for _ in range(max_tries):
        N_total = max(n_arrived, int(round(rng.normal(loc=mu, scale=sigma))))
        scores  = rng.beta(beta_a, beta_b, size=N_total)
        if int(np.sum(scores < s_mid)) == n_arrived:
            return int(np.sum(scores >= s_mid))

    # fallback — posterior
    p = beta_dist.cdf(s_mid, beta_a, beta_b)
    likelihood_peak = int(n_arrived / p) if p > 1e-9 else int(mu)
    N_max = int(max(mu + 6 * sigma, likelihood_peak + 6 * sigma))
    N_vals = np.arange(n_arrived, N_max + 1)
    prior      = norm.pdf(N_vals, mu, sigma)
    likelihood = binom.pmf(n_arrived, N_vals, p)
    weights    = prior * likelihood
    total      = weights.sum()
    try:
        N_total    = max(n_arrived, likelihood_peak) if total < 1e-300 else int(
            rng.choice(N_vals, p=weights)
            )
    except:
        print(sum(weights))
    return max(0, N_total - n_arrived)


# ─────────────────────────────────────────────────────────────────────────────
# Parameters (mirror the example state in OCS.py __main__)
# ─────────────────────────────────────────────────────────────────────────────
MU1, SIGMA1 = 25,  8
MU2, SIGMA2 = 75, 15

# Varying observation counts to test the samplers across different
# "how early in the horizon are we?" scenarios
SCENARIOS = [
    #{"label": "early  (t≈10%)",  "n1": 1,  "n2":  7,  "s_mid": 0.10},
    {"label": "mid    (t≈40%)",  "n1": 7,  "n2": 28,  "s_mid": 0.40},
    {"label": "late   (t≈80%)",  "n1": 15, "n2": 57,  "s_mid": 0.75},
]

N_SAMPLES = 5_000
SEED      = 42
rng       = np.random.default_rng(SEED)

BETA_P1 = (3.0, 1.0)
BETA_P2 = (1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — compare N_future distributions across arrival scenarios
# ─────────────────────────────────────────────────────────────────────────────
# print("=" * 70)
# print("PART 1 — _sample_N_future  vs  _sample_N_future_rejection")
# print("=" * 70)

# fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(13, 4 * len(SCENARIOS)))
# fig.suptitle(
#     "_sample_N_future (posterior)  vs  _sample_N_future_rejection\n"
#     f"mu1={MU1} σ1={SIGMA1}   mu2={MU2} σ2={SIGMA2}   n_samples={N_SAMPLES}",
#     fontsize=11, fontweight="bold",
# )

# for row, sc in enumerate(SCENARIOS):
#     n1, n2, s_mid = sc["n1"], sc["n2"], sc["s_mid"]
#     label = sc["label"]

#     post_p1, rej_p1 = [], []
#     post_p2, rej_p2 = [], []

#     for _ in range(N_SAMPLES):
#         post_p1.append(_sample_N_future_posterior(
#             MU1, SIGMA1, n1, s_mid, *BETA_P1, rng))
#         post_p2.append(_sample_N_future_posterior(
#             MU2, SIGMA2, n2, s_mid, *BETA_P2, rng))
#         rej_p1.append(_sample_N_future_rejection_sampler(
#             MU1, SIGMA1, n1, s_mid, *BETA_P1, rng))
#         rej_p2.append(_sample_N_future_rejection_sampler(
#             MU2, SIGMA2, n2, s_mid, *BETA_P2, rng))

#     post_p1, rej_p1 = np.array(post_p1), np.array(rej_p1)
#     post_p2, rej_p2 = np.array(post_p2), np.array(rej_p2)

#     ks1 = ks_2samp(post_p1, rej_p1)
#     ks2 = ks_2samp(post_p2, rej_p2)

#     print(f"\n{label}  |  s_mid={s_mid}  n1_arrived={n1}  n2_arrived={n2}")
#     print(f"  P1  posterior : mean={post_p1.mean():.1f}  std={post_p1.std():.1f}  "
#           f"min={post_p1.min()}  max={post_p1.max()}")
#     print(f"  P1  rejection : mean={rej_p1.mean():.1f}  std={rej_p1.std():.1f}  "
#           f"min={rej_p1.min()}  max={rej_p1.max()}")
#     print(f"  P1  KS stat={ks1.statistic:.4f}  p={ks1.pvalue:.4f}  "
#           f"→ {'SAME dist ✓' if ks1.pvalue > 0.05 else 'DIFFERENT ✗'}")
#     print(f"  P2  posterior : mean={post_p2.mean():.1f}  std={post_p2.std():.1f}  "
#           f"min={post_p2.min()}  max={post_p2.max()}")
#     print(f"  P2  rejection : mean={rej_p2.mean():.1f}  std={rej_p2.std():.1f}  "
#           f"min={rej_p2.min()}  max={rej_p2.max()}")
#     print(f"  P2  KS stat={ks2.statistic:.4f}  p={ks2.pvalue:.4f}  "
#           f"→ {'SAME dist ✓' if ks2.pvalue > 0.05 else 'DIFFERENT ✗'}")

#     # ── plots ──
#     all_bins = np.linspace(0, max(post_p1.max(), rej_p1.max()) + 5, 40)
#     ax = axes[row][0]
#     ax.hist(post_p1, bins=all_bins, alpha=0.6, label=f"posterior  μ={post_p1.mean():.1f}", color="steelblue")
#     ax.hist(rej_p1,  bins=all_bins, alpha=0.6, label=f"rejection  μ={rej_p1.mean():.1f}",  color="darkorange")
#     ax.set_title(f"P1 N_future  |  {label}  s_mid={s_mid}\nKS p={ks1.pvalue:.3f}", fontsize=9)
#     ax.set_xlabel("N_future P1"); ax.legend(fontsize=8)

#     all_bins2 = np.linspace(0, max(post_p2.max(), rej_p2.max()) + 5, 40)
#     ax = axes[row][1]
#     ax.hist(post_p2, bins=all_bins2, alpha=0.6, label=f"posterior  μ={post_p2.mean():.1f}", color="steelblue")
#     ax.hist(rej_p2,  bins=all_bins2, alpha=0.6, label=f"rejection  μ={rej_p2.mean():.1f}",  color="darkorange")
#     ax.set_title(f"P2 N_future  |  {label}  s_mid={s_mid}\nKS p={ks2.pvalue:.3f}", fontsize=9)
#     ax.set_xlabel("N_future P2"); ax.legend(fontsize=8)

# plt.tight_layout()
# plt.savefig("/mnt/user-data/outputs/part1_N_future_comparison.png", dpi=130, bbox_inches="tight")
# plt.close()
# print("\n→ Plot saved: part1_N_future_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — horizon statistics: generate_instance vs regenerated
#
#  We run K independent episodes. Each episode:
#    1. generate_instance()  → ground-truth full horizon stats
#    2. Simulate T steps (T = mid-point) using greedy policy
#    3. Call regenerate_scenario() with each of the two N-samplers
#    4. Collect stats from the regenerated horizon
#
#  We compare:
#    - Total N1, N2 counts
#    - Score distribution (per priority) 
#    - Duration distribution (per priority)
#    - Preferred phys distribution
#    - Eligible count per patient
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2 — horizon stats: generate_instance vs regenerated")
print("=" * 70)

try:
    from OCS import AppointmentScheduler
    OCS_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Could not import OCS: {e}")
    OCS_AVAILABLE = False

if OCS_AVAILABLE:
    K_EPISODES  = 1000   # number of independent instances
    # The state at which we regenerate (fraction of horizon elapsed)
    CUT_FRACS   = [0.15, 0.40, 0.75, 0.9]

    # Containers: {frac: {method: {metric: [values]}}}
    from collections import defaultdict

    def collect_patient_stats(patients, label=""):
        """Return a flat dict of summary stats for a patient list."""
        if not patients:
            return {}
        p1 = [p for p in patients if p.priority == 1]
        p2 = [p for p in patients if p.priority == 2]
        return {
            "N_total": len(patients),
            "N1": len(p1), "N2": len(p2),
            "score_p1":  [p.score for p in p1],
            "score_p2":  [p.score for p in p2],
            "dur_p1":    [p.duration for p in p1],
            "dur_p2":    [p.duration for p in p2],
            "pref_phys": [p.preferred_phys for p in patients],
            "elig_count":[sum(p.eligible.values()) for p in patients],
        }

    # ── build one scheduler (fixed params, varied seeds per episode) ──
    BASE_SEED = 7
    sched = AppointmentScheduler(
        n_patients1=MU1, sigma1=SIGMA1,
        n_patients2=MU2, sigma2=SIGMA2,
        n_phys=4, n_slots=8,
        k_max=3, k_min=1,
        c_miss_1=80, c_miss_2=50, c_overtime=4, c_np_per=10,
        physician_weights=True,
        release_cap=80, cap_per_phys=25,
        seed=BASE_SEED, time_limit=5,
        out_dir="/tmp/regen_test/", suffix=False,
    )
    sched.init_sim(enforce_cap_per_phys=True, respect_release_cap=True, reward_scale=1.0)

    # storage: ground truth and two regen methods, per cut fraction
    gt_stats   = defaultdict(list)   # gt_stats[frac] = list of stat dicts
    post_stats = defaultdict(list)   # posterior method
    rej_stats  = defaultdict(list)   # rejection method

    print(f"Running {K_EPISODES} episodes × {len(CUT_FRACS)} cut points …")

    for ep in range(K_EPISODES):
        sched.set_seed(BASE_SEED + ep * 31)
        sched.generate_instance()

        # Ground truth: stats of the FULL horizon
        full_stats = collect_patient_stats(sched.patients)

        # Simulate up to each cut fraction, collect state, then regenerate
        for frac in CUT_FRACS:
            T_cut = max(1, int(frac * sched.N_total))

            # ── run simulation up to T_cut ──
            state, done = sched.reset_simulator()
            for _ in range(T_cut):
                if done: break
                action = sched.select_physician(state)
                state, _, _, done = sched.step(action)

            if done or state.get("current_patient") is None:
                continue   # episode ended early, skip

            # ── ground truth: only the patients from T_cut onward ──
            t_idx = state["t"]
            future_gt = sched.patients[t_idx:]
            gt_stats[frac].append(collect_patient_stats(future_gt))

            # ── regenerate with POSTERIOR method ──
            # Temporarily monkey-patch _sample_N_future to use posterior
            orig_fn = sched._sample_N_future.__func__ if hasattr(sched._sample_N_future, '__func__') else None
            sched.regenerate_scenario(state, new=1)   # uses _sample_N_future (posterior)
            post_stats[frac].append(collect_patient_stats(sched.regenerated_patients[1:]))  # skip cur_pat

            # ── regenerate with REJECTION method ──
            # Swap regenerate_scenario to use rejection sampler
            s_mid   = float(state["current_patient"]["score"])
            N_curr  = state["patient_count"]
            sched.set_seed(BASE_SEED + ep * 31)   # reset RNG for fair comparison
            sched.generate_instance()
            state2, done2 = sched.reset_simulator()
            for _ in range(T_cut):
                if done2: break
                action = sched.select_physician(state2)
                state2, _, _, done2 = sched.step(action)
            if done2 or state2.get("current_patient") is None:
                continue
            N1f, N2f = sched._sample_N_future_rejection(N_curr, s_mid)
            cur_state2 = state2.get("current_patient")
            cur_pat2   = sched._synthesize_patient_from_state_cur(cur_state2)
            regen2 = [cur_pat2]
            for k in range(N1f):
                regen2.append(sched.generate_patient(cur_pat2._id + 1 + k, 1, score_lower=s_mid))
            for k in range(N2f):
                regen2.append(sched.generate_patient(cur_pat2._id + N1f + 1 + k, 2, score_lower=s_mid))
            rej_stats[frac].append(collect_patient_stats(regen2[1:]))

        if (ep + 1) % 100 == 0:
            print(f"  {ep+1}/{K_EPISODES} episodes done")

    # ── Summarise ──
    def mean_of(stat_list, key):
        vals = [s[key] for s in stat_list if key in s and not isinstance(s[key], list)]
        return np.mean(vals) if vals else float("nan")

    def flatten(stat_list, key):
        out = []
        for s in stat_list:
            if key in s and isinstance(s[key], list):
                out.extend(s[key])
        return np.array(out)

    print(f"\n{'Metric':<22} {'Cut':>6}  {'GT':>8}  {'Posterior':>10}  {'Rejection':>10}  KS(Post) KS(Rej)")
    print("-" * 80)

    fig2, axes2 = plt.subplots(len(CUT_FRACS), 4, figsize=(18, 4 * len(CUT_FRACS)))
    fig2.suptitle(
        "Regenerated horizon stats vs ground truth\n"
        "Posterior (_sample_N_future)  vs  Rejection (_sample_N_future_rejection)",
        fontsize=11, fontweight="bold",
    )

    for ri, frac in enumerate(CUT_FRACS):
        if not gt_stats[frac]:
            continue

        for metric in ["N1", "N2", "N_total"]:
            gt_m  = mean_of(gt_stats[frac],   metric)
            po_m  = mean_of(post_stats[frac],  metric)
            rj_m  = mean_of(rej_stats[frac],   metric)
            print(f"  {metric:<20} {frac:>6.0%}  {gt_m:>8.1f}  {po_m:>10.1f}  {rj_m:>10.1f}")

        for dist_key, col, prio_label in [
            ("score_p1", 0, "Score P1"),
            ("score_p2", 1, "Score P2"),
            ("dur_p1",   2, "Dur P1"),
            ("dur_p2",   3, "Dur P2"),
        ]:
            gt_arr  = flatten(gt_stats[frac],  dist_key)
            po_arr  = flatten(post_stats[frac], dist_key)
            rj_arr  = flatten(rej_stats[frac],  dist_key)

            if len(gt_arr) < 2 or len(po_arr) < 2 or len(rj_arr) < 2:
                continue

            ks_po = ks_2samp(gt_arr, po_arr)
            ks_rj = ks_2samp(gt_arr, rj_arr)
            print(f"  {dist_key:<20} {frac:>6.0%}  "
                  f"μ={gt_arr.mean():.2f}  "
                  f"μ={po_arr.mean():.2f}      "
                  f"μ={rj_arr.mean():.2f}      "
                  f"KS={ks_po.statistic:.3f}({'✓' if ks_po.pvalue>0.05 else '✗'})  "
                  f"KS={ks_rj.statistic:.3f}({'✓' if ks_rj.pvalue>0.05 else '✗'})")

            ax = axes2[ri][col]
            bins = 30
            ax.hist(gt_arr, bins=bins, alpha=0.5, label=f"GT μ={gt_arr.mean():.2f}", color="black",      density=True)
            ax.hist(po_arr, bins=bins, alpha=0.5, label=f"Post μ={po_arr.mean():.2f}", color="steelblue", density=True)
            ax.hist(rj_arr, bins=bins, alpha=0.5, label=f"Rej μ={rj_arr.mean():.2f}",  color="darkorange", density=True)
            ax.set_title(f"{prio_label}  |  cut={frac:.0%}\nKS_post={ks_po.statistic:.3f} KS_rej={ks_rj.statistic:.3f}", fontsize=8)
            ax.legend(fontsize=7)
            if col in (0,1): ax.set_xlabel("Score")
            else:            ax.set_xlabel("Duration (min)")

        print()

    plt.tight_layout()
    plt.savefig("plots/part2_horizon_stats.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("\n→ Plot saved: part2_horizon_stats.png")

else:
    print("Skipped Part 2 (OCS not importable).")