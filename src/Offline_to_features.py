# ---------- 1) Build feature names (matches your spec) ----------
def build_feature_names(P: int) -> list[str]:
    names: list[str] = []
    # Context
    names += ["t_current", 
              "remaining_release", 
              "priority_is_p1", 
              "arrival_score", 
              "duration"]
    # Per-physician blocks
    for p in range(P):
        names += [
            f"phys{p}_is_preferred",
            f"phys{p}_cap_remaining_norm",
            f"phys{p}_acc_p1_norm",
            f"phys{p}_acc_p2_norm",
            f"phys{p}_load_pressure"
        ]
    # Masks
    for p in range(P):
        names.append(f"mask_phys{p}")
    names.append("mask_reject")
    return names


# ---------- 2) Converter that outputs named columns ----------
import os, glob
import pandas as pd
from typing import Optional, List, Dict, Any
import ast
import re


def convert_solutions_to_features_csv_named(
    N:int,
    folder: str,
    out_csv: str,
    file_glob: str = "*_schedule.csv",
    Rmax: Optional[int] = None,
    cap_per_phys: Optional[int] = None,
    P_override: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reads offline *_schedule.csv files and writes a single CSV with:
      - Named feature columns from build_feature_names(P)
      - Labels/IDs: expert_action, instance, step_t, n, pid, priority, preferred_phys
    """
    paths = sorted(glob.glob(os.path.join(folder, file_glob)))
    if not paths:
        raise FileNotFoundError(f"No schedule CSVs in {folder!r} matching {file_glob!r}")

    all_rows: List[Dict[str, Any]] = []
    session = 240
    eps = 1e-9

    for path in paths:
        df = pd.read_csv(path)
        df = df[df["kind"].isin(["assign", "reject"])].copy()
        df['eligible'] = df['eligible'].apply(ast.literal_eval)
        if df.empty:
            continue
        df.sort_values(["n"], inplace=True)

        # Infer N and P (unless overridden)
        if P_override is not None:
            P = int(P_override)
        else:
            if (df["kind"] == "assign").any():
                P = int(df.loc[df["kind"] == "assign", "p"].max()) + 1
            else:
                P = int(df["preferred_phys"].max()) + 1 if "preferred_phys" in df.columns else 0

        # Precompute header for this P
        feat_names = build_feature_names(P)
        n_feat = len(feat_names)

        # Running accumulators BEFORE each decision
        assigned_per_phys = [0] * P
        acc_p1_by_phys = [0] * P
        acc_p2_by_phys = [0] * P
        accepted_total = 0
        work_per_phys = [0.0] * P 

        inst_tag = os.path.splitext(os.path.basename(path))[0]
        inst_id = int(re.search(r'\d+', inst_tag).group())

        for t, row in enumerate(df.itertuples(index=False)):
            # Context
            t_current = t / max(1, N)
            if Rmax is not None and Rmax > 0:
                remaining_release = max(0, Rmax - accepted_total) / float(Rmax)
            else:
                remaining_release = 0.0
            priority_is_p1 = 1 if int(row.priority) == 1 else 0
            feats = [t_current, remaining_release, priority_is_p1, row.score, row.duration/60]

            # Per-phys features (BEFORE applying current decision)
            if cap_per_phys is not None and cap_per_phys > 0:
                cap_rem = [max(0, cap_per_phys - x) for x in assigned_per_phys]
                cap_rem_norm = [cr / float(cap_per_phys) for cr in cap_rem]
                mask_phys = [1 if cr <= 0 else 0 for cr in cap_rem]
            else:
                cap_rem_norm = [1.0] * P
                mask_phys = [0] * P
            
            mask_phys = [max(mask_phys[i], 1 - row.eligible[i]) for i in range(P)]
            
            p1_total_acc = sum(acc_p1_by_phys)
            p2_total_acc = sum(acc_p2_by_phys)
            if cap_per_phys is not None and cap_per_phys > 0:
                acc_p1_norm = [c for c in acc_p1_by_phys]
                acc_p2_norm = [c for c in acc_p2_by_phys]
            else:
                acc_p1_norm = [float(c) for c in acc_p1_by_phys]
                acc_p2_norm = [float(c) for c in acc_p2_by_phys]
                
            load_pressure = [w / max(eps, session) for w in work_per_phys]
            preferred_phys = int(row.preferred_phys)
            for p in range(P):
                feats.append(1 if p == preferred_phys else 0)   # is_preferred
                feats.append(cap_rem_norm[p])                   # cap_remaining_norm
                feats.append(acc_p1_norm[p])                    # acc_p1_norm
                feats.append(acc_p2_norm[p])                    # acc_p2_norm
                feats.append(load_pressure[p])  
            
            feats.extend(mask_phys)  # mask_phys{p}
            feats.append(0)          # mask_reject

            # Sanity: feature length must match header
            if len(feats) != n_feat:
                raise RuntimeError(f"Feature length {len(feats)} != header length {n_feat} for file {path} at step {t}")

            # Label
            expert_action = int(row.p) if row.kind == "assign" else P

            # Build record with named feature columns
            rec = {feat_names[i]: feats[i] for i in range(n_feat)}
            rec.update({
                "p1_total_acc":p1_total_acc,
                "p2_total_acc":p2_total_acc,
                "label": expert_action,
                "expert_action": expert_action,
                "instance": inst_tag,
                "instance_id":inst_id,
                "step_t": t,
                "n": int(row.n),
                "pid": str(row.pid),
                "priority": int(row.priority),
                "preferred_phys": int(row.preferred_phys)
            })
            all_rows.append(rec)

            # Update accumulators AFTER applying decision
            if row.kind == "assign":
                p_sel = int(row.p)
                assigned_per_phys[p_sel] += 1
                accepted_total += 1
                if int(row.priority) == 1:
                    acc_p1_by_phys[p_sel] += 1
                else:
                    acc_p2_by_phys[p_sel] += 1
                y_np = 1.0
                if hasattr(row, "Y") and isinstance(row.Y, dict):
                    y_np = float(row.Y.get(p_sel, 1.0))
                work_per_phys[p_sel] += float(row.duration) * y_np


    out_df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df

N = 100
P = 4
I = 8
Rmax = 80
L = 20

df = convert_solutions_to_features_csv_named(
    N=N,
    folder   ="../data/Deterministic/Instances_N100_s8_P4_I8_L20_1_4_1_200_50_0_10/instances_solution",
    out_csv  ="../data/Deterministic/Instances_N100_s8_P4_I8_L20_1_4_1_200_50_0_10/instances_solution/offline_features_data.csv",
    file_glob="*_schedule.csv",
    Rmax=Rmax,
    cap_per_phys=L,
    P_override=P,
)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
