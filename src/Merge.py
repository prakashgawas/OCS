#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:26:10 2025

@author: prakashgawas
"""

import os
from typing import Optional, Dict, Any
import pandas as pd
import json
import csv
import glob

def merge_files(folder: str, output_file: str, files:str = "OCS", remain:str='schedule',  start_id=1, end_id=1000):
    dfs = []
    for i in range(start_id, end_id + 1):
        filename = os.path.join(folder, f"{files}_{i}_{remain}.csv")
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            df['instance_id'] = i  # optional: keep track of which file it came from
            dfs.append(df)
        else:
            print(f"File not found: {filename}")

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Merged {len(dfs)} files into {output_file}")
    else:
        print("No files found to merge.")

def _num(x):
    # Safe cast for numbers (handles None/str gracefully)
    try:
        return float(x)
    except Exception:
        return x

def combine_json_meta_to_csv(
    folder: str,
    output_csv: str,
    base: str = "OCS",
    start_id: int = 1,
    end_id: int = 1000,
    suffix: str = "_solution.json",
):
    """
    Combine meta fields from many JSON files into one CSV.

    Expects files like: {folder}/{base}_{i}{suffix}
    where each file looks like:
      {
        "meta": {
          "objective": ...,
          "status": ...,
          "termination": ...,
          "gap": ...,
          "solve_time": ...,
          "breakdown": {
            "reject_pen_total": ...,
            "overtime_pen_total": ...,
            "nonpref_pen_total": ...,
            "objective_total": ...,
            "objective_check": ...
          },
          ... (other fields ignored)
        },
        ... (other top-level keys ignored)
      }

    Writes a CSV with columns:
      file_id, objective, status, termination, gap, solve_time,
      reject_pen_total, overtime_pen_total, nonpref_pen_total,
      objective_total, objective_check
    """
    cols = [
        "file_id",
        "objective", "status", "termination", "gap", "solve_time",
        "reject_pen_total", "overtime_pen_total", "nonpref_pen_total",
        "objective_total", "objective_check",
    ]

    rows = []
    for i in range(start_id, end_id + 1):
        path = os.path.join(folder, f"{base}_{i}{suffix}")
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            # bad json, skip
            continue

        meta: Optional[Dict[str, Any]] = data.get("meta")
        if not isinstance(meta, dict):
            continue

        breakdown: Dict[str, Any] = meta.get("breakdown", {}) or {}

        row = {
            "file_id": i,
            "objective": _num(meta.get("objective")),
            "status": meta.get("status"),
            "termination": meta.get("termination"),
            "gap": _num(meta.get("gap")),
            "solve_time": _num(meta.get("solve_time")),
            "reject_pen_total": _num(breakdown.get("reject_pen_total")),
            "overtime_pen_total": _num(breakdown.get("overtime_pen_total")),
            "nonpref_pen_total": _num(breakdown.get("nonpref_pen_total")),
            "objective_total": _num(breakdown.get("objective_total")),
            "objective_check": _num(breakdown.get("objective_check")),
        }
        rows.append(row)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Merging complete")

# optional fast parser
try:
    import orjson as _json
    def _loads(b: bytes): return _json.loads(b)
except Exception:
    import json as _json
    def _loads(b: bytes): return _json.loads(b.decode("utf-8"))

def combine_flat_jsons_to_csv(
    folder: str,
    pattern: str = "Sim_Stats_*.json",
    output_csv: str = "merged_stats.csv",
    include_source: bool = True,
) -> str:
    """
    Merge many flat JSON files (episode-level stats) into a single CSV.

    • Streams rows: minimal memory
    • Auto-discovers union of keys in a cheap first pass
    • Writes once with a stable header (no header rewrites)
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files found: {os.path.join(folder, pattern)}")

    # 1) First pass: collect union of keys (cheap—just read + parse)
    all_keys = set()
    for p in paths:
        try:
            with open(p, "rb") as f:
                data = _loads(f.read())
            if isinstance(data, dict):
                all_keys.update(data.keys())
        except Exception:
            # skip unreadable/bad JSON
            continue

    if include_source:
        all_keys.add("source_file")

    # Prefer a human-friendly ordering up front
    preferred = [
        "release_Rmax", "release_fraction_used",
        "episode_length", "total_reward",
        "accepted_total", "rejected_total",
        "rejected_p1", "rejected_p2",
        "forced_rejects_p1", "forced_rejects_p2", "forced_rejects_total",
        "nonpref_assignments_total",
        "overtime_minutes_total",
        "reject_penalty", "nonpref_penalty", "overtime_penalty",
        "total_cost",
    ]
    header = [k for k in preferred if k in all_keys] + sorted(k for k in all_keys if k not in preferred)

    # 2) Second pass: stream rows directly to CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    written = 0
    with open(output_csv, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for p in paths:
            try:
                with open(p, "rb") as f:
                    data = _loads(f.read())
                if not isinstance(data, dict):
                    continue
                if include_source:
                    data = {**data, "source_file": os.path.basename(p)}
                w.writerow(data)
                written += 1
            except Exception:
                # skip bad file
                continue

    if written == 0:
        raise RuntimeError("No valid JSON rows were written.")
    print(f"Wrote {written} rows → {output_csv}")
    return output_csv


if __name__ == "__main__":
    base_path = "../data/Deterministic/Instances_N100_s8_P4_I8_L20_1_4_1_200_50_0_10/"
    #base_path = "..data/Data_Dagger_0.8_flat/Instances_N100_s8_P4_I8_L20_1_4_1_200_50_0_10/NN_iter1_runs1000_um1_new1_stoch1_sc30_van1_gat0_tl30_mg0.01_nc5/"

    folder_path = base_path + "instances_solution/"  # or the folder where the files are located
    output_path = folder_path + "merged_solutions.csv"
    merge_files(folder_path, output_path, files="OCS" )
    output_path = folder_path + "combined_status.csv"
    combine_json_meta_to_csv(folder_path, output_path, base="OCS")
    folder_path = base_path + "instances/"  # or the folder where the files are located
    output_path = folder_path + "merged_instances_AYE.csv"
    merge_files(folder_path, output_path, files="OCS", remain = 'AYE' )
    output_path = folder_path + "merged_instances.csv"
    merge_files(folder_path, output_path, files="OCS", remain = 'patients' )
    folder_path = base_path + "greedy/" 
    output_path = folder_path + "combined_stats_greedy.csv"
    combine_flat_jsons_to_csv(folder_path, output_csv= output_path)



