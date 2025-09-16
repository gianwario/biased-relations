#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ3 Analysis (simplified, with correct baseline detection).
- For MLM task: accuracy + perplexity
- For other tasks: accuracy + f1_macro
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
FILES = [
    "binary_incivility_suppression_results.csv",
    "binary_requirement-type_suppression_results.csv",
    "binary_tone-bearing_suppression_results.csv",
    "MLM_req-compl_suppression_results.csv",
    "multiclass_sentiment_suppression_results.csv",
]
OUTDIR = Path("analysis_rq3")
OUTDIR.mkdir(exist_ok=True)

# ---------------- HELPERS ----------------
def safe_percentage_change(after, before):
    if before is None or pd.isna(before) or before == 0:
        return None
    return (after - before) / abs(before)

# ---------------- MAIN ----------------
all_rows = []

for fname in FILES:
    df = pd.read_csv("./se_tasks_benchmark/"+fname)
    df.columns = [c.lower().strip() for c in df.columns]
    
    rel_col = "biased_rel_removed"
    model_col = "model"
    task_col = "task"

    task = df[task_col].iloc[0]
    # Decide metrics
    if "mlm" in fname.lower():
        metrics = ["accuracy", "perplexity"]
    else:
        # prefer accuracy + f1_macro
        # check if "f1_macro" exists, else fallback to "f1"
        metrics = ["accuracy", "f1_macro"] if "f1_macro" in df.columns else ["accuracy", "f1"]

    # Baseline row
    base = df[df[rel_col].isna() | (df[rel_col].astype(str).str.lower() == "none")]
    if base.empty:
        print(f"[WARN] No baseline row found in {fname}")
        continue
    base_row = base.iloc[0]

    df_sup = df[df[rel_col].fillna("").str.upper().str.startswith("BR")]

    for _, r in df_sup.iterrows():
        for m in metrics:
            if m not in df.columns:
                continue
            before = base_row[m]
            after = r[m]
            delta = after - before
            percentage = safe_percentage_change(after, before)

            row = {
                "task": task,
                "relation": r[rel_col],
                "metric": m,
                "value_baseline": before,
                "value_after": after,
                "delta": delta,
                "percentage_change": percentage,
                "model": r[model_col],
            }
            all_rows.append(row)

# ---------------- SAVE ----------------
df_long = pd.DataFrame(all_rows)
df_long.to_csv(OUTDIR / "rq3_results.csv", index=False)
# ---------------- Per-task, per-model summary ----------------
if "model" in df_long.columns:
    summary_task_model = (
        df_long.groupby(["task", "model", "metric"])
               .agg(mean_delta=("delta", "mean"),
                    median_delta=("delta", "median"),
                    min_delta=("delta", "min"),
                    max_delta=("delta", "max"),
                    mean_percentage=("percentage_change", "mean"))
               .reset_index()
    )
    summary_task_model.to_csv(OUTDIR / "rq3_summary_task_model.csv", index=False)
    summary_task_model.round(3).to_latex(OUTDIR / "rq3_summary_task_model.tex", index=False)
# ---------------- Per-task, per-model, per-relation results ----------------
if not df_long.empty:
    per_br = (
        df_long.groupby(["task", "model", "relation", "metric"])
               .agg(delta_mean=("delta", "mean"),
                    delta_median=("delta", "median"))
               .reset_index()
    )
    per_br.to_csv(OUTDIR / "rq3_per_relation.csv", index=False)
    per_br.round(3).to_latex(OUTDIR / "rq3_per_relation.tex", index=False)
# ---------------- Aggregate by relation (BR) ----------------
if not df_long.empty:
    summary_br = (
        df_long.groupby(["relation", "metric"])
               .agg(mean_delta=("delta", "mean"),
                    median_delta=("delta", "median"),
                    min_delta=("delta", "min"),
                    max_delta=("delta", "max"),
                    mean_percentage=("percentage_change", "mean"))
               .reset_index()
    )
    summary_br.to_csv(OUTDIR / "rq3_summary_by_br.csv", index=False)
    summary_br.round(3).to_latex(OUTDIR / "rq3_summary_by_br.tex", index=False)
# Summary
summary = (df_long.groupby(["task", "metric"])
           .agg(mean_delta=("delta", "mean"),
                median_delta=("delta", "median"),
                min_delta=("delta", "min"),   # worst drop
                max_delta=("delta", "max"),   # any improvements
                mean_percentage=("percentage_change", "mean"))
           .reset_index())
summary.to_csv(OUTDIR / "rq3_summary.csv", index=False)
summary.round(3).to_latex(OUTDIR / "rq3_summary.tex", index=False)

# ---------------- PLOTS ----------------
for t in df_long["task"].unique():
    sub = df_long[df_long["task"] == t]
    for m in sub["metric"].unique():
        plt.figure(figsize=(8, 4))
        subm = sub[sub["metric"] == m]
        plt.bar(subm["relation"], subm["delta"])
        plt.title(f"{t} · Δ {m} after suppression")
        plt.ylabel("Delta (after - baseline)")
        plt.xlabel("Relation")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"{t}_{m}_delta.png", dpi=300)
        plt.close()

print(f"Analysis done. Outputs in {OUTDIR.resolve()}")
