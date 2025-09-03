#!/usr/bin/env python3
import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ========================= CONFIG =========================
RESULTS_ROOT = Path("../results")
RELATION_ORDER = [f"BR{str(i).zfill(2)}" for i in range(1, 10)]  # BR01..BR09
SUMMARY_DIRNAME = "erasing_report"

# If you want to restrict to a specific list, set MODEL_DIRS to aliases you have:
MODEL_DIRS: List[str] = []  # e.g., ["bert-base-cased", "bert-large-cased"]; empty -> auto-detect
# =========================================================

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

COLOR_A = "#1f77b4"  # blue
COLOR_B = "#ff7f0e"  # orange
COLOR_C = "#2ca02c"  # green
LINE_CONNECT = "#aaaaaa"

# --------------------- Helpers ---------------------
def find_model_dirs(root: Path) -> List[Path]:
    if MODEL_DIRS:
        return [root / m for m in MODEL_DIRS if (root / m).is_dir()]
    # Auto: any folder in results/* that contains erasing_summary files
    candidates = []
    for p in root.glob("*"):
        if p.is_dir() and any(p.glob("erasing_summary_BR0*.json")):
            candidates.append(p)
        # models that store summaries inside subfolders (less likely):
        elif p.is_dir() and any(p.rglob("erasing_summary_BR0*.json")):
            candidates.append(p)
    return sorted(list(set(candidates)))

def read_erasing_jsons(model_dir: Path) -> Dict[str, Dict]:
    data = {}
    for br in RELATION_ORDER:
        # Search both in model_dir and its subfolders (some pipelines save under kn/)
        paths = list(model_dir.glob(f"erasing_summary_{br}.json"))
        if not paths:
            paths = list(model_dir.rglob(f"erasing_summary_{br}.json"))
        if not paths:
            continue
        fpath = paths[0]
        try:
            with open(fpath, "r") as f:
                content = json.load(f)
                data[br] = content
        except Exception as e:
            print(f"[warn] Could not read {fpath}: {e}")
    return data

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

# --------------------- Writers ---------------------
def write_combined(model_dir: Path, out_dir: Path, rows: List[Dict]):
    # Save JSON
    with open(out_dir / "erasing_summary_combined.json", "w") as f:
        json.dump(rows, f, indent=2)
    # Save CSV
    # Keep stable column order
    cols = [
        "relation",
        "original_accuracy", "erased_accuracy", "erased_ratio_accuracy",
        "num_kn_rel",
        "original_ppl", "erased_ppl", "ppl_increase_ratio",
        "other_original_accuracy", "other_erased_accuracy", "other_erased_ratio_accuracy",
        "other_original_ppl", "other_erased_ppl", "other_ppl_increase_ratio",
        "time"
    ]
    import csv
    with open(out_dir / "erasing_summary_combined.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

def write_stats(out_dir: Path, rows: List[Dict]):
    def col(v): return [to_float(r.get(v)) for r in rows]
    stats = {
        "original_accuracy": compute_stats(col("original_accuracy")),
        "erased_accuracy": compute_stats(col("erased_accuracy")),
        "erased_ratio_accuracy": compute_stats(col("erased_ratio_accuracy")),
        "num_kn_rel": compute_stats(col("num_kn_rel")),
        "original_ppl": compute_stats(col("original_ppl")),
        "erased_ppl": compute_stats(col("erased_ppl")),
        "ppl_increase_ratio": compute_stats(col("ppl_increase_ratio")),
        "other_original_accuracy": compute_stats(col("other_original_accuracy")),
        "other_erased_accuracy": compute_stats(col("other_erased_accuracy")),
        "other_erased_ratio_accuracy": compute_stats(col("other_erased_ratio_accuracy")),
        "other_original_ppl": compute_stats(col("other_original_ppl")),
        "other_erased_ppl": compute_stats(col("other_erased_ppl")),
        "other_ppl_increase_ratio": compute_stats(col("other_ppl_increase_ratio")),
        "time_seconds": compute_stats(col("time")),
        "n_relations": len(rows),
    }
    with open(out_dir / "erasing_summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

# --------------------- Plots ---------------------
def wrap_labels(labels: List[str], width: int = 10) -> List[str]:
    import textwrap
    return ["\n".join(textwrap.wrap(lbl, width=width, break_long_words=False)) for lbl in labels]

def plot_ppl_before_after(rows: List[Dict], title: str, save_path: Path):
    rels = [r["relation"] for r in rows]
    x = np.arange(len(rels))
    orig = np.array([to_float(r["original_ppl"]) for r in rows])
    era  = np.array([to_float(r["erased_ppl"]) for r in rows])
    w = 0.4

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45*len(rels)+1)))
    ax.set_title(title)
    ax.bar(x - w/2, orig, width=w, label="Original PPL", color=COLOR_A)
    ax.bar(x + w/2, era,  width=w, label="Erased PPL",   color=COLOR_B)
    ax.set_xticks(x)
    ax.set_xticklabels(rels, rotation=0)
    ax.set_ylabel("Perplexity")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)

def plot_ppl_increase_ratio(rows: List[Dict], title: str, save_path: Path):
    rels = [r["relation"] for r in rows]
    vals = np.array([to_float(r["ppl_increase_ratio"]) for r in rows])  # can be >1 if 105% -> 1.05
    # If your ratio is already a multiplicative factor (e.g., 1.0549 = +105.49%),
    # convert to percent increase relative to baseline:
    percent = (vals - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45*len(rels)+1)))
    ax.set_title(title)
    ax.bar(np.arange(len(rels)), percent, color=COLOR_C)
    ax.set_xticks(np.arange(len(rels)))
    ax.set_xticklabels(rels)
    ax.set_ylabel("Perplexity Increase (%)")
    ax.axhline(0, color=LINE_CONNECT, linewidth=1)
    # Add value labels
    for i, v in enumerate(percent):
        ax.text(i, v + (1.5 if v >= 0 else -1.5), f"{v:.1f}%", ha="center", va="bottom" if v>=0 else "top", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)

def plot_accuracy_before_after(rows: List[Dict], title: str, save_path: Path):
    rels = [r["relation"] for r in rows]
    x = np.arange(len(rels))
    orig = np.array([to_float(r["original_accuracy"]) for r in rows])
    era  = np.array([to_float(r["erased_accuracy"]) for r in rows])
    w = 0.4

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45*len(rels)+1)))
    ax.set_title(title)
    ax.bar(x - w/2, orig, width=w, label="Original Acc", color=COLOR_A)
    ax.bar(x + w/2, era,  width=w, label="Erased Acc",   color=COLOR_B)
    ax.set_xticks(x)
    ax.set_xticklabels(rels)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)

def plot_erased_ratio_accuracy(rows: List[Dict], title: str, save_path: Path):
    rels = [r["relation"] for r in rows]
    vals = np.array([to_float(r["erased_ratio_accuracy"]) for r in rows])  # fractional (0..1)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45*len(rels)+1)))
    ax.set_title(title)
    ax.bar(np.arange(len(rels)), vals, color=COLOR_B)
    ax.set_xticks(np.arange(len(rels)))
    ax.set_xticklabels(rels)
    ax.set_ylabel("Accuracy Change Ratio")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.02), f"{v*100:.1f}%", ha="center", va="bottom" if v>=0 else "top", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)

def plot_num_kn_vs_ppl_increase(rows: List[Dict], title: str, save_path: Path):
    x = np.array([to_float(r["num_kn_rel"]) for r in rows])
    y = np.array([to_float(r["ppl_increase_ratio"]) for r in rows])
    # Convert multiplicative ratio to % increase for readability
    y_pct = (y - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(title)
    ax.scatter(x, y_pct, s=40, color=COLOR_A)
    ax.set_xlabel("# KNeurons (relation)")
    ax.set_ylabel("Perplexity Increase (%)")

    # Trend line (ignore NaNs)
    mask = (~np.isnan(x)) & (~np.isnan(y_pct))
    if mask.sum() >= 2:
        coeffs = np.polyfit(x[mask], y_pct[mask], deg=1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ys = coeffs[0]*xs + coeffs[1]
        ax.plot(xs, ys, color=LINE_CONNECT, linewidth=1.0, label=f"trend (slope={coeffs[0]:.2f})")
        ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)

# --------------------- Main per model ---------------------
def process_model(model_dir: Path):
    model_name = model_dir.name
    out_dir = model_dir / SUMMARY_DIRNAME
    ensure_dir(out_dir)

    data = read_erasing_jsons(model_dir)
    if not data:
        print(f"[skip] No erasing_summary_BR0X.json found in {model_dir}")
        return

    # Build ordered rows BR01..BR09, keep only those we found
    rows = []
    for br in RELATION_ORDER:
        if br in data:
            rows.append({
                "relation": br,
                "original_accuracy": to_float(data[br].get("original_accuracy")),
                "erased_accuracy": to_float(data[br].get("erased_accuracy")),
                "erased_ratio_accuracy": to_float(data[br].get("erased_ratio_accuracy")),
                "num_kn_rel": int(data[br].get("num_kn_rel")) if data[br].get("num_kn_rel") is not None else np.nan,
                "original_ppl": to_float(data[br].get("original_ppl")),
                "erased_ppl": to_float(data[br].get("erased_ppl")),
                "ppl_increase_ratio": to_float(data[br].get("ppl_increase_ratio")),
                "other_original_accuracy": to_float(data[br].get("other_original_accuracy")),
                "other_erased_accuracy": to_float(data[br].get("other_erased_accuracy")),
                "other_erased_ratio_accuracy": to_float(data[br].get("other_erased_ratio_accuracy")),
                "other_original_ppl": to_float(data[br].get("other_original_ppl")),
                "other_erased_ppl": to_float(data[br].get("other_erased_ppl")),
                "other_ppl_increase_ratio": to_float(data[br].get("other_ppl_increase_ratio")),
                "time": to_float(data[br].get("time")),
            })

    if not rows:
        print(f"[skip] No valid BR rows for {model_name}")
        return

    # Write combined
    write_combined(model_dir, out_dir, rows)
    # Write stats
    write_stats(out_dir, rows)

    # Plots (fixed relation order)
    plot_ppl_before_after(rows, f"{model_name} · Perplexity Before/After Erasing", out_dir / "ppl_before_after")
    plot_ppl_increase_ratio(rows, f"{model_name} · Perplexity Increase After Erasing", out_dir / "ppl_increase_ratio")
    plot_accuracy_before_after(rows, f"{model_name} · Accuracy Before/After Erasing (Own)", out_dir / "acc_before_after")
    plot_erased_ratio_accuracy(rows, f"{model_name} · Accuracy Change Ratio (Own)", out_dir / "erased_ratio_accuracy")
    plot_num_kn_vs_ppl_increase(rows, f"{model_name} · #KNeurons vs PPL Increase", out_dir / "num_kn_vs_ppl_increase")

    print(f"[ok] Wrote reports to {out_dir}")

def main():
    if not RESULTS_ROOT.exists():
        print(f"[error] RESULTS_ROOT not found: {RESULTS_ROOT}")
        return
    model_dirs = find_model_dirs(RESULTS_ROOT)
    if not model_dirs:
        print(f"[warn] No model folders with erasing summaries under {RESULTS_ROOT}")
        return
    for md in model_dirs:
        process_model(md)

if __name__ == "__main__":
    main()
