import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("./")

def collect_kn_data():
    rows = []
    for model_dir in BASE.glob("bert-*"):
        json_path = model_dir / "analyzed_kn.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stats = data.get("stats", {})
        intersection = data.get("intersection", {})

        rows.append({
            "model": model_dir.name,
            "avg_ig_kn": stats.get("average_ig_kn"),
            "avg_base_kn": stats.get("average_base_kn"),
            "ig_inner_inter": intersection.get("ig_inner_ave_intersec"),
            "ig_inter_inter": intersection.get("ig_inter_ave_intersec"),
            "base_inner_inter": intersection.get("base_inner_ave_intersec"),
            "base_inter_inter": intersection.get("base_inter_ave_intersec"),
        })
    return pd.DataFrame(rows)

def main():
    df = collect_kn_data()
    if df.empty:
        print("No analyzed_kn.json files found.")
        return

    # Save raw table
    outdir = Path("analysis_rq1")
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "rq1_kn_summary.csv", index=False)

    # Aggregated statistics
    agg = df.mean(numeric_only=True).to_frame("mean").round(3)
    agg["std"] = df.std(numeric_only=True).round(3)
    agg.to_csv(outdir / "rq1_kn_aggregated.csv")

    print("=== Summary across models ===")
    print(agg)

    # Ratio IG vs baseline
    df["ig_vs_base_ratio"] = df["avg_ig_kn"] / df["avg_base_kn"]

    # Plot avg neurons per model
    ax = df.set_index("model")[["avg_ig_kn", "avg_base_kn"]].plot(
        kind="bar", figsize=(10,5), title="Average neurons per relation (IG vs Baseline)"
    )
    ax.set_ylabel("Average # of neurons")
    plt.tight_layout()
    plt.savefig(outdir / "avg_neurons_per_model.png", dpi=300)
    plt.close()

    # Plot overlaps
    ax = df.set_index("model")[["ig_inner_inter", "base_inner_inter"]].plot(
        kind="bar", figsize=(10,5), title="Within-method overlap (inner intersections)"
    )
    ax.set_ylabel("Average overlap")
    plt.tight_layout()
    plt.savefig(outdir / "overlap_inner.png", dpi=300)
    plt.close()

    ax = df.set_index("model")[["ig_inter_inter", "base_inter_inter"]].plot(
        kind="bar", figsize=(10,5), title="Cross-method overlap (inter intersections)"
    )
    ax.set_ylabel("Average overlap")
    plt.tight_layout()
    plt.savefig(outdir / "overlap_inter.png", dpi=300)
    plt.close()

    # Export a simple narrative-ready table
    df_round = df.round(2)
    df_round.to_latex(outdir / "rq1_kn_summary.tex", index=False)

if __name__ == "__main__":
    main()
