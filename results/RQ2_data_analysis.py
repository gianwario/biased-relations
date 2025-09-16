import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon

BASE = Path("./")

# ---------- Utilities ----------

def safe_num(x):
    """Convert field to a float if possible, otherwise fallback."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, list):
        return float(len(x))
    if isinstance(x, dict):
        try:
            vals = [float(v) for v in x.values()]
            return float(sum(vals) / len(vals)) if vals else None
        except Exception:
            return None
    return None

def cliffs_delta(x, y):
    """Compute Cliff's Delta for paired samples."""
    n = len(x)
    gt, lt = 0, 0
    for xi, yi in zip(x, y):
        if xi > yi: gt += 1
        elif xi < yi: lt += 1
    return (gt - lt) / n

# ---------- Loaders ----------

def load_kn_stats(model_dir: Path):
    """Load analyzed_kn.json (RQ1 stats)."""
    kn_file = model_dir / "analyzed_kn.json"
    if not kn_file.exists():
        return None
    with open(kn_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "avg_ig_kn": data.get("stats", {}).get("average_ig_kn"),
        "avg_base_kn": data.get("stats", {}).get("average_base_kn"),
        "ig_inner_inter": data.get("intersection", {}).get("ig_inner_ave_intersec"),
        "ig_inter_inter": data.get("intersection", {}).get("ig_inter_ave_intersec"),
        "base_inner_inter": data.get("intersection", {}).get("base_inner_ave_intersec"),
        "base_inter_inter": data.get("intersection", {}).get("base_inter_ave_intersec"),
    }

def load_erasing_data(model_dir: Path):
    """Load erasing_summary JSONs in erasing_report/."""
    erasing_dir = model_dir / "erasing_report"
    rows = []
    if not erasing_dir.exists():
        return pd.DataFrame()

    for jf in erasing_dir.glob("*combined.json"):
        with open(jf, "r", encoding="utf-8") as f:
            content = json.load(f)

        if isinstance(content, list):
            records = content
        elif isinstance(content, dict):
            records = [content]
        else:
            continue

        for j in records:
            rows.append({
                "relation": j.get("relation"),
                "num_kn_rel": safe_num(j.get("num_kn_rel")),
                "orig_ppl": safe_num(j.get("original_ppl")),
                "erased_ppl": safe_num(j.get("erased_ppl")),
                "ppl_increase_ratio": safe_num(j.get("ppl_increase_ratio")),
                "other_orig_ppl": safe_num(j.get("other_original_ppl")),
                "other_erased_ppl": safe_num(j.get("other_erased_ppl")),
                "other_ppl_increase_ratio": safe_num(j.get("other_ppl_increase_ratio")),
                "time_s": safe_num(j.get("time")),
            })
    return pd.DataFrame(rows)

# ---------- Main ----------

def main():
    outdir = Path("analysis_rq2")
    outdir.mkdir(exist_ok=True)

    all_rel_rows = []
    all_model_rows = []

    for model_dir in BASE.iterdir():
        if not model_dir.is_dir():
            continue
        if not (model_dir.name.startswith("bert-") or model_dir.name.startswith("bert_")):
            continue

        model = model_dir.name
        kn_stats = load_kn_stats(model_dir)
        df_erasing = load_erasing_data(model_dir)

        if df_erasing.empty:
            continue

        # Clean relations
        df_erasing = df_erasing.dropna(subset=["relation"])

        # Model-level summary
        model_summary = {
            "model": model,
            "n_relations": df_erasing["relation"].nunique(),
            "mean_num_kn_rel": df_erasing["num_kn_rel"].mean(),
            "mean_ppl_increase_ratio": df_erasing["ppl_increase_ratio"].mean(),
            "mean_other_ppl_increase_ratio": df_erasing["other_ppl_increase_ratio"].mean(),
        }
        if kn_stats:
            model_summary.update(kn_stats)
        all_model_rows.append(model_summary)

        # Per-relation rows
        df_erasing["model"] = model
        if kn_stats:
            for k, v in kn_stats.items():
                df_erasing[k] = v
        all_rel_rows.extend(df_erasing.to_dict(orient="records"))

    # Build DataFrames
    df_rel = pd.DataFrame(all_rel_rows)
    df_model = pd.DataFrame(all_model_rows)

    df_rel.to_csv(outdir / "per_relation.csv", index=False)
    df_model.to_csv(outdir / "per_model.csv", index=False)

    # ---------- Correlations ----------
    if not df_rel.empty:
        rho1, p1 = spearmanr(df_rel["num_kn_rel"], df_rel["ppl_increase_ratio"])
        rho2, p2 = spearmanr(df_rel["ig_inner_inter"], df_rel["other_ppl_increase_ratio"])
        with open(outdir / "correlations.txt", "w", encoding="utf-8") as f:
            f.write(f"Correlation num_kn_rel <-> ppl_increase_ratio: rho={rho1:.3f}, p={p1:.3g}\n")
            f.write(f"Correlation ig_inner_inter <-> other_ppl_increase_ratio: rho={rho2:.3f}, p={p2:.3g}\n")

    # ---------- Statistical tests ----------
    if not df_rel.empty:
        valid = df_rel.dropna(subset=["orig_ppl", "erased_ppl"])
        if not valid.empty:
            stat, pval = wilcoxon(valid["erased_ppl"], valid["orig_ppl"], alternative="greater")
            d = cliffs_delta(valid["erased_ppl"].values, valid["orig_ppl"].values)
            with open(outdir / "stats_tests.txt", "w", encoding="utf-8") as f:
                f.write(f"Wilcoxon signed-rank test (erased > original): W={stat}, p={pval:.3g}, n={len(valid)}\n")
                f.write(f"Cliff's Delta: {d:.3f}\n")

    # ---------- Plots ----------
    if not df_rel.empty:
        ax = df_rel.plot.scatter(
            x="num_kn_rel", y="ppl_increase_ratio", alpha=0.7,
            title="Num KNeurons vs PPL Increase (per relation)"
        )
        plt.tight_layout()
        plt.savefig(outdir / "numkn_vs_ppl.png", dpi=300)
        plt.close()

        df_rel.groupby(["relation"])["ppl_increase_ratio"].mean().plot(
            kind="bar", title="Avg PPL Increase per Relation"
        )
        plt.ylabel("PPL increase ratio")
        plt.tight_layout()
        plt.savefig(outdir / "ppl_increase_per_relation.png", dpi=300)
        plt.close()

    if not df_model.empty:
        df_model.round(3).to_latex(outdir / "per_model.tex", index=False)

    print("Analysis complete. Outputs in:", outdir)

if __name__ == "__main__":
    main()
