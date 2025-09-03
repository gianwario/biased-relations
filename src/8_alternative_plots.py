import os, json, textwrap
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ------------------------------ Styling ------------------------------
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

COLOR_OURS = "#1f77b4"     # blue (colorblind-friendly)
COLOR_BASE = "#ff7f0e"     # orange (colorblind-friendly)
LINE_CONNECT = "#AAAAAA"

# ------------------------------ Helpers ------------------------------
def alias(model_name: str) -> str:
    parts = model_name.split("/")
    return parts[1] if len(parts) > 1 else parts[0]

def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def get_relations(modified_rlts: Dict) -> List[str]:
    # preserve insertion order from JSON keys
    rels = []
    for k in modified_rlts.keys():
        rel = k.split("-")[-1]
        if rel not in rels:
            rels.append(rel)
    return rels

def collect_pairs(
    rel_list: List[str],
    ours: Dict,
    base: Dict,
    ours_key_prefix: str,
    base_key_prefix: str,
    metric_key: str
) -> List[Tuple[str, float, float]]:
    """
    Returns [(rel, ours_value, base_value), ...] in fractional units (e.g., 0.12=12%).
    Missing values are skipped.
    """
    rows = []
    for rel in rel_list:
        ok = f"{ours_key_prefix}-{rel}"
        bk = f"{base_key_prefix}-{rel}"
        if ok in ours and bk in base and metric_key in ours[ok] and metric_key in base[bk]:
            rows.append((rel, float(ours[ok][metric_key]), float(base[bk][metric_key])))
    return rows

def wrap_labels(labels: List[str], width: int = 12) -> List[str]:
    return ["\n".join(textwrap.wrap(lbl, width=width, break_long_words=False)) for lbl in labels]

def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# ------------------------------ Plotters ------------------------------
def plot_slope(
    rows: List[Tuple[str, float, float]],
    title: str,
    ylabel: str,
    save_path: Path
):
    """
    Slope chart (Cleveland dot plot):
    rows: [(rel, ours, base), ...]
    sort_by: "ours_desc" | "gap_desc" | "alpha"
    """
    if not rows:
        return

    rels = [r[0] for r in rows]
    ours = np.array([r[1] for r in rows])
    base = np.array([r[2] for r in rows])

    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(rows) + 1)))
    ax.set_title(title, pad=10)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(ylabel)
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(rels, width=14))

    # connecting lines
    for i in range(len(rows)):
        ax.plot([base[i], ours[i]], [y[i], y[i]], color=LINE_CONNECT, lw=1.0, zorder=1)

    ax.scatter(base, y, s=35, label="Baseline", color=COLOR_BASE, zorder=2)
    ax.scatter(ours, y, s=35, label="Ours", color=COLOR_OURS, zorder=3)

    # value labels (only if space permits)
    def label_points(vals, dy=0.15, color="black"):
        for xi, yi in zip(vals, y):
            ax.text(xi, yi + dy, f"{xi*100:.1f}%", ha="center", va="bottom", fontsize=9, color=color)

    label_points(base, dy=-0.25, color=COLOR_BASE)
    label_points(ours, dy=+0.15, color=COLOR_OURS)

    # nice margins
    xmin = min(base.min(), ours.min())
    xmax = max(base.max(), ours.max())
    span = xmax - xmin
    ax.set_xlim(xmin - span * 0.08, xmax + span * 0.08)

    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)

def plot_grouped_bars_horizontal(
    rows: List[Tuple[str, float, float]],
    title: str,
    ylabel: str,
    save_path: Path,
    sort_by: str = "ours_desc"
):
    """
    Horizontal grouped bars with percent formatting and value labels.
    """
    if not rows:
        return

    if sort_by == "gap_desc":
        rows = sorted(rows, key=lambda x: (x[1] - x[2]), reverse=True)
    elif sort_by == "alpha":
        rows = sorted(rows, key=lambda x: x[0])
    else:  # "ours_desc"
        rows = sorted(rows, key=lambda x: x[1], reverse=True)

    rels = [r[0] for r in rows]
    ours = np.array([r[1] for r in rows])
    base = np.array([r[2] for r in rows])

    n = len(rows)
    y = np.arange(n)
    h = 0.38

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * n + 1)))
    ax.set_title(title, pad=10)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(ylabel)
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(rels, width=14))

    ax.barh(y - h/2, ours, height=h, label="Ours", color=COLOR_OURS)
    ax.barh(y + h/2, base, height=h, label="Baseline", color=COLOR_BASE)

    # value labels on bars
    for xi, yi in zip(ours, y - h/2):
        ax.text(xi, yi, f" {xi*100:.1f}%", va="center", ha="left", fontsize=9)
    for xi, yi in zip(base, y + h/2):
        ax.text(xi, yi, f" {xi*100:.1f}%", va="center", ha="left", fontsize=9)

    # tidy x-limits with padding
    xmin = min(ours.min(), base.min())
    xmax = max(ours.max(), base.max())
    span = xmax - xmin
    ax.set_xlim(xmin - span * 0.08, xmax + span * 0.12)

    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)

# ------------------------------ Main ------------------------------
model_list = [
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-uncased",
    "bert-large-uncased",
    "answerdotai/ModernBERT-large",
    "answerdotai/ModernBERT-base",
    # FINETUNED models:
    "aieng-lab/bert-large-cased_requirement-completion",
    "aieng-lab/ModernBERT-large_requirement-completion",
    "aieng-lab/bert-large-cased_incivility",
    "aieng-lab/ModernBERT-large_incivility",
    "aieng-lab/bert-large-cased_tone-bearing",
    "aieng-lab/ModernBERT-large_tone-bearing",
    "aieng-lab/bert-large-cased_sentiment",
    "aieng-lab/ModernBERT-large_sentiment",
    "aieng-lab/bert-large-cased_requirement-type",
    "aieng-lab/ModernBERT-large_requirement-type",
]

# Choose your preferred plot type here:
PLOT_KIND = "slope"    # "slope" or "bars"

for model_name in model_list:
    model_alias = alias(model_name)
    kn_dir = Path(f"../results/{model_alias}/kn/")
    fig_dir = Path(f"../results/{model_alias}/figs_pretty/")
    ensure_dirs(fig_dir)

    mod_path = kn_dir / "modify_activation_rlt.json"
    base_path = kn_dir / "base_modify_activation_rlt.json"

    if not base_path.exists() or not mod_path.exists():
        # Skip models without both files to avoid breaking the loop
        print(f"[skip] Missing files for {model_alias}")
        continue

    modified_rlts = load_json(mod_path)
    base_modified_rlts = load_json(base_path)

    rel_list = get_relations(modified_rlts)

    # ---------------- suppress ----------------
    sup_rows = collect_pairs(
        rel_list,
        ours=modified_rlts,
        base=base_modified_rlts,
        ours_key_prefix="kn_bag",
        base_key_prefix="base_kn_bag",
        metric_key="rm_own:ave_delta_ratio"
    )

    title_sup = f"{model_alias} · Suppress (remove activation)"
    ylabel_sup = "Correct Probability Change Ratio"

    if PLOT_KIND == "slope":
        plot_slope(
            sup_rows,
            title=title_sup,
            ylabel=ylabel_sup,
            save_path=fig_dir / "suppress_pretty",
            
        )
    else:
        plot_grouped_bars_horizontal(
            sup_rows,
            title=title_sup,
            ylabel=ylabel_sup,
            save_path=fig_dir / "suppress_pretty",
            
        )

    # ---------------- amplify ----------------
    amp_rows = collect_pairs(
        rel_list,
        ours=modified_rlts,
        base=base_modified_rlts,
        ours_key_prefix="kn_bag",
        base_key_prefix="base_kn_bag",
        metric_key="eh_own:ave_delta_ratio"
    )

    title_amp = f"{model_alias} · Amplify (enhance activation)"
    ylabel_amp = "Correct Probability Change Ratio"

    if PLOT_KIND == "slope":
        plot_slope(
            amp_rows,
            title=title_amp,
            ylabel=ylabel_amp,
            save_path=fig_dir / "amplify_pretty",
            
        )
    else:
        plot_grouped_bars_horizontal(
            amp_rows,
            title=title_amp,
            ylabel=ylabel_amp,
            save_path=fig_dir / "amplify_pretty",
            
        )
