"""
AutoRAGEvals — publication-quality figures.

Generates four charts and saves them as PNGs in figures/.

Run:
    python visualizations.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.8,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "figure.dpi":        150,
})

COLOR_KEPT     = "#2ecc71"   # green
COLOR_REVERTED = "#e74c3c"   # red
COLOR_BASELINE = "#3498db"   # blue
COLOR_BEST     = "#e67e22"   # orange
COLOR_RAGAS    = "#2c3e50"   # dark navy
COLOR_ROUGE    = "#8e44ad"   # purple

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open("experiment_log.json", encoding="utf-8") as f:
    log = json.load(f)

with open("held_out_results.json", encoding="utf-8") as f:
    held_out = json.load(f)

# Adversarial set scores (from log)
adv_baseline_scores = next(e["scores"] for e in log if e["type"] == "baseline")
adv_best_scores     = max(
    (e["scores"] for e in log if e["kept"]),
    key=lambda s: s["overall"],
)

# Held-out set scores
ho_baseline_scores = held_out["baseline"]["scores"]
ho_best_scores     = held_out["best"]["scores"]

METRICS_DISPLAY = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer\nRelevancy",
    "context_precision": "Context\nPrecision",
    "context_recall":    "Context\nRecall",
    "overall":           "Overall",
}


# ---------------------------------------------------------------------------
# Figure 1 — Iteration score chart
# ---------------------------------------------------------------------------
def fig1_iteration_scores() -> None:
    iterations   = [e["iteration"] for e in log]
    overall      = [e["scores"]["overall"] for e in log]
    kept         = [e["kept"] for e in log]
    std_vals     = [e.get("scores_std", {}).get("overall", None) for e in log]
    baseline_val = overall[0]

    # x-axis labels
    labels = []
    for e in log:
        if e["type"] == "baseline":
            labels.append("Baseline")
        else:
            p   = e["proposal"]
            old = p["old_value"]
            new = p["new_value"]
            # shorten long values
            if isinstance(old, str) and len(old) > 10:
                old = old[:7] + "…"
            if isinstance(new, str) and len(new) > 10:
                new = new[:7] + "…"
            labels.append(f"{p['parameter']}\n{old}→{new}")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw connecting line
    ax.plot(iterations, overall, color="#95a5a6", linewidth=1.2,
            zorder=1, linestyle="--")

    # Plot each point coloured by kept/reverted
    for i, (it, ov, kp, sd) in enumerate(
            zip(iterations, overall, kept, std_vals)):
        color  = COLOR_KEPT if kp else COLOR_REVERTED
        marker = "o"
        zorder = 3

        if sd is not None:
            ax.errorbar(it, ov, yerr=sd, fmt="none",
                        ecolor=color, elinewidth=1.5, capsize=4, zorder=2)

        ax.scatter(it, ov, color=color, s=90, zorder=zorder,
                   marker=marker, edgecolors="white", linewidths=0.8)

    # Baseline dashed line
    ax.axhline(baseline_val, color="#7f8c8d", linewidth=1.2,
               linestyle=":", label=f"Baseline ({baseline_val:.4f})")

    # Legend handles
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_KEPT,
               markersize=9, label="Kept"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_REVERTED,
               markersize=9, label="Reverted"),
        Line2D([0], [0], color="#7f8c8d", linewidth=1.2, linestyle=":",
               label=f"Baseline ({baseline_val:.4f})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    ax.set_xticks(iterations)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Iteration / Parameter Change", labelpad=8)
    ax.set_ylabel("Overall RAGAS Score")
    ax.set_title("Optimizer Iteration Scores (Adversarial Set)")
    ax.set_ylim(0.65, 0.82)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_iteration_scores.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Per-metric bar chart (adversarial set)
# ---------------------------------------------------------------------------
def fig2_per_metric_bars() -> None:
    metric_keys = list(METRICS_DISPLAY.keys())
    x           = np.arange(len(metric_keys))
    width       = 0.35

    baseline_vals = [adv_baseline_scores[m] for m in metric_keys]
    best_vals     = [adv_best_scores[m]     for m in metric_keys]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_b = ax.bar(x - width / 2, baseline_vals, width,
                    label="Baseline (top_k=3)", color=COLOR_BASELINE,
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_e = ax.bar(x + width / 2, best_vals, width,
                    label="Best (top_k=7)", color=COLOR_BEST,
                    alpha=0.85, edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar in list(bars_b) + list(bars_e):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([METRICS_DISPLAY[m] for m in metric_keys])
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Baseline vs Best Config — Per-Metric Scores (Adversarial Set)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_per_metric_bars.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Generalization chart
# ---------------------------------------------------------------------------
def fig3_generalization() -> None:
    conditions = [
        "Adversarial\nBaseline",
        "Adversarial\nBest",
        "Held-out\nBaseline",
        "Held-out\nBest",
    ]
    scores = [
        adv_baseline_scores["overall"],
        adv_best_scores["overall"],
        ho_baseline_scores["overall"],
        ho_best_scores["overall"],
    ]
    colors = [COLOR_BASELINE, COLOR_BEST, COLOR_BASELINE, COLOR_BEST]
    alphas = [0.85, 0.85, 0.55, 0.55]   # lighter shade for held-out

    x   = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, scores, width=0.5,
                  color=[c for c in colors],
                  alpha=1.0, edgecolor="white", linewidth=0.5)

    # Apply per-bar alpha manually
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # Value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{score:.4f}", ha="center", va="bottom", fontsize=10)

    # Bracket annotations for generalization gaps
    def draw_gap(ax, x1, x2, y, label, color):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<->", color=color,
                                   lw=1.3))
        ax.text((x1 + x2) / 2, y + 0.012, label,
                ha="center", va="bottom", fontsize=8.5, color=color)

    gap_y = max(scores) + 0.06
    draw_gap(ax, 0, 2, gap_y,
             f"Gen. gap baseline\n−{adv_baseline_scores['overall'] - ho_baseline_scores['overall']:.4f}",
             COLOR_BASELINE)
    draw_gap(ax, 1, 3, gap_y,
             f"Gen. gap best\n−{adv_best_scores['overall'] - ho_best_scores['overall']:.4f}",
             COLOR_BEST)

    # Legend for color meaning
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_BASELINE, alpha=0.85, label="Baseline config (top_k=3)"),
        Patch(facecolor=COLOR_BEST,     alpha=0.85, label="Best config (top_k=7)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("Overall RAGAS Score (0–1)")
    ax.set_title("Generalization Gap: Adversarial Set vs Held-out Set")
    ax.set_ylim(0, gap_y + 0.12)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_generalization.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4 — ROUGE-L vs RAGAS overall (dual y-axis)
# ---------------------------------------------------------------------------
def fig4_rouge_vs_ragas() -> None:
    iterations = [e["iteration"] for e in log]
    overall    = [e["scores"]["overall"] for e in log]
    rouge_l    = [e["scores"]["rouge_l"] for e in log]
    std_ragas  = [e.get("scores_std", {}).get("overall", None) for e in log]
    std_rouge  = [e.get("scores_std", {}).get("rouge_l", None) for e in log]

    labels = []
    for e in log:
        if e["type"] == "baseline":
            labels.append("Baseline")
        else:
            p   = e["proposal"]
            old = p["old_value"]
            new = p["new_value"]
            if isinstance(old, str) and len(old) > 8:
                old = old[:5] + "…"
            if isinstance(new, str) and len(new) > 8:
                new = new[:5] + "…"
            labels.append(f"{p['parameter']}\n{old}→{new}")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    # RAGAS overall on ax1
    line1, = ax1.plot(iterations, overall, color=COLOR_RAGAS,
                      linewidth=2.0, marker="o", markersize=7,
                      label="RAGAS Overall", zorder=3)
    for i, (it, ov, sd) in enumerate(zip(iterations, overall, std_ragas)):
        if sd is not None:
            ax1.errorbar(it, ov, yerr=sd, fmt="none",
                         ecolor=COLOR_RAGAS, elinewidth=1.5,
                         capsize=4, zorder=2)

    # ROUGE-L on ax2
    line2, = ax2.plot(iterations, rouge_l, color=COLOR_ROUGE,
                      linewidth=2.0, marker="s", markersize=7,
                      linestyle="--", label="ROUGE-L", zorder=3)
    for i, (it, rl, sd) in enumerate(zip(iterations, rouge_l, std_rouge)):
        if sd is not None:
            ax2.errorbar(it, rl, yerr=sd, fmt="none",
                         ecolor=COLOR_ROUGE, elinewidth=1.5,
                         capsize=4, zorder=2)

    # Annotate the iter-7 ROUGE-L spike
    it7_idx = next(i for i, e in enumerate(log) if e["iteration"] == 7)
    ax2.annotate(
        f"Concise prompt\nROUGE-L={rouge_l[it7_idx]:.3f}",
        xy=(7, rouge_l[it7_idx]),
        xytext=(5.8, rouge_l[it7_idx] - 0.025),
        fontsize=8.5,
        color=COLOR_ROUGE,
        arrowprops=dict(arrowstyle="->", color=COLOR_ROUGE, lw=1.0),
    )

    # Annotate the iter-6 ROUGE-L drop
    it6_idx = next(i for i, e in enumerate(log) if e["iteration"] == 6)
    ax2.annotate(
        f"Chain-of-thought\nROUGE-L={rouge_l[it6_idx]:.3f}",
        xy=(6, rouge_l[it6_idx]),
        xytext=(4.5, rouge_l[it6_idx] + 0.035),
        fontsize=8.5,
        color=COLOR_ROUGE,
        arrowprops=dict(arrowstyle="->", color=COLOR_ROUGE, lw=1.0),
    )

    ax1.set_xticks(iterations)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_xlabel("Iteration / Parameter Change", labelpad=8)
    ax1.set_ylabel("RAGAS Overall Score", color=COLOR_RAGAS)
    ax1.tick_params(axis="y", labelcolor=COLOR_RAGAS)
    ax1.set_ylim(0.65, 0.83)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    ax2.set_ylabel("ROUGE-L Score", color=COLOR_ROUGE)
    ax2.tick_params(axis="y", labelcolor=COLOR_ROUGE)
    ax2.set_ylim(0.0, 0.28)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Combined legend
    lines = [line1, line2]
    labels_leg = [l.get_label() for l in lines]
    ax1.legend(lines, labels_leg, loc="lower left", fontsize=10)

    ax1.set_title("RAGAS Overall vs ROUGE-L Across Optimizer Iterations")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig4_rouge_vs_ragas.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    fig1_iteration_scores()
    fig2_per_metric_bars()
    fig3_generalization()
    fig4_rouge_vs_ragas()
    print("Done. All figures saved to figures/")
