"""
views/visualizer.py
--------------------
Generates all charts for the benchmark analysis.

Charts produced
---------------
1. score_distribution.png   — Violin + boxplot of suspicion scores per profile
2. confusion_matrix.png     — Heatmap of the 3×3 confusion matrix
3. roc_curve.png            — ROC curve for LSTM probability (binary cheater detection)
4. flag_frequency.png       — Bar chart of flag trigger rates per profile

Each chart is:
  - Displayed interactively via plt.show()
  - Saved as a high-resolution PNG in outputs/
"""

import os
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

from models.player import Player

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROFILE_ORDER  = ["casual", "skilled", "cheater"]
PROFILE_COLORS = {
    "casual":  "#4C9BE8",   # blue
    "skilled": "#F5A623",   # orange
    "cheater": "#E84C4C",   # red
}

DPI = 150   # resolution for saved PNGs


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  ✓ Saved → {path}")


# ---------------------------------------------------------------------------
# Chart 1 — Score distribution (violin + boxplot)
# ---------------------------------------------------------------------------

def plot_score_distribution(results: Dict) -> None:
    """
    Violin plots overlaid with box plots showing the distribution of
    suspicion scores for each player profile.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    data_by_profile = {
        profile: results["profile_stats"][profile]["scores"]
        for profile in PROFILE_ORDER
    }

    positions = range(len(PROFILE_ORDER))

    # Violin plots
    parts = ax.violinplot(
        [data_by_profile[p] for p in PROFILE_ORDER],
        positions=list(positions),
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PROFILE_COLORS[PROFILE_ORDER[i]])
        pc.set_alpha(0.55)

    # Box plots
    bp = ax.boxplot(
        [data_by_profile[p] for p in PROFILE_ORDER],
        positions=list(positions),
        widths=0.12,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PROFILE_COLORS[PROFILE_ORDER[i]])
        patch.set_alpha(0.85)

    # Threshold line
    threshold = results["threshold"]
    ax.axhline(threshold, color="#333333", linestyle="--", linewidth=1.2,
               label=f"Suspicion threshold ({threshold:.0f})")

    ax.set_xticks(list(positions))
    ax.set_xticklabels([p.capitalize() for p in PROFILE_ORDER], fontsize=12)
    ax.set_ylabel("Suspicion Score [0–100]", fontsize=11)
    ax.set_title("Suspicion Score Distribution by Player Profile", fontsize=13, fontweight="bold")
    ax.set_ylim(-5, 108)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()
    _save(fig, "score_distribution.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2 — Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(results: Dict) -> None:
    """Heatmap of the 3×3 confusion matrix."""
    labels = results["confusion_matrix"]["labels"]
    matrix = np.array(results["confusion_matrix"]["matrix"])

    # Normalise each row to percentages for readability
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_pct = np.where(row_sums > 0, matrix / row_sums * 100, 0)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.heatmap(
        matrix_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=[l.capitalize() for l in labels],
        yticklabels=[l.capitalize() for l in labels],
        linewidths=0.5,
        linecolor="#cccccc",
        ax=ax,
        cbar_kws={"label": "% of true class"},
    )

    # Overlay raw counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j + 0.5, i + 0.72,
                f"(n={matrix[i, j]})",
                ha="center", va="center",
                fontsize=8, color="#444444",
            )

    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalised %)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.show()
    _save(fig, "confusion_matrix.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3 — ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(players: List[Player], results: Dict) -> None:
    """
    ROC curve for the LSTM probability on the binary classification task
    (cheater vs. non-cheater).
    """
    y_true = [1 if p.profile == "cheater" else 0 for p in players]
    y_prob = [p.lstm_prob for p in players]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = results["classification"]["auc_roc"]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color="#E84C4C", linewidth=2,
            label=f"LSTM (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--",
            linewidth=1, label="Random classifier")

    ax.fill_between(fpr, tpr, alpha=0.10, color="#E84C4C")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — LSTM Cheat Probability\n(Cheater vs. Non-Cheater)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()
    _save(fig, "roc_curve.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4 — Flag frequency per profile
# ---------------------------------------------------------------------------

def plot_flag_frequency(players: List[Player]) -> None:
    """
    Grouped bar chart showing how often each detection flag was triggered
    per profile, as a percentage of players in that profile.
    """
    from controllers.heuristic_scorer import FLAGS

    flag_ids = [f[0] for f in FLAGS]

    # Count flag triggers per profile
    counts = {
        profile: {fid: 0 for fid in flag_ids}
        for profile in PROFILE_ORDER
    }
    totals = {profile: 0 for profile in PROFILE_ORDER}

    for p in players:
        totals[p.profile] += 1
        for fid in p.flags:
            if fid in counts[p.profile]:
                counts[p.profile][fid] += 1

    # Convert to percentages
    pct = {
        profile: [
            100.0 * counts[profile][fid] / max(1, totals[profile])
            for fid in flag_ids
        ]
        for profile in PROFILE_ORDER
    }

    x      = np.arange(len(flag_ids))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, profile in enumerate(PROFILE_ORDER):
        ax.bar(
            x + i * width,
            pct[profile],
            width,
            label=profile.capitalize(),
            color=PROFILE_COLORS[profile],
            alpha=0.85,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [fid.replace("_", "\n") for fid in flag_ids],
        fontsize=8,
    )
    ax.set_ylabel("% of players in profile", fontsize=11)
    ax.set_title("Detection Flag Trigger Rate by Profile", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()
    _save(fig, "flag_frequency.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_charts(players: List[Player], results: Dict) -> None:
    """Generate and save all benchmark visualisations."""
    print("[visualizer] Generating charts...")
    matplotlib.use("Agg")   # non-interactive backend for saving; remove for live display
    plot_score_distribution(results)
    plot_confusion_matrix(results)
    plot_roc_curve(players, results)
    plot_flag_frequency(players)
    print(f"  ✓ All charts saved to outputs/\n")