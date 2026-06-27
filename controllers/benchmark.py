"""
controllers/benchmark.py
-------------------------
Computes classification metrics from the pipeline's suspicion scores.

Outputs
-------
- Per-profile score statistics (mean, std, median, % flagged at threshold)
- Precision, Recall, F1 on the "cheater" class
- Confusion matrix (3×3: casual / skilled / cheater)
- AUC-ROC of the LSTM probability on the binary cheater vs. non-cheater task
- A results dict saved to data/benchmark_results.json
"""

import json
import os
from typing import List, Dict

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from models.player import Player

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUSPICION_THRESHOLD = 70.0  # score above which a player is "flagged"

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmark_results.json"
)

PROFILE_ORDER = ["casual", "skilled", "cheater"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_stats(players: List[Player], profile: str) -> Dict:
    """Compute descriptive statistics for one profile."""
    scores = [p.suspicion_score for p in players if p.profile == profile]
    if not scores:
        return {}

    flagged = sum(1 for s in scores if s >= SUSPICION_THRESHOLD)
    return {
        "n":              len(scores),
        "mean":           round(float(np.mean(scores)),   2),
        "std":            round(float(np.std(scores)),    2),
        "median":         round(float(np.median(scores)), 2),
        "min":            round(float(np.min(scores)),    2),
        "max":            round(float(np.max(scores)),    2),
        "pct_flagged":    round(100.0 * flagged / len(scores), 1),
        "scores":         [round(s, 2) for s in scores],   # kept for visualisation
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(players: List[Player]) -> Dict:
    """
    Run the full benchmark suite and return a results dictionary.

    Parameters
    ----------
    players : List[Player]
        Players with suspicion_score and lstm_prob already filled.

    Returns
    -------
    Dict
        Structured benchmark results, also written to data/benchmark_results.json.
    """
    print("[benchmark] Computing metrics...")

    # --- Per-profile descriptive statistics ---
    profile_stats = {
        profile: _profile_stats(players, profile)
        for profile in PROFILE_ORDER
    }

    # --- Classification metrics (binary: cheater vs. non-cheater) ---
    y_true_binary = [1 if p.profile == "cheater" else 0 for p in players]
    y_pred_binary = [1 if p.suspicion_score >= SUSPICION_THRESHOLD else 0 for p in players]
    y_prob_lstm   = [p.lstm_prob for p in players]

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall    = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1        = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    auc       = roc_auc_score(y_true_binary, y_prob_lstm)

    # --- 3×3 confusion matrix ---
    y_true_3class = [p.profile for p in players]
    y_pred_3class = []
    for p in players:
        if p.suspicion_score >= SUSPICION_THRESHOLD:
            y_pred_3class.append("cheater")
        elif p.suspicion_score >= 20.0:
            y_pred_3class.append("skilled")
        else:
            y_pred_3class.append("casual")

    cm = confusion_matrix(y_true_3class, y_pred_3class, labels=PROFILE_ORDER)

    # --- False positive analysis ---
    fp_skilled = sum(
        1 for p in players
        if p.profile == "skilled" and p.suspicion_score >= SUSPICION_THRESHOLD
    )
    fp_casual = sum(
        1 for p in players
        if p.profile == "casual" and p.suspicion_score >= SUSPICION_THRESHOLD
    )

    results = {
        "threshold":     SUSPICION_THRESHOLD,
        "profile_stats": profile_stats,
        "classification": {
            "precision":           round(float(precision), 4),
            "recall":              round(float(recall),    4),
            "f1_score":            round(float(f1),        4),
            "auc_roc":             round(float(auc),       4),
        },
        "confusion_matrix": {
            "labels":  PROFILE_ORDER,
            "matrix":  cm.tolist(),
        },
        "false_positives": {
            "skilled_flagged_as_cheater": fp_skilled,
            "casual_flagged_as_cheater":  fp_casual,
        },
    }

    # --- Persist ---
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Benchmark complete — results saved to {RESULTS_PATH}\n")
    return results