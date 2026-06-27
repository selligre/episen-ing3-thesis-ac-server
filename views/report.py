"""
views/report.py
----------------
Console report of benchmark results.
Formats the results dict produced by controllers/benchmark.py.
"""

from typing import Dict

PROFILE_ORDER = ["casual", "skilled", "cheater"]
DIVIDER = "─" * 60


def print_report(results: Dict) -> None:
    """Print a structured benchmark report to stdout."""

    threshold = results["threshold"]

    print(f"\n{'═' * 60}")
    print("  CS2 SERVER-SIDE ANTI-CHEAT — BENCHMARK REPORT")
    print(f"{'═' * 60}")
    print(f"  Suspicion threshold: {threshold:.0f} / 100\n")

    # --- Per-profile stats ---
    print("  PER-PROFILE SUSPICION SCORE STATISTICS")
    print(f"  {DIVIDER}")
    header = f"  {'Profile':<12} {'N':>5}  {'Mean':>7}  {'Std':>7}  {'Median':>8}  {'% Flagged':>10}"
    print(header)
    print(f"  {DIVIDER}")

    for profile in PROFILE_ORDER:
        s = results["profile_stats"][profile]
        print(
            f"  {profile.capitalize():<12} {s['n']:>5}  "
            f"{s['mean']:>7.2f}  {s['std']:>7.2f}  "
            f"{s['median']:>8.2f}  {s['pct_flagged']:>9.1f}%"
        )

    print(f"  {DIVIDER}")

    # --- Classification metrics ---
    c = results["classification"]
    print(f"\n  CLASSIFICATION METRICS  (cheater vs. non-cheater, binary)")
    print(f"  {DIVIDER}")
    print(f"  {'Precision':<20} {c['precision']:.4f}")
    print(f"  {'Recall':<20} {c['recall']:.4f}")
    print(f"  {'F1-Score':<20} {c['f1_score']:.4f}")
    print(f"  {'AUC-ROC (LSTM prob)':<20} {c['auc_roc']:.4f}")
    print(f"  {DIVIDER}")

    # --- Confusion matrix ---
    print(f"\n  CONFUSION MATRIX  (rows = true, cols = predicted)")
    labels = results["confusion_matrix"]["labels"]
    matrix = results["confusion_matrix"]["matrix"]

    col_w = 12
    header_row = " " * 18 + "".join(f"{l.capitalize():>{col_w}}" for l in labels)
    print(f"  {header_row}")
    for i, label in enumerate(labels):
        row_str = "".join(f"{matrix[i][j]:>{col_w}}" for j in range(len(labels)))
        print(f"  {label.capitalize():<18}{row_str}")

    # --- False positives ---
    fp = results["false_positives"]
    print(f"\n  FALSE POSITIVE ANALYSIS")
    print(f"  {DIVIDER}")
    print(f"  Skilled players flagged as cheaters : {fp['skilled_flagged_as_cheater']}")
    print(f"  Casual  players flagged as cheaters : {fp['casual_flagged_as_cheater']}")
    print(f"  {DIVIDER}")
    print(f"\n{'═' * 60}\n")