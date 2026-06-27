"""
main.py
--------
Entry point for the CS2 server-side anti-cheat analytics pipeline.

Pipeline steps
--------------
1. GENERATE   — Simulate telemetry for 300 players (100 per profile)
2. HEURISTIC  — Rule-based suspicion scoring
3. LSTM       — Sequential model scoring + memory persistence
4. PERSIST    — Write scoring results back to player JSON files
5. BENCHMARK  — Compute classification metrics
6. REPORT     — Print results to console
7. VISUALIZE  — Generate and save all charts

Usage
-----
    python main.py              # full pipeline (re-generates data)
    python main.py --no-regen   # skip generation, reload existing player files
"""

import argparse
import sys
import os

# Make sure imports resolve from project root
sys.path.insert(0, os.path.dirname(__file__))

from controllers.generator       import generate_all_players, load_all_players
from controllers.heuristic_scorer import score_all_players
from controllers.lstm_scorer      import lstm_score_all_players
from controllers.persistence      import save_scored_players
from controllers.benchmark        import run_benchmark
from views.report                 import print_report
from views.visualizer             import generate_all_charts


def main():
    parser = argparse.ArgumentParser(
        description="CS2 Server-Side Anti-Cheat Analytics Pipeline"
    )
    parser.add_argument(
        "--no-regen",
        action="store_true",
        help="Skip data generation and reload existing player JSON files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for telemetry generation (default: 42)",
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  CS2 Server-Side Anti-Cheat Analytics Pipeline")
    print("  Thesis — Gilles Meunier — UPEC / EPISEN / Saint-Gobain GDII")
    print("═" * 60 + "\n")

    # ------------------------------------------------------------------
    # Step 1 — Data generation (or reload)
    # ------------------------------------------------------------------
    if args.no_regen:
        print("[main] Loading existing player data...")
        players = load_all_players()
        print(f"  ✓ {len(players)} players loaded\n")
    else:
        players = generate_all_players(seed=args.seed)

    # ------------------------------------------------------------------
    # Step 2 — Heuristic scoring
    # ------------------------------------------------------------------
    players = score_all_players(players)

    # ------------------------------------------------------------------
    # Step 3 — LSTM scoring + final score combination
    # ------------------------------------------------------------------
    players = lstm_score_all_players(players)

    # ------------------------------------------------------------------
    # Step 4 — Persist scores to JSON
    # ------------------------------------------------------------------
    save_scored_players(players)

    # ------------------------------------------------------------------
    # Step 5 — Benchmark
    # ------------------------------------------------------------------
    results = run_benchmark(players)

    # ------------------------------------------------------------------
    # Step 6 — Console report
    # ------------------------------------------------------------------
    print_report(results)

    # ------------------------------------------------------------------
    # Step 7 — Visualisations
    # ------------------------------------------------------------------
    generate_all_charts(players, results)

    print("[main] Pipeline complete.\n")


if __name__ == "__main__":
    main()