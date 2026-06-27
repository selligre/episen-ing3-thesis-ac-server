"""
controllers/persistence.py
---------------------------
Saves scored player objects back to their individual JSON files
so that scoring results are persisted alongside the raw telemetry.
"""

import json
import os
from typing import List

from models.player import Player

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "players")


def save_scored_players(players: List[Player]) -> None:
    """
    Overwrite each player's JSON file with the current state of the
    Player object, including suspicion_score, heuristic_score,
    lstm_prob, and flags.
    """
    for player in players:
        path = os.path.join(DATA_DIR, f"{player.player_id}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        data["suspicion_score"] = player.suspicion_score
        data["heuristic_score"] = player.heuristic_score
        data["lstm_prob"]       = player.lstm_prob
        data["flags"]           = player.flags

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"[persistence] {len(players)} player files updated with scoring results\n")