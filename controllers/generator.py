"""
controllers/generator.py
-------------------------
Generates synthetic player telemetry based on parametric distributions
defined in models/profile_config.py.

Each player is serialised to an individual JSON file under data/players/.
A summary index (data/players/index.json) maps player IDs to their profiles
for fast loading by downstream pipeline steps.

Design notes
------------
- numpy is used for all random sampling (reproducible via seed).
- Each KillEvent, Round, Session and Player is built bottom-up.
- Session aggregates (hs_rate, kd_ratio, adr, clutch_win_rate,
  utility_efficiency) are computed after generating all rounds so that
  they reflect the true distribution of events rather than being sampled
  independently — this preserves realistic correlations.
"""

import json
import os
import numpy as np
from typing import List

from models.player import Player, Session, Round, KillEvent
from models.profile_config import (
    PROFILE_DISTRIBUTIONS,
    HIT_ZONE_WEIGHTS,
    HIT_ZONES,
    SIMULATION_CONFIG,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "players")


# ---------------------------------------------------------------------------
# Low-level samplers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sample(rng: np.random.Generator, dist: tuple) -> float:
    """Sample one value from a Normal distribution and clamp it."""
    mean, std, lo, hi = dist
    return _clamp(float(rng.normal(mean, std)), lo, hi)


def _sample_hit_zone(rng: np.random.Generator, profile: str) -> str:
    weights = HIT_ZONE_WEIGHTS[profile]
    return rng.choice(HIT_ZONES, p=weights)


# ---------------------------------------------------------------------------
# Kill event generation
# ---------------------------------------------------------------------------

def _generate_kill_event(rng: np.random.Generator, profile: str, dist: dict) -> KillEvent:
    return KillEvent(
        ttd_ms=_sample(rng, dist["ttd_ms"]),
        pre_aim_delta_deg=_sample(rng, dist["pre_aim_delta_deg"]),
        spray_deviation=_sample(rng, dist["spray_deviation"]),
        reaction_time_ms=_sample(rng, dist["reaction_time_ms"]),
        hit_zone=_sample_hit_zone(rng, profile),
    )


# ---------------------------------------------------------------------------
# Round generation
# ---------------------------------------------------------------------------

def _generate_round(rng: np.random.Generator, profile: str, dist: dict) -> Round:
    cfg  = SIMULATION_CONFIG

    # Number of kills this round follows a Poisson distribution
    n_kills = int(rng.poisson(cfg["kills_per_round_mean"]))

    kill_events = [_generate_kill_event(rng, profile, dist) for _ in range(n_kills)]

    return Round(
        kill_events=kill_events,
        utility_dmg=_sample(rng, dist["utility_dmg"]),
        # clutch and opening duel are Bernoulli-sampled from player means
        clutch_result=bool(rng.random() < dist["clutch_win_rate"][0]),
        opening_duel=bool(rng.random() < 0.30),   # uniform across profiles
    )


# ---------------------------------------------------------------------------
# Session generation
# ---------------------------------------------------------------------------

def _generate_session(rng: np.random.Generator, profile: str, dist: dict) -> Session:
    cfg = SIMULATION_CONFIG
    rounds = [_generate_round(rng, profile, dist) for _ in range(cfg["rounds_per_session"])]

    # --- Compute session-level aggregates from generated rounds ---
    all_kills = [k for r in rounds for k in r.kill_events]
    n_kills   = len(all_kills)
    n_rounds  = len(rounds)

    hs_count   = sum(1 for k in all_kills if k.hit_zone == "head")
    hs_rate    = hs_count / n_kills if n_kills > 0 else 0.0

    kd_ratio = _sample(rng, dist["kd_ratio"])
    adr      = _sample(rng, dist["adr"])

    clutch_rounds = [r for r in rounds if r.opening_duel]   # proxy for clutch situations
    clutch_wins   = sum(1 for r in clutch_rounds if r.clutch_result)
    clutch_win_rate = clutch_wins / len(clutch_rounds) if clutch_rounds else 0.0

    utility_efficiency = sum(r.utility_dmg for r in rounds) / n_rounds

    session = Session(rounds=rounds)
    session.hs_rate            = round(hs_rate, 4)
    session.kd_ratio           = round(kd_ratio, 3)
    session.adr                = round(adr, 2)
    session.clutch_win_rate    = round(clutch_win_rate, 4)
    session.utility_efficiency = round(utility_efficiency, 2)

    return session


# ---------------------------------------------------------------------------
# Player generation
# ---------------------------------------------------------------------------

def _generate_player(player_id: str, profile: str, seed: int) -> Player:
    rng = np.random.default_rng(seed)
    cfg = SIMULATION_CONFIG

    # ponytail: generate player-specific mean offsets to create a natural spread between individual players
    # This prevents all players of the same profile from having identical session averages.
    player_dist = {}
    for feat, dist_tuple in PROFILE_DISTRIBUTIONS[profile].items():
        mean, std, lo, hi = dist_tuple
        # We sample a player-specific mean around the profile mean using std / 2 as the variance.
        p_mean = float(rng.normal(mean, std / 2.0))
        p_mean = max(lo, min(hi, p_mean))
        player_dist[feat] = (p_mean, std, lo, hi)

    sessions = [
        _generate_session(rng, profile, player_dist)
        for _ in range(cfg["sessions_per_player"])
    ]

    return Player(player_id=player_id, profile=profile, sessions=sessions)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _player_to_dict(player: Player) -> dict:
    """Convert a Player dataclass tree to a plain dict for JSON serialisation."""
    return {
        "player_id": player.player_id,
        "profile":   player.profile,
        "suspicion_score": player.suspicion_score,
        "heuristic_score": player.heuristic_score,
        "lstm_prob":       player.lstm_prob,
        "flags":           player.flags,
        "sessions": [
            {
                "hs_rate":            s.hs_rate,
                "kd_ratio":           s.kd_ratio,
                "adr":                s.adr,
                "clutch_win_rate":    s.clutch_win_rate,
                "utility_efficiency": s.utility_efficiency,
                "rounds": [
                    {
                        "utility_dmg":   r.utility_dmg,
                        "clutch_result": r.clutch_result,
                        "opening_duel":  r.opening_duel,
                        "kill_events": [
                            {
                                "ttd_ms":            k.ttd_ms,
                                "pre_aim_delta_deg": k.pre_aim_delta_deg,
                                "spray_deviation":   k.spray_deviation,
                                "reaction_time_ms":  k.reaction_time_ms,
                                "hit_zone":          k.hit_zone,
                            }
                            for k in r.kill_events
                        ],
                    }
                    for r in s.rounds
                ],
            }
            for s in player.sessions
        ],
    }


def _save_player(player: Player) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{player.player_id}.json")
    with open(path, "w") as f:
        json.dump(_player_to_dict(player), f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_players(seed: int = 42) -> List[Player]:
    """
    Generate all synthetic players for the three profiles and write each
    one to its own JSON file.  Returns the list of Player objects for
    immediate use by the scoring pipeline.

    Parameters
    ----------
    seed : int
        Base random seed.  Each player gets seed + index to ensure
        reproducibility while maintaining independence.

    Returns
    -------
    List[Player]
        300 Player objects (100 per profile).
    """
    cfg = SIMULATION_CONFIG
    profiles = ["casual", "skilled", "cheater"]
    players: List[Player] = []

    print("[generator] Starting synthetic telemetry generation...")
    index = {}

    global_idx = 0
    for profile in profiles:
        for i in range(cfg["players_per_profile"]):
            player_id = f"{profile}_{i:03d}"
            player    = _generate_player(player_id, profile, seed=seed + global_idx)
            _save_player(player)
            players.append(player)
            index[player_id] = profile
            global_idx += 1

        print(f"  ✓ {cfg['players_per_profile']} {profile} players generated")

    # Write the index file
    index_path = os.path.join(DATA_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[generator] Done — {len(players)} players written to {DATA_DIR}/\n")
    return players


def load_all_players() -> List[Player]:
    """
    Reload all Player objects from existing JSON files.
    Used to skip re-generation when only re-running scoring or benchmark.
    """
    index_path = os.path.join(DATA_DIR, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "No player data found. Run generate_all_players() first."
        )

    with open(index_path) as f:
        index = json.load(f)

    players = []
    for player_id, profile in index.items():
        path = os.path.join(DATA_DIR, f"{player_id}.json")
        with open(path) as f:
            data = json.load(f)
        player = _dict_to_player(data)
        players.append(player)

    return players


def _dict_to_player(data: dict) -> Player:
    """Reconstruct a Player dataclass tree from a JSON dict."""
    sessions = []
    for s in data["sessions"]:
        rounds = []
        for r in s["rounds"]:
            kill_events = [
                KillEvent(
                    ttd_ms=k["ttd_ms"],
                    pre_aim_delta_deg=k["pre_aim_delta_deg"],
                    spray_deviation=k["spray_deviation"],
                    reaction_time_ms=k["reaction_time_ms"],
                    hit_zone=k["hit_zone"],
                )
                for k in r["kill_events"]
            ]
            rounds.append(Round(
                kill_events=kill_events,
                utility_dmg=r["utility_dmg"],
                clutch_result=r["clutch_result"],
                opening_duel=r["opening_duel"],
            ))
        session = Session(rounds=rounds)
        session.hs_rate            = s["hs_rate"]
        session.kd_ratio           = s["kd_ratio"]
        session.adr                = s["adr"]
        session.clutch_win_rate    = s["clutch_win_rate"]
        session.utility_efficiency = s["utility_efficiency"]
        sessions.append(session)

    player = Player(player_id=data["player_id"], profile=data["profile"], sessions=sessions)
    player.suspicion_score = data.get("suspicion_score", 0.0)
    player.heuristic_score = data.get("heuristic_score", 0.0)
    player.lstm_prob       = data.get("lstm_prob", 0.0)
    player.flags           = data.get("flags", [])
    return player