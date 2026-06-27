"""
models/player.py
----------------
Pure data structures representing the telemetry domain.

Hierarchy:
    Player
      └── Session  (one match)
            └── Round
                  └── KillEvent  (one kill inside a round)
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Kill-level granularity
# ---------------------------------------------------------------------------

@dataclass
class KillEvent:
    """
    Represents a single kill performed by a player during a round.

    All timing values are in milliseconds.
    Angles are in degrees.
    """
    ttd_ms: float           # Time-To-Damage: ms between first enemy visibility and first hit
    pre_aim_delta_deg: float # Angular gap between crosshair and enemy head BEFORE enemy appeared
    spray_deviation: float   # Std-dev from the theoretical recoil pattern (lower = more perfect)
    reaction_time_ms: float  # ms between enemy entering FOV and weapon fire
    hit_zone: str            # "head" | "chest" | "stomach" | "legs"


# ---------------------------------------------------------------------------
# Round-level granularity
# ---------------------------------------------------------------------------

@dataclass
class Round:
    """
    Aggregated stats for a single round, plus the individual kill events.
    """
    kill_events: List[KillEvent] = field(default_factory=list)

    # Round-level aggregates
    utility_dmg: float = 0.0       # HE grenade damage dealt this round
    clutch_result: bool = False     # Won a clutch (1vN) situation this round
    opening_duel: bool = False      # Player initiated the first contact of the round


# ---------------------------------------------------------------------------
# Session-level granularity  (one full match)
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """
    A full match session consisting of multiple rounds.
    """
    rounds: List[Round] = field(default_factory=list)

    # Cross-round session aggregates (computed after generation)
    hs_rate: float = 0.0            # Headshot rate across all kills in the session
    kd_ratio: float = 0.0           # Kill / Death ratio
    adr: float = 0.0                # Average Damage per Round
    clutch_win_rate: float = 0.0    # Fraction of clutch situations won
    utility_efficiency: float = 0.0 # Mean HE damage per round


# ---------------------------------------------------------------------------
# Player-level granularity
# ---------------------------------------------------------------------------

@dataclass
class Player:
    """
    Represents a simulated player with multiple match sessions.

    Attributes
    ----------
    player_id : str
        Unique identifier (e.g. "casual_042").
    profile : str
        Ground-truth label: "casual" | "skilled" | "cheater".
    sessions : List[Session]
        All match sessions recorded for this player.
    suspicion_score : float
        Final suspicion score [0–100] assigned by the scoring pipeline.
    heuristic_score : float
        Score from the rule-based heuristic component alone.
    lstm_prob : float
        Cheat probability [0–1] produced by the LSTM component.
    flags : List[str]
        Human-readable list of triggered detection flags.
    """
    player_id: str
    profile: str                        # ground-truth label
    sessions: List[Session] = field(default_factory=list)

    # Scoring outputs (filled by the pipeline)
    suspicion_score: float = 0.0
    heuristic_score: float = 0.0
    lstm_prob: float = 0.0
    flags: List[str] = field(default_factory=list)