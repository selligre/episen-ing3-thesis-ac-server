"""
controllers/heuristic_scorer.py
--------------------------------
Rule-based suspicion scoring engine.

Each player's telemetry is evaluated against a set of weighted flags.
Flags fire when a per-session metric crosses a threshold.  The final
heuristic score reflects *how often* and *how strongly* each flag fires
across the player's sessions, rather than a simple binary trigger.

This design deliberately introduces realistic overlap between the
'skilled' and 'cheater' profiles — a key point discussed in thesis
Section 3.4.2 (High-Skill Bias): some cheater metrics are individually
indistinguishable from exceptional legitimate play.  Only the *joint*
presence of multiple flags, combined with cross-session consistency,
drives the score above the suspicion threshold.
"""

import numpy as np
from typing import List, Tuple, Dict
from models.player import Player, Session

# ---------------------------------------------------------------------------
# Flag definitions
# Format: (flag_id, description, base_weight)
# ---------------------------------------------------------------------------

FLAGS = [
    ("TTD_CRITICAL",      "Mean TTD < 80 ms across session",             25),
    ("PRE_AIM_LOCK",      "Mean pre-aim delta < 3° consistently",        20),
    ("HS_RATE_EXTREME",   "HS rate > 65% over sufficient kills",         20),
    ("SPRAY_PERFECT",     "Mean spray deviation < 2.5",                  15),
    ("REACTION_INHUMAN",  "Mean reaction time < 60 ms",                  20),
    ("KD_ADR_COMBO",      "KD > 2.5 AND ADR > 100 combined",            15),
    ("CLUTCH_INHUMAN",    "Clutch win rate > 70%",                       10),
    ("CONSISTENCY_BONUS", "≥4 flags firing in ≥80% of sessions",        20),
]

FLAG_WEIGHTS  = {f[0]: f[2] for f in FLAGS}
# Maximum achievable raw score (all flags + consistency bonus)
MAX_RAW_SCORE = sum(f[2] for f in FLAGS)


# ---------------------------------------------------------------------------
# Per-session helpers
# ---------------------------------------------------------------------------

def _kills(session: Session):
    return [k for r in session.rounds for k in r.kill_events]


def _avg(values):
    return sum(values) / len(values) if values else None


def _session_avg_ttd(session: Session):
    vals = [k.ttd_ms for k in _kills(session)]
    return _avg(vals)


def _session_avg_pre_aim(session: Session):
    vals = [k.pre_aim_delta_deg for k in _kills(session)]
    return _avg(vals)


def _session_avg_spray(session: Session):
    vals = [k.spray_deviation for k in _kills(session)]
    return _avg(vals)


def _session_avg_reaction(session: Session):
    vals = [k.reaction_time_ms for k in _kills(session)]
    return _avg(vals)


def _session_n_kills(session: Session) -> int:
    return sum(len(r.kill_events) for r in session.rounds)


# ---------------------------------------------------------------------------
# Per-session flag evaluation — returns a dict of flag_id → intensity [0,1]
# Intensity > 0 means the flag fired; intensity = 1 means the metric is
# deep in the cheater range.  This gives us a continuous signal rather
# than a binary trigger.
# ---------------------------------------------------------------------------

def _evaluate_session_flags(session: Session) -> Dict[str, float]:
    """
    Returns {flag_id: intensity} for each flag that fired this session.
    Intensity is a continuous value in (0, 1] reflecting how far the
    metric is beyond the flag threshold into the cheater range.
    """
    fired: Dict[str, float] = {}

    avg_ttd      = _session_avg_ttd(session)
    avg_pre_aim  = _session_avg_pre_aim(session)
    avg_spray    = _session_avg_spray(session)
    avg_reaction = _session_avg_reaction(session)
    n_kills      = _session_n_kills(session)

    # TTD: threshold=80ms, cheater centre=40ms  → intensity ramps 80→5ms
    if avg_ttd is not None and avg_ttd < 80.0:
        fired["TTD_CRITICAL"] = np.clip((80.0 - avg_ttd) / 75.0, 0, 1)

    # Pre-aim: threshold=3°, cheater centre=1.5°
    if avg_pre_aim is not None and avg_pre_aim < 3.0:
        fired["PRE_AIM_LOCK"] = np.clip((3.0 - avg_pre_aim) / 2.9, 0, 1)

    # HS rate: threshold=0.65, need ≥5 kills
    if n_kills >= 5 and session.hs_rate > 0.65:
        fired["HS_RATE_EXTREME"] = np.clip((session.hs_rate - 0.65) / 0.35, 0, 1)

    # Spray: threshold=2.5, cheater centre=1.2
    if avg_spray is not None and avg_spray < 2.5:
        fired["SPRAY_PERFECT"] = np.clip((2.5 - avg_spray) / 2.4, 0, 1)

    # Reaction: threshold=60ms, cheater centre=35ms
    if avg_reaction is not None and avg_reaction < 60.0:
        fired["REACTION_INHUMAN"] = np.clip((60.0 - avg_reaction) / 55.0, 0, 1)

    # KD+ADR combo
    if session.kd_ratio > 2.5 and session.adr > 100.0:
        intensity = np.clip(
            ((session.kd_ratio - 2.5) / 3.5 + (session.adr - 100.0) / 60.0) / 2.0,
            0, 1
        )
        fired["KD_ADR_COMBO"] = intensity

    # Clutch
    if session.clutch_win_rate > 0.70:
        fired["CLUTCH_INHUMAN"] = np.clip((session.clutch_win_rate - 0.70) / 0.30, 0, 1)

    return fired


# ---------------------------------------------------------------------------
# Player-level scoring
# ---------------------------------------------------------------------------

def score_player(player: Player) -> Player:
    """
    Compute the heuristic suspicion score for a player.

    Algorithm
    ---------
    1. For each session, compute per-flag intensities.
    2. For each flag, aggregate across sessions:
       - fire_rate  = fraction of sessions where the flag triggered
       - avg_intensity = mean intensity across sessions where it fired
    3. A flag contributes to the score if fire_rate ≥ 0.3
       (fires in at least 30% of sessions).
    4. Contribution = weight × fire_rate × avg_intensity
       This gives continuous variation: a cheater who always fires a
       flag at high intensity scores much more than a skilled player
       who occasionally crosses a threshold with low intensity.
    5. Consistency bonus added if ≥4 flags fire in ≥80% of sessions.
    6. Raw score normalised to [0, 100].
    """
    n_sessions = len(player.sessions)
    if n_sessions == 0:
        return player

    # Collect per-session flag intensities
    # structure: {flag_id: [intensity_s1, intensity_s2, ...]}
    flag_intensities: Dict[str, List[float]] = {f[0]: [] for f in FLAGS if f[0] != "CONSISTENCY_BONUS"}

    for session in player.sessions:
        fired = _evaluate_session_flags(session)
        for flag_id in flag_intensities:
            if flag_id in fired:
                flag_intensities[flag_id].append(fired[flag_id])
            else:
                flag_intensities[flag_id].append(0.0)   # did not fire

    # Compute fire rate and average intensity for each flag
    raw_score = 0.0
    active_flags = []
    high_consistency_flags = []

    for flag_id, intensities in flag_intensities.items():
        fire_rate     = sum(1 for v in intensities if v > 0) / n_sessions
        active_vals   = [v for v in intensities if v > 0]
        avg_intensity = np.mean(active_vals) if active_vals else 0.0

        if fire_rate >= 0.30:
            contribution = FLAG_WEIGHTS[flag_id] * fire_rate * avg_intensity
            raw_score   += contribution
            active_flags.append(flag_id)

        if fire_rate >= 0.80:
            high_consistency_flags.append(flag_id)

    # Consistency bonus
    if len(high_consistency_flags) >= 4:
        active_flags.append("CONSISTENCY_BONUS")
        raw_score += FLAG_WEIGHTS["CONSISTENCY_BONUS"] * 0.8

    # Normalise to [0, 100]
    # Denominator chosen so that a perfect cheater (all flags, full intensity,
    # all sessions) scores ~95-98, not exactly 100 — preserves realism
    normalised = min(100.0, (raw_score / (MAX_RAW_SCORE * 0.85)) * 100.0)

    player.heuristic_score = round(float(normalised), 2)
    player.flags           = active_flags

    return player


def score_all_players(players: List[Player]) -> List[Player]:
    """Score all players and return the updated list."""
    print("[heuristic_scorer] Scoring players...")
    for player in players:
        score_player(player)
    print(f"  ✓ {len(players)} players scored\n")
    return players