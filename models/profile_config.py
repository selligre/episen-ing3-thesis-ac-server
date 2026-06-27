"""
models/profile_config.py
-------------------------
Parametric distributions that define each player profile.

Each entry describes a Normal distribution N(mean, std) for every
telemetry feature.  Bounds (min/max) are applied after sampling to
prevent physically impossible values (e.g. negative reaction times).

The values are calibrated from:
  - CS2 community benchmarks for human reaction times and aim metrics
  - Loo et al. (2025) AntiCheatPT feature analysis
  - Collins et al. (2024) cheat behaviour taxonomy
"""

# ---------------------------------------------------------------------------
# Feature distributions per profile
# Format: { feature_name: (mean, std, min_clamp, max_clamp) }
# ---------------------------------------------------------------------------

PROFILE_DISTRIBUTIONS = {

    # -----------------------------------------------------------------------
    # CASUAL — low-skill legitimate player
    # High TTD, large pre-aim gaps, inconsistent spray, slow reactions
    # -----------------------------------------------------------------------
    "casual": {
        "ttd_ms":            (500.0,  100.0,  150.0,  900.0),
        "pre_aim_delta_deg": (22.0,    8.0,    4.0,   45.0),
        "spray_deviation":   (18.0,    5.0,    4.0,   35.0),
        "reaction_time_ms":  (380.0,   90.0,  120.0,  700.0),

        # Round-level
        "utility_dmg":       (18.0,   12.0,    0.0,   80.0),
        "clutch_win_rate":   (0.15,    0.08,   0.0,    0.5),
        "kd_ratio":          (0.75,    0.20,   0.1,    1.8),
        "adr":               (55.0,   18.0,   15.0,  110.0),

        # Kill distribution
        "hs_rate":           (0.28,    0.08,   0.05,   0.55),
    },

    # -----------------------------------------------------------------------
    # SKILLED — high-level legitimate player
    # Fast TTD, tight pre-aim, controlled spray — but with human variance
    # -----------------------------------------------------------------------
    "skilled": {
        "ttd_ms":            (220.0,   50.0,   80.0,  400.0),
        "pre_aim_delta_deg": (10.0,    4.0,    1.5,   22.0),
        "spray_deviation":   (8.0,     3.0,    1.5,   18.0),
        "reaction_time_ms":  (200.0,   45.0,   80.0,  380.0),

        # Round-level
        "utility_dmg":       (35.0,   14.0,    0.0,   90.0),
        "clutch_win_rate":   (0.38,    0.10,   0.1,    0.7),
        "kd_ratio":          (1.40,    0.30,   0.5,    3.0),
        "adr":               (82.0,   15.0,   45.0,  130.0),

        # Kill distribution
        "hs_rate":           (0.47,    0.07,   0.25,   0.70),
    },

    # -----------------------------------------------------------------------
    # CHEATER — aimbot / triggerbot / spray script user
    # Near-zero TTD, locked pre-aim, perfect spray, inhumanly fast reactions
    # High HS rate, extreme consistency (low std relative to mean)
    # -----------------------------------------------------------------------
    "cheater": {
        "ttd_ms":            (40.0,   15.0,    5.0,   90.0),
        "pre_aim_delta_deg": (1.5,     0.8,    0.1,    4.0),
        "spray_deviation":   (1.2,     0.5,    0.1,    3.0),
        "reaction_time_ms":  (35.0,   12.0,    5.0,   75.0),

        # Round-level
        "utility_dmg":       (28.0,   10.0,    0.0,   70.0),   # cheaters focus aim, not utility
        "clutch_win_rate":   (0.72,    0.10,   0.4,    1.0),
        "kd_ratio":          (3.20,    0.60,   1.5,    6.0),
        "adr":               (115.0,  18.0,   70.0,  160.0),

        # Kill distribution
        "hs_rate":           (0.78,    0.06,   0.55,   0.98),
    },
}

# ---------------------------------------------------------------------------
# Hit zone probability weights per profile
# Cheaters lock onto head; skilled players have good but imperfect aim;
# casual players hit body more often
# ---------------------------------------------------------------------------

HIT_ZONE_WEIGHTS = {
    #              head   chest  stomach  legs
    "casual":  [0.20,  0.45,   0.25,   0.10],
    "skilled": [0.45,  0.38,   0.12,   0.05],
    "cheater": [0.82,  0.14,   0.03,   0.01],
}

HIT_ZONES = ["head", "chest", "stomach", "legs"]

# ---------------------------------------------------------------------------
# Simulation volume
# ---------------------------------------------------------------------------

SIMULATION_CONFIG = {
    "players_per_profile": 100,   # 100 casual + 100 skilled + 100 cheater = 300 total
    "sessions_per_player": 20,    # matches per player
    "rounds_per_session":  15,    # rounds per match
    "kills_per_round_mean": 1.2,  # average kills per round (Poisson-sampled)
}