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
    # CASUAL — Écart-type augmenté. Parfois bons, souvent moyens.
    # -----------------------------------------------------------------------
    "casual": {
        "ttd_ms":            (450.0,  120.0,  150.0,  900.0),
        "pre_aim_delta_deg": (22.0,    8.0,    4.0,   45.0),
        "spray_deviation":   (18.0,    5.0,    4.0,   35.0),
        "reaction_time_ms":  (350.0,   90.0,  120.0,  700.0),
        "utility_dmg":       (18.0,   12.0,    0.0,   80.0),
        "clutch_win_rate":   (0.15,    0.10,   0.0,    0.6),
        "kd_ratio":          (0.85,    0.25,   0.1,    2.0),
        "adr":               (60.0,   20.0,   15.0,  110.0),
        "hs_rate":           (0.30,    0.10,   0.05,   0.60),
    },

    # -----------------------------------------------------------------------
    # SKILLED — Excellents, mais avec la faillibilité humaine (std large).
    # Certains frôleront les statistiques des tricheurs les bons jours.
    # -----------------------------------------------------------------------
    "skilled": {
        "ttd_ms":            (210.0,   60.0,   80.0,  400.0), 
        "pre_aim_delta_deg": (11.0,    4.5,    1.5,   25.0),
        "spray_deviation":   (8.0,     3.5,    1.5,   18.0),
        "reaction_time_ms":  (190.0,   50.0,   80.0,  380.0),
        "utility_dmg":       (35.0,   15.0,    0.0,   90.0),
        "clutch_win_rate":   (0.40,    0.15,   0.1,    0.8),
        "kd_ratio":          (1.45,    0.35,   0.5,    3.0),
        "adr":               (85.0,   20.0,   45.0,  140.0),
        "hs_rate":           (0.45,    0.09,   0.20,   0.75),
    },

    # -----------------------------------------------------------------------
    # CHEATER — "Closet Cheater". Métriques très proches des Skilled,
    # mais avec une constance mécanique effrayante (std extrêmement faible).
    # -----------------------------------------------------------------------
    "cheater": {
        "ttd_ms":            (110.0,   15.0,   40.0,  180.0), # Très régulier
        "pre_aim_delta_deg": (6.0,     1.0,    1.0,   10.0),  # Aim-assist lissé
        "spray_deviation":   (4.0,     0.8,    1.0,   8.0),
        "reaction_time_ms":  (110.0,   15.0,   40.0,  180.0),
        "utility_dmg":       (25.0,   10.0,    0.0,   70.0),
        "clutch_win_rate":   (0.60,    0.05,   0.4,    1.0),
        "kd_ratio":          (2.20,    0.20,   1.5,    4.0),
        "adr":               (100.0,  10.0,   70.0,  140.0),
        "hs_rate":           (0.60,    0.04,   0.45,   0.85),
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