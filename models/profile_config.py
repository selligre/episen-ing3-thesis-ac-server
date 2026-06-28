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
    # CASUAL — Dégradé vers le bas, très grande variance pour couvrir 
    # du débutant absolu au joueur occasionnel.
    # -----------------------------------------------------------------------
    "casual": {
        "ttd_ms":            (750.0,  250.0,  350.0, 1500.0),
        "pre_aim_delta_deg": (40.0,   20.0,   15.0,  100.0),
        "spray_deviation":   (30.0,   15.0,    8.0,   70.0),
        "reaction_time_ms":  (500.0,  200.0,  300.0, 1100.0),
        "utility_dmg":       (8.0,     6.0,    0.0,   30.0),
        "clutch_win_rate":   (0.08,    0.05,   0.0,    0.3),
        "kd_ratio":          (0.50,    0.25,   0.1,    1.0),
        "adr":               (35.0,   12.0,    8.0,   60.0),
        "hs_rate":           (0.10,    0.05,   0.0,    0.25),
    },

    # -----------------------------------------------------------------------
    # SKILLED — Ajusté à la baisse pour rester humain et réaliste.
    # Écart-type augmenté pour simuler des journées "sans" et "avec".
    # -----------------------------------------------------------------------
    "skilled": {
        "ttd_ms":            (260.0,  100.0,  140.0,  600.0),
        "pre_aim_delta_deg": (10.0,    5.0,    3.0,   30.0),
        "spray_deviation":   (8.0,     4.0,    2.0,   20.0),
        "reaction_time_ms":  (230.0,   70.0,  150.0,  450.0),
        "utility_dmg":       (35.0,   18.0,    5.0,   90.0),
        "clutch_win_rate":   (0.30,    0.15,   0.1,    0.6),
        "kd_ratio":          (1.05,    0.45,   0.5,    2.2),
        "adr":               (70.0,   25.0,   40.0,  110.0),
        "hs_rate":           (0.35,    0.12,   0.15,   0.60),
    },

    # -----------------------------------------------------------------------
    # CHEATER — Moyennes mécaniques élevées, mais bornes basses abaissées
    # pour permettre des recouvrements occasionnels avec les meilleurs Skilled.
    # -----------------------------------------------------------------------
    "cheater": {
        "ttd_ms":            (180.0,   50.0,  120.0,  350.0),
        "pre_aim_delta_deg": (6.0,     2.5,    2.0,   15.0),
        "spray_deviation":   (4.5,     1.8,    1.5,   10.0),
        "reaction_time_ms":  (170.0,   40.0,  120.0,  300.0),
        "utility_dmg":       (25.0,   12.0,    0.0,   60.0),
        "clutch_win_rate":   (0.45,    0.12,   0.2,    0.8),
        "kd_ratio":          (1.60,    0.50,   0.9,    3.0),
        "adr":               (85.0,   18.0,   50.0,  130.0),
        "hs_rate":           (0.55,    0.10,   0.35,   0.80),
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