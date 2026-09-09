"""Named ablation arms and their explicit removed component."""

ABLATION_ARMS = {
    "full_hybrid": "Physics rollout plus learned residual, optimization, and migration",
    "physics_mpc": "No learned residual correction",
    "learned_only": "No physical rollout in the controller predictor",
    "no_optimization": "Forecast-triggered deterministic rule without search",
    "no_migration": "Full hybrid search with migration candidates removed",
}

