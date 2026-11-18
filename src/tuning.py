"""Hyperparameter tuning with Optuna. Called by train.py and by nested CV in evaluate.py."""


def tune_pipeline(task: str, cfg: dict, X, y, groups, cv) -> dict:
    """
    Searches `cfg["tuning"]["search_spaces"][<model type>]` with Optuna and returns the best
    params as {"model__<name>": value}, ready for `pipeline.set_params(**best_params)`.

    Each trial builds `build_pipeline(task, cfg)` and scores it with `cross_validate_pipeline`
    on (X, y, groups, cv), using `cfg["tuning"]["metric"]`. Tuning stays out of build_pipeline,
    so CV folds, the final refit and nested CV never start a search by accident.

    Placeholder until tuning is implemented: returns {} (no tuning), so callers already work.
    """
    # TODO: Optuna study, sampler seeded with cfg["seed"], cfg["tuning"]["n_trials"] trials
    return {}
