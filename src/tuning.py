"""
Hyperparameter tuning. Called by train.py (final search on all rows) and by nested CV in evaluate.py.

For now a basic random search with scikit-learn; it will be replaced by Optuna. Whatever the method,
keep the signature of `tune_pipeline`, its return value and the `tuning.search_spaces` config format.
"""
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from src.models import build_pipeline

_FLOAT_POINTS = 20  # candidate values drawn for each float hyperparameter


def _candidates(name: str, spec: dict) -> list:
    """The values one hyperparameter can take, from its config spec ({type, low, high, log?} or {type, choices})."""
    if spec["type"] == "int":
        return list(range(spec["low"], spec["high"] + 1))
    if spec["type"] == "float":
        space = np.geomspace if spec.get("log", False) else np.linspace
        return list(space(spec["low"], spec["high"], _FLOAT_POINTS))
    if spec["type"] == "categorical":
        return list(spec["choices"])
    raise ValueError(f"Unknown search-space type for {name}: {spec['type']}. Options: int, float, categorical")


def tune_pipeline(task: str, cfg: dict, X, y, groups, cv):
    """
    Random search over `cfg["tuning"]["search_spaces"][<model type>]`: `tuning.n_trials` candidates, each
    scored on `tuning.metric` with `cv` (grouped by `groups`). Only the search runs here; refitting is
    left to the caller, so nested CV and the final training refit on the rows they choose.

    Returns (best_params, best_score): params as {"model__<name>": value}, ready for
    `pipeline.set_params(**best_params)`, and the mean CV score of the best candidate (optimistic,
    because the same folds chose it). Without a search space for the model, returns ({}, None).
    """
    tuning_cfg = cfg["tuning"]
    model_type = cfg["models"][task]["type"]
    space = tuning_cfg["search_spaces"].get(model_type)
    if not space:
        print(f"No search space for {model_type}: using the config params as they are.")
        return {}, None

    candidates = {f"model__{name}": _candidates(name, spec) for name, spec in space.items()}
    search = RandomizedSearchCV(build_pipeline(task, cfg), candidates, n_iter=tuning_cfg["n_trials"], cv=cv,
                                scoring=tuning_cfg["metric"], random_state=cfg["seed"], refit=False, n_jobs=-1)
    search.fit(X, y, groups=groups)
    return search.best_params_, float(search.best_score_)
