"""Scorers and cross-validation, shared by both tasks."""
from sklearn.metrics import fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """{name: scorer} for scikit-learn metric names, plus `fbeta_<beta>` (e.g. fbeta_2)."""
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def cross_validate_pipeline(pipeline, X, y, groups, cv, scorers: dict) -> dict:
    """
    Fits a fresh copy of the whole pipeline on each fold's training part (folds grouped by
    customer) and scores it on the validation part. Returns {metric: {"mean", "std"}}.
    """
    scores = cross_validate(pipeline, X, y, groups=groups, cv=cv, scoring=scorers)
    return {name: {"mean": float(scores[f"test_{name}"].mean()), "std": float(scores[f"test_{name}"].std(ddof=1))}
            for name in scorers}


def nested_cross_validate(*args, **kwargs):
    raise NotImplementedError("TODO: nested CV of the tuning procedure (course nested_cross_validate + groups)")


def corrected_paired_test(*args, **kwargs):
    raise NotImplementedError("TODO: corrected paired t-test (course corrected_paired_test)")


def append_scoreboard(*args, **kwargs):
    raise NotImplementedError("TODO: append one row per run to results/runs_<task>.csv")
