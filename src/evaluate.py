"""Scorers, cross-validation and nested cross-validation, shared by both tasks."""
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import cross_validate

from src.models import build_pipeline
from src.tuning import fit_with_threshold, tune_pipeline


def build_scorers(metrics: list) -> dict:
    """{name: scorer} for scikit-learn metric names, plus `fbeta_<beta>` (e.g. fbeta_2)."""
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def summarize_folds(fold_scores: pd.DataFrame, metrics) -> dict:
    """{metric: {"mean", "std"}} over the folds of a fold-score table."""
    return {name: {"mean": float(fold_scores[name].mean()), "std": float(fold_scores[name].std(ddof=1))}
            for name in metrics}


def cross_validate_folds(pipeline, X, y, groups, cv, scorers: dict) -> pd.DataFrame:
    """
    Fits a fresh copy of the whole pipeline on each fold's training part and scores it on the
    validation part. Returns one row per fold with the validation score of every metric.
    """
    scores = cross_validate(pipeline, X, y, groups=groups, cv=cv, scoring=scorers)
    fold_scores = pd.DataFrame({name: scores[f"test_{name}"] for name in scorers})
    fold_scores.insert(0, "fold", range(1, len(fold_scores) + 1))
    return fold_scores


def cross_validate_pipeline(pipeline, X, y, groups, cv, scorers: dict) -> dict:
    """Plain CV of one fixed pipeline. Returns {metric: {"mean", "std"}}."""
    return summarize_folds(cross_validate_folds(pipeline, X, y, groups, cv, scorers), scorers)


def nested_cross_validate(task: str, cfg: dict, X, y, groups, outer_cv, inner_cv, scorers: dict):
    """
    Honest estimate of the whole "tune + fit" procedure. For every outer fold: tune on the outer-training
    rows (`tune_pipeline` with `inner_cv`), refit the best pipeline on them (for classification together
    with its decision threshold, tuned on the same inner folds), and score it on the outer-validation
    rows, which took no part in the tuning. Adapted from the course helper, with groups.

    Returns (fold_scores, oof_pred): one row per outer fold (train and validation score of
    `tuning.metric`, their gap, every validation metric, the inner best score, the threshold and the
    chosen params),
    and the out-of-fold prediction for every row (probability of class 1 for classification).
    """
    metric = cfg["tuning"]["metric"]
    rows = []
    oof_pred = np.full(len(X), np.nan)
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train, y_train, g_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        best_params, inner_best = tune_pipeline(task, cfg, X_train, y_train, g_train, inner_cv)
        pipeline = build_pipeline(task, cfg).set_params(**best_params)
        threshold = None
        if task == "classification":  # the threshold is tuned on the outer-training rows only, like the params
            pipeline, threshold = fit_with_threshold(pipeline, cfg, X_train, y_train, g_train, inner_cv)
        else:
            pipeline.fit(X_train, y_train)

        train_score = scorers[metric](pipeline, X_train, y_train)
        val_scores = {name: scorer(pipeline, X_val, y_val) for name, scorer in scorers.items()}
        oof_pred[val_idx] = (pipeline.predict_proba(X_val)[:, 1] if task == "classification"
                             else pipeline.predict(X_val))
        rows.append({"fold": fold, "train": train_score, "validation": val_scores[metric],
                     "gap": train_score - val_scores[metric], **val_scores,
                     "inner_best": inner_best, "threshold": threshold, **best_params})
        inner_text = "no search" if inner_best is None else f"inner best {inner_best:.3f}"
        threshold_text = "" if threshold is None else f", threshold {threshold:.3f}"
        print(f"  outer fold {fold}: {metric} {val_scores[metric]:.3f} ({inner_text}), params {best_params}"
              f"{threshold_text}")
    return pd.DataFrame(rows), oof_pred


def corrected_paired_test(*args, **kwargs):
    raise NotImplementedError("TODO: corrected paired t-test (course corrected_paired_test)")


def append_scoreboard(*args, **kwargs):
    raise NotImplementedError("TODO: append one row per run to results/runs_<task>.csv")
