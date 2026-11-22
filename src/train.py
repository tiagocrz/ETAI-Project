"""
Training, one function per `cv.strategy`:
  - stratified_group: nested CV of the tuning procedure, final tuning on all rows, refit, save the model;
  - time_series: re-evaluates the saved tuned params of the same model type with a time split (no save).
"""
import hashlib
import json
from datetime import datetime

import joblib
from sklearn.model_selection import FixedThresholdClassifier

from src.cleaning import clean, drop_duplicate_rows
from src.data import ROOT, join_clientes, load_raw, split_xyg
from src.evaluate import (build_scorers, cross_validate_folds, cross_validate_pipeline, nested_cross_validate,
                          summarize_folds)
from src.models import build_pipeline
from src.splitting import build_cv
from src.tuning import fit_with_threshold, tune_pipeline


def config_hash(cfg: dict) -> str:
    """Short hash of the merged config, so every saved model can be traced to its exact settings."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def load_training_data(cfg: dict, task: str):
    """
    Loads, cleans and de-duplicates train.csv. Returns (X, y, groups, dates); the purchase dates are only
    used by the time split and never enter X.
    """
    df = load_raw(cfg, "train_file")
    df = clean(df, cfg)
    df = drop_duplicate_rows(df, cfg["data"]["id_column"])
    df = join_clientes(df, cfg)
    X, y, groups = split_xyg(df, cfg, task)
    return X, y, groups, df[cfg["data"]["date_column"]]


def print_scores(scores: dict, title: str):
    print(title)
    for name, s in scores.items():
        print(f"  {name:<18s} {s['mean']:.3f} ± {s['std']:.3f}")


def write_json(obj: dict, path):
    """Writes `obj` as indented JSON, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Saved {path}")


def best_params_path(task: str, cfg: dict):
    """results_dir/best_params_<task>_<model type>.json: written by stratified_group, read by time_series."""
    return ROOT / cfg["output"]["results_dir"] / f"best_params_{task}_{cfg['models'][task]['type']}.json"


def save_model(pipeline, meta: dict, task: str, cfg: dict):
    """Saves models_dir/<task>.joblib and models_dir/<task>_meta.json."""
    models_dir = ROOT / cfg["output"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / f"{task}.joblib")
    print(f"Saved {models_dir / f'{task}.joblib'}")
    write_json(meta, models_dir / f"{task}_meta.json")


def train_stratified_group(task: str, cfg: dict, X, y, groups, dates):
    """
    With tuning: nested CV (StratifiedGroupKFold in both loops) estimates the "tune + fit" procedure
    without optimization bias, then the search runs once more on all rows to pick the final params.
    Without tuning there is nothing to bias, so plain CV is enough. Either way the final model is refit
    on all rows with its decision threshold (`threshold.method`) and saved, together with its best
    params and threshold for a later time-split re-evaluation.
    """
    scorers = build_scorers(cfg["cv"]["metrics"])
    cv = build_cv(cfg)
    model_type = cfg["models"][task]["type"]
    fold_scores, best_params, tuning_score = None, {}, None

    if cfg["tuning"]["enabled"]:
        print(f"Nested CV of {model_type} ({cfg['cv']['n_splits']} outer x {cfg['cv']['n_splits']} inner folds, "
              f"{cfg['tuning']['n_trials']} candidates per search):")
        fold_scores, _ = nested_cross_validate(task, cfg, X, y, groups, cv, cv, scorers)
        cv_scores = summarize_folds(fold_scores, scorers)
        print_scores(cv_scores, f"{model_type}, nested CV (mean ± std over outer folds):")
        best_params, tuning_score = tune_pipeline(task, cfg, X, y, groups, cv)
        print(f"Final params (searched on all rows): {best_params}")
    else:
        cv_scores = cross_validate_pipeline(build_pipeline(task, cfg), X, y, groups, cv, scorers)
        print_scores(cv_scores, f"{model_type}, {cfg['cv']['n_splits']}-fold grouped CV, no tuning (mean ± std):")

    model, threshold = fit_with_threshold(build_pipeline(task, cfg).set_params(**best_params), cfg, X, y, groups, cv)
    threshold_score = getattr(model, "best_score_", None)  # only the tuned method has one
    print(f"Decision threshold ({cfg['threshold']['method']}): {threshold:.3f}"
          + ("" if threshold_score is None else f", {cfg['threshold']['metric']} {threshold_score:.3f} on grouped CV")
          + f". All-positive baseline F1: {2 * y.mean() / (1 + y.mean()):.3f}")
    meta = {
        "task": task,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "feature_names": list(model.estimator_[:-1].get_feature_names_out()),
        "best_params": best_params,
        "tuning_score_optimistic": tuning_score,  # chosen on the same folds it is scored on
        "threshold": threshold,
        "threshold_score_optimistic": threshold_score,  # same: the cut was chosen on these folds
        "cv_scores": cv_scores,                    # nested CV when tuned, plain CV otherwise
        "nested_cv": None if fold_scores is None else fold_scores.to_dict("records"),
    }
    save_model(model, meta, task, cfg)
    write_json({"model_type": model_type, "config_hash": meta["config_hash"], "best_params": best_params,
                "threshold": threshold}, best_params_path(task, cfg))


def train_time_series(task: str, cfg: dict, X, y, groups, dates):
    """
    Second opinion on an already tuned model: plain TimeSeriesSplit CV (with the train-test gap) of the
    params and threshold saved by its stratified_group run. No tuning, no refit, and the saved model is left
    untouched.
    """
    model_type = cfg["models"][task]["type"]
    path = best_params_path(task, cfg)
    best_params, threshold = {}, cfg["threshold"]["value"]
    if path.exists():
        with open(path) as f:
            saved = json.load(f)
        best_params, threshold = saved["best_params"], saved.get("threshold", threshold)
        print(f"Using tuned params from {path}: {best_params}, threshold {threshold:.3f}")
    else:
        print(f"WARNING: {path} not found (run with cv.strategy: stratified_group first). "
              f"Using the config params and threshold {threshold}.")

    # Scored at the saved cut, the one the deployed model uses (no threshold tuning here)
    model = FixedThresholdClassifier(build_pipeline(task, cfg).set_params(**best_params), threshold=threshold)
    folds = build_cv(cfg, dates)
    fold_scores = cross_validate_folds(model, X, y, None, folds, build_scorers(cfg["cv"]["metrics"]))
    fold_scores.insert(1, "train_rows", [len(train_idx) for train_idx, _ in folds])
    fold_scores.insert(2, "train_end", [dates.iloc[train_idx].max().date() for train_idx, _ in folds])
    fold_scores.insert(3, "val_start", [dates.iloc[val_idx].min().date() for _, val_idx in folds])
    fold_scores.insert(4, "val_end", [dates.iloc[val_idx].max().date() for _, val_idx in folds])
    print(fold_scores.round(3).to_string(index=False))
    cv_scores = summarize_folds(fold_scores, cfg["cv"]["metrics"])
    print_scores(cv_scores, f"{model_type}, time series {cfg['cv']['n_splits']}-fold with a "
                            f"{cfg['cv']['time_gap_days']}-day gap (mean ± std):")

    write_json({"model_type": model_type, "config_hash": config_hash(cfg), "best_params": best_params,
                "threshold": threshold, "cv_scores": cv_scores, "folds": fold_scores.to_dict("records")},
               ROOT / cfg["output"]["results_dir"] / f"time_series_{task}_{model_type}.json")


_STRATEGIES = {
    "stratified_group": train_stratified_group,
    "time_series": train_time_series,
}


def train(task: str, cfg: dict):
    """Runs the training flow of `cfg["cv"]["strategy"]` on all labelled rows (no holdout)."""
    if task == "regression":
        raise NotImplementedError("TODO: train the regressor on rows with repeat_purchase_90d == 1")
    strategy = cfg["cv"]["strategy"]
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown cv.strategy: {strategy}. Options: {list(_STRATEGIES)}")
    _STRATEGIES[strategy](task, cfg, *load_training_data(cfg, task))
