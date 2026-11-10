"""Training: cross-validate on the development rows, refit on all training rows, save model + metadata."""
import hashlib
import json
from datetime import datetime

import joblib

from src.cleaning import clean, drop_duplicate_rows
from src.data import ROOT, join_clientes, load_raw, split_xyg
from src.evaluate import build_scorers, cross_validate_pipeline
from src.models import build_pipeline
from src.splitting import build_cv, split_holdout


def config_hash(cfg: dict) -> str:
    """Short hash of the merged config, so every saved model can be traced to its exact settings."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def load_training_data(cfg: dict, task: str):
    """Loads, cleans and de-duplicates train.csv, then returns (X, y, groups)."""
    df = load_raw(cfg, "train_file")
    df = clean(df, cfg)
    df = drop_duplicate_rows(df, cfg["data"]["id_column"])
    df = join_clientes(df, cfg)
    return split_xyg(df, cfg, task)


def print_scores(scores: dict, title: str):
    print(title)
    for name, s in scores.items():
        print(f"  {name:<18s} {s['mean']:.3f} ± {s['std']:.3f}")


def save_model(pipeline, meta: dict, task: str, cfg: dict):
    """Saves models_dir/<task>.joblib and models_dir/<task>_meta.json."""
    models_dir = ROOT / cfg["output"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / f"{task}.joblib")
    with open(models_dir / f"{task}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved {models_dir / f'{task}.joblib'}")


def train(task: str, cfg: dict):
    """
    CV on the development rows (the locked holdout is left out), then refit on all training
    rows so the saved model uses all the labelled data.
    """
    if task == "regression":
        raise NotImplementedError("TODO: train the regressor on rows with repeat_purchase_90d == 1")

    X, y, groups = load_training_data(cfg, task)
    dev_idx, holdout_idx = split_holdout(X, y, groups, cfg)
    print(f"{len(X)} training rows: {len(dev_idx)} development, {len(holdout_idx)} locked holdout (not used here)")

    cv_scores = cross_validate_pipeline(build_pipeline(task, cfg), X.iloc[dev_idx], y.iloc[dev_idx],
                                        groups.iloc[dev_idx], build_cv(cfg), build_scorers(cfg["cv"]["metrics"]))
    print_scores(cv_scores, f"{cfg['models'][task]['type']}, {cfg['cv']['n_splits']}-fold grouped CV (mean ± std):")

    pipeline = build_pipeline(task, cfg).fit(X, y)
    meta = {
        "task": task,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "feature_names": list(pipeline[:-1].get_feature_names_out()),
        "threshold": cfg["threshold"],
        "cv_scores": cv_scores,
    }
    save_model(pipeline, meta, task, cfg)
