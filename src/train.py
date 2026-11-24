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
from src.tuning import tune_pipeline, nested_cross_validate

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

#--------------------
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

    model_type = cfg['models'][task]['type']
    pipeline = build_pipeline(task, cfg)
    
    X_dev = X.iloc[dev_idx]
    y_dev = y.iloc[dev_idx]
    groups_dev = groups.iloc[dev_idx]
    
    if cfg.get("tuning", {}).get("enabled", False): #se o tuning estiver ativo otimizar
        search_space = cfg["tuning"]["search_spaces"][model_type]
        n_trials = cfg["tuning"]["n_trials"]
        tuning_seed = cfg["tuning"].get("random_state", 42)
        scoring = cfg["tuning"]["metric"]
        sampler_choice = cfg["tuning"].get("sampler", "tpe")
        
        outer_cv = build_cv(cfg) #splitters da cross validation
        inner_cv = build_cv(cfg) 
        
        print("Executing Nested CV for realistic Tuning evaluation...")

        nested_df, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, groups_dev, outer_cv, inner_cv, scoring, search_space, n_trials, tuning_seed, 1, sampler_choice
        )
        print(" Final Tuning on the developing data completed...")

        best_pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, groups_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, 1, sampler_choice
        )
        
        best_params = study.best_params
        print(f"Best params: {best_params}")

        final_model = best_pipeline.fit(X, y)
        cv_scores = {
            "validation": {
                "mean": float(nested_df["validation"].mean()),
                "std": float(nested_df["validation"].std(ddof=1))
            }
        }
        
    else:
        # Caso contrário, treina com os parâmetros padrão (o que ja estava)
        best_params = {}
        cv_scores = cross_validate_pipeline(
            pipeline, X_dev, y_dev, groups_dev, build_cv(cfg), build_scorers(cfg["cv"]["metrics"])
        )
        print_scores(cv_scores, f"{model_type}, {cfg['cv']['n_splits']}-fold grouped CV (mean ± std):")

        final_model = pipeline.fit(X, y)

    meta = {
        "task": task,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "feature_names": list(final_model[:-1].get_feature_names_out()),
        "best_params": best_params,
        "threshold": cfg["threshold"],
        "cv_scores": cv_scores,
    }
    
    save_model(final_model, meta, task, cfg)