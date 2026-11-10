"""Batch prediction. predict_both() is the only place where the two models are combined."""
import json

import joblib
import pandas as pd

from src.cleaning import clean
from src.data import ROOT, join_clientes, load_raw, split_xyg


def predict_both(classifier, regressor, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    P(return) and the 0/1 decision (P >= threshold) for every row. Spend is meant to be
    predicted only for rows with decision 1 (not yet implemented).
    """
    if regressor is not None:
        raise NotImplementedError("TODO: predict spend_next_90d for rows predicted to return")
    proba = classifier.predict_proba(X)[:, 1]
    return pd.DataFrame({"proba": proba, "decision": (proba >= threshold).astype(int)}, index=X.index)


def load_model(task: str, cfg: dict):
    """Loads models_dir/<task>.joblib and its metadata. Never retrains."""
    models_dir = ROOT / cfg["output"]["models_dir"]
    with open(models_dir / f"{task}_meta.json") as f:
        meta = json.load(f)
    return joblib.load(models_dir / f"{task}.joblib"), meta


def run_predict(cfg: dict):
    """
    Predicts every test row and writes the Kaggle CSV (ID, repeat_purchase_90d) in the same
    row order as sample_submission.csv. Each row uses only itself, the model and clientes.csv.
    """
    classifier, meta = load_model("classification", cfg)
    test = join_clientes(clean(load_raw(cfg, "test_file"), cfg), cfg)
    X, _, _ = split_xyg(test, cfg, "classification")
    preds = predict_both(classifier, None, X, meta["threshold"])

    id_col, target = cfg["data"]["id_column"], cfg["data"]["targets"]["classification"]
    by_id = pd.Series(preds["decision"].values, index=test[id_col].values)
    submission = load_raw(cfg, "sample_submission_file")[[id_col]]
    submission[target] = submission[id_col].map(by_id)
    if submission[target].isna().any() or len(submission) != len(test):
        raise ValueError("Test IDs do not match sample_submission.csv")
    submission[target] = submission[target].astype(int)

    results_dir = ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "submission_classification.csv"
    submission.to_csv(path, index=False)
    print(f"Wrote {path} ({len(submission)} rows, {submission[target].mean():.1%} predicted to return)")
    # TODO: regression CSV (ID, repeat_purchase_90d, spend_next_90d) for rows predicted to return
