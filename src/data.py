"""Loading the raw files, joining clientes.csv and splitting into features, target and groups."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_raw(cfg: dict, file_key: str) -> pd.DataFrame:
    """Loads one raw CSV named by `cfg["data"][file_key]` (e.g. "train_file")."""
    return pd.read_csv(ROOT / cfg["data"]["dir"] / cfg["data"][file_key])


def join_clientes(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Left-joins the configured clientes.csv columns on the customer id. Row-preserving, and each
    row only uses its own customer's record, so it is safe on test and client input.
    """
    columns = cfg["data"]["clientes_columns"]
    if not columns:
        return df
    key = cfg["data"]["group_column"]
    clientes = load_raw(cfg, "clientes_file")[[key, *columns]]
    return df.merge(clientes, on=key, how="left", validate="many_to_one")


def feature_columns(cfg: dict) -> list:
    """The columns X is made of: the configured numeric and categorical features, plus joined clientes columns."""
    return cfg["features"]["numeric"] + cfg["features"]["categorical"] + cfg["data"]["clientes_columns"]


def split_xyg(df: pd.DataFrame, cfg: dict, task: str):
    """
    Returns (X, y, groups). X holds only the configured feature columns, so ids, dates and both
    targets can never be features. On data without the target (test), y is None.
    """
    target = cfg["data"]["targets"][task]
    X = df[feature_columns(cfg)]
    y = df[target] if target in df.columns else None
    groups = df[cfg["data"]["group_column"]]
    return X, y, groups
