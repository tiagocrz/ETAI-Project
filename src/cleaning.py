"""
Stateless, row-preserving cleaning rules. The same code runs on train, test and client input;
anything that learns from data belongs in the Pipeline instead.
"""
import pandas as pd


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Type fixes only (for now): the purchase date becomes a datetime, and count columns become
    float so train (which has NaNs) and test get the same dtypes. Every row is kept.
    """
    out = df.copy()
    date_column = cfg["data"]["date_column"]
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce")
    for col in cfg["cleaning"]["float_columns"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    Training data only: drops exact duplicate rows and repeated ids (keeping the first).
    Never call it on data to predict: every row needs a prediction.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out
