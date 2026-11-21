"""
CV splitters. Customers appear in several rows, so the default split is grouped on customer_id.
The time split is a second opinion: the test set holds later purchases.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, TimeSeriesSplit


def time_series_folds(dates: pd.Series, n_splits: int, gap_days: int) -> list:
    """
    Expanding-window folds in date order, with a gap of `gap_days` before each validation block, like the
    gap between train.csv and test.csv. TimeSeriesSplit's own `gap` counts rows, so the gap in days is
    applied here by dropping training rows dated less than `gap_days` before the first validation date.
    Rows without a date cannot be placed in time, so they are left out of every fold.
    Returns a list of (train_idx, val_idx) positional arrays, usable as `cv=` in scikit-learn.
    """
    dates = pd.to_datetime(pd.Series(dates)).to_numpy()
    dated = np.flatnonzero(~np.isnat(dates))
    if len(dated) < len(dates):
        print(f"Time split: {len(dates) - len(dated)} rows without a purchase date are left out of the folds.")
    order = dated[np.argsort(dates[dated], kind="stable")]  # positions sorted by date: no pre-sorting needed
    gap = np.timedelta64(gap_days, "D")
    folds = []
    for train_pos, val_pos in TimeSeriesSplit(n_splits).split(order):
        train_idx, val_idx = order[train_pos], order[val_pos]
        val_start = dates[val_idx].min()
        folds.append((train_idx[dates[train_idx] <= val_start - gap], val_idx))
    return folds


_SPLITTERS = {
    "stratified_group": lambda cfg, dates: StratifiedGroupKFold(
        n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"]),
    "time_series": lambda cfg, dates: time_series_folds(dates, cfg["cv"]["n_splits"], cfg["cv"]["time_gap_days"]),
}


def build_cv(cfg: dict, dates: pd.Series = None):
    """
    The CV splitter named in `cfg["cv"]["strategy"]`. `dates` (purchase date per row, aligned with X) is
    only needed by the time split.
    """
    strategy = cfg["cv"]["strategy"]
    if strategy not in _SPLITTERS:
        raise ValueError(f"Unknown cv.strategy: {strategy}. Options: {list(_SPLITTERS)}")
    return _SPLITTERS[strategy](cfg, dates)
