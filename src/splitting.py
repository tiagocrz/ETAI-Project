"""CV splitters and the locked holdout. Customers appear in several rows, so splits are grouped on customer_id."""
from sklearn.model_selection import StratifiedGroupKFold

_SPLITTERS = {
    "stratified_group": lambda n_splits, seed: StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
    # TODO: "time_split" on purchase_date (test holds later purchases)
}


def build_cv(cfg: dict):
    """The CV splitter named in `cfg["cv"]["strategy"]`."""
    return _SPLITTERS[cfg["cv"]["strategy"]](cfg["cv"]["n_splits"], cfg["seed"])


def split_holdout(X, y, groups, cfg: dict):
    """
    Splits off the locked holdout (about `holdout.size` of the rows, whole customers only).
    Returns (dev_idx, holdout_idx) as positional indices.
    """
    n_splits = round(1 / cfg["holdout"]["size"])
    splitter = _SPLITTERS[cfg["holdout"]["strategy"]](n_splits, cfg["seed"])
    dev_idx, holdout_idx = next(splitter.split(X, y, groups))
    return dev_idx, holdout_idx
