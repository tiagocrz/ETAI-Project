"""Model registry and the full Pipeline: features -> prep -> select -> model."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.selection import build_selector

_MODELS = {
    "classification": {
        "dummy": DummyClassifier,
        "random_forest": RandomForestClassifier,
    },
    "regression": {},  # TODO: add regression models
}


def build_model(task: str, cfg: dict):
    """Builds the model named in `cfg["models"][task]`, seeded with `cfg["seed"]` when it takes one."""
    model_cfg = cfg["models"][task]
    if model_cfg["type"] not in _MODELS[task]:
        raise ValueError(f"Unknown {task} model: {model_cfg['type']}. Options: {list(_MODELS[task])}")
    model = _MODELS[task][model_cfg["type"]](**(model_cfg.get("params") or {}))
    if "random_state" in model.get_params():
        model.set_params(random_state=cfg["seed"])
    return model


def build_pipeline(task: str, cfg: dict) -> Pipeline:
    """
    Everything learned lives in this one Pipeline, so each CV fold re-fits all of it and the
    saved model is the whole recipe.
    """
    return Pipeline([
        ("features", "passthrough"),  # TODO: FeatureEngineer (src/features.py)
        ("prep", build_preprocessor(cfg)),
        ("select", build_selector(cfg)),
        ("model", build_model(task, cfg)),
    ])
