"""
Command-line entry point. Parses arguments, loads the config and calls src/. No logic here.

    python main.py train --task classification [--config configs/dev.yaml]
    python main.py predict [--config configs/dev.yaml]
"""
import argparse
from pathlib import Path

import yaml

from src.predict import run_predict
from src.train import train

ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "config.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Returns `base` updated with `override`: nested dicts merge key by key, lists and scalars replace."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str = None) -> dict:
    """Loads config.yaml and, if `path` is another file, deep-merges it on top as an override."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != BASE_CONFIG:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="ETAI project: repeat purchase and spend prediction.")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="cross-validate, fit on all training rows and save the model")
    train_parser.add_argument("--task", choices=["classification", "regression"], required=True)
    train_parser.add_argument("--config", default=str(BASE_CONFIG))

    predict_parser = commands.add_parser("predict", help="load the saved model and write the Kaggle CSV")
    predict_parser.add_argument("--config", default=str(BASE_CONFIG))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.command == "train":
        train(args.task, cfg)
    elif args.command == "predict":
        run_predict(cfg)


if __name__ == "__main__":
    main()
