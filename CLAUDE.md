# CLAUDE.md

Guidance for Claude Code (and teammates) working in this repository.

## Project
ETAI 2026/27 group project, "Everything & Then Some": a wholesale company's purchase data. For every purchase occasion, predict:
1. **Classification**: will the customer buy again within 90 days? (`repeat_purchase_90d`, the Kaggle target)
2. **Regression**: how much will they spend in those 90 days? (`spend_next_90d`, only for customers predicted to return)

Deliverables: this repo (pipelines + README + `config.yaml` + requirements), a zip of all code, and a Kaggle submission that matches the pipeline's output. **Deadline: 22 Dec 2026, 18:00.** Submitting early gives +0.15/day, up to +1.

Grading (out of 20): classification 5 · regression 4 · EDA & cleaning 3 · code quality 3 · explainability & extra 3 · deployment 2. The Kaggle score is only 1 point, so favour sound methodology over leaderboard tuning.

## Status
<!-- Keep this current: what exists, what is a stub, what was decided. -->
- [ ] Skeleton (`main.py`, `src/`, `config.yaml`): not built yet
- [ ] EDA + cleaning decisions table
- [ ] Classification: baseline → ≥5 models → feature selection → tuning
- [ ] Regression: ≥4 models
- [ ] Batch prediction + client interface
- [ ] Explainability (SHAP/LIME) + extra section

## Data (`data/raw/`)
- `train.csv`: 6,534 purchase occasions with both targets. `test.csv`: 986 **later** occasions, no targets.
- `clientes.csv`: optional customer table, joined on `customer_id`. `sample_submission.csv`: Kaggle format (`ID`, `repeat_purchase_90d`).
- Column groups: current monetary values, current purchase details, current categoricals (payment, salesperson, zone, transport, warehouse, series), historical timing, historical value, rolling frequency/spend (30/90/180/365 days), `random_noise` (check whether it carries any signal).
- Customers appear in **several rows**, so `customer_id` is a grouping variable.

## Hard rules (from the brief: never break these)
- **Never use `spend_next_90d` as a feature** for classification, and never use either target as a feature.
- Features must be computable **at the time of the purchase**.
- Each test row is predicted using **only that row, the training data and `clientes.csv`**, never other test rows. No groupby or aggregation over `test.csv`.
- Allowed libraries: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, LIME, imbalanced-learn, TensorFlow/Keras, PyTorch (plus pandas/numpy/matplotlib/PyYAML). **No AutoML** (Lazy Predict, Featuretools, TSFresh, etc.).
- The whole pipeline must run **from the terminal without editing code**, on Mac and Windows, from a fresh clone. Any choice that can change goes in `config.yaml`.
- Predicting never retrains: `train` saves models, `predict` and `client` only load them.

## Commands (target interface)
```bash
python main.py train --task classification   [--config configs/dev.yaml]
python main.py train --task regression
python main.py evaluate --task classification   # CV / nested CV / model comparison → results/
python main.py predict                          # → Kaggle CSV + regression CSV
python main.py client --input purchase.json     # one purchase → P(return), decision, expected spend
```
Run the full dev flow `train → predict → client` with `--config configs/dev.yaml` before every merge.

## Config
- `config.yaml` is the **single source of truth**: it holds every setting and produces the final models and the Kaggle submission.
- Files in `configs/` are **overrides**. They contain only the keys that differ, and are deep-merged on top of `config.yaml` by `load_config()` in `main.py`. Never make full copies of the config.
  - `configs/dev.yaml`: fast smoke run (e.g. `cv.n_splits: 2`, `tuning.n_trials: 5`).
  - `configs/exp_*.yaml`: one experiment each (e.g. `cv: {strategy: time_split}`).
- Merge semantics: nested dicts merge key by key. **Lists and scalar values replace** the base value; they are not appended.
- Override configs must set their own `output.models_dir` / `output.results_dir` (e.g. `models/dev`, `results/dev`), so they never overwrite the final models or the main results log. `predict` and `client` read models from the merged config's `models_dir`.
- The **merged** config is saved in `<task>_meta.json`, and its hash is logged in the results log, so every model can be traced to its exact settings.

## Architecture
```
main.py            CLI only: parses args, loads/merges config, calls src/. No logic here.
config.yaml        Every changeable choice. Overrides go in configs/.
src/data.py        load data, join clientes, split X / y / groups
src/cleaning.py    stateless, row-preserving rules from the EDA; drop duplicates on TRAIN only
src/features.py    FeatureEngineer (sklearn transformer)
src/preprocessing.py  build_preprocessor: impute / encode / scale
src/selection.py   _SELECTORS + build_selector
src/models.py      _MODELS["classification"|"regression"] + build_pipeline(task, cfg)
src/splitting.py   _SPLITTERS + locked test split
src/evaluate.py    scorers, CV, nested CV, corrected paired t-test, scoreboard
src/tuning.py      Optuna tuning
src/predict.py     predict_both(): the ONLY place where the two models are combined
src/client.py      input parsing, new vs returning customer handling
notebooks/         EDA + experiments: these import from src, never copy its code
models/, results/  generated outputs
```

### Design principles
1. **Config + registries.** Every swappable piece (CV scheme, feature groups, selector, encoder, scaler, model, target transform, metrics, search spaces) is a named option in `config.yaml`, looked up in a module-level dict (`_MODELS`, `_SPLITTERS`, `_SELECTORS`, ...). To add an option: write one function, add one entry to the dict, change the config. Don't add `if/else` chains for options.
2. **Everything learned lives inside one sklearn `Pipeline`:**
   `features → prep → select → model`. Never fit anything (imputer, encoder, selector, feature lookup) outside the pipeline or on the full dataset; that leaks data across CV folds. The saved `.joblib` is the whole recipe.
3. **Cleaning vs. learning.** `cleaning.py` is stateless and keeps every row, so the same code runs on train, test and client input. Anything that learns from data belongs in the Pipeline.
4. **Two-stage model.** The classifier trains on all rows. The regressor trains only on rows with `repeat_purchase_90d == 1`. At prediction time: `return = P ≥ threshold` (threshold from config, chosen on out-of-fold probabilities), and spend is predicted only when `return == 1`. `predict` and `client` both call `predict_both()`.
5. **One task-agnostic evaluation core.** CV, tuning and nested CV work for both tasks. They take `groups=`, and only call `predict_proba` for classification.
6. **Validation.** Default to `StratifiedGroupKFold` grouped on `customer_id`. Also consider a time-ordered split on `purchase_date`, since the test set holds later purchases. The locked holdout is used once, for the final assessment.
7. **Metadata with every model.** Next to `models/<task>.joblib`, save `models/<task>_meta.json` (merged config, final feature names, threshold, CV scores). The client and SHAP read it.
8. **Scoreboard.** Append one row per evaluation run to `results/runs_<task>.csv` (timestamp, config hash, model, CV mean/std per metric).

### Origin of the code
The core comes from the course pipeline (`course_helpers.py`, weeks 3–8): `clean_dataset`, `flag_invalid_values`, `drop_duplicate_rows`, `add_missingness_indicators`, `build_preprocessor`, `build_model`, `build_pipeline`, `build_scorers`, `cross_validate_pipeline`, `tune_pipeline`, `nested_cross_validate`, `corrected_paired_test`. Adapt these rather than writing new versions. The known changes needed are `groups=` support, regression support, and a group-aware CV for `TargetEncoder`.

## Outputs
- Kaggle file: `ID, repeat_purchase_90d`, the same rows and IDs as `sample_submission.csv`.
- Regression file: `ID, repeat_purchase_90d, spend_next_90d`, only the rows predicted to return.
- Regression is graded with RMSE, MAE and R² on customers who actually returned.
- The client must work for both a **returning** customer (`customer_id` present in train) and a **new** one (no history features).

## Code conventions
- Use `pathlib`, with paths relative to the repo root. No absolute paths, no OS-specific commands.
- Use one `seed` from config, passed to every estimator, splitter and Optuna sampler.
- Keep functions small, with docstrings that say *what* and *why*. Don't put theory explanations of class topics in code or docs.
- Keep code simple and explainable. Every team member must be able to explain any line at the defense, and the code goes through plagiarism and AI-generation checks. Don't add techniques the team can't justify.
- Use pinned versions in `requirements.txt`, and state the Python version in the README.
- Clear notebook outputs before committing. Don't commit large generated files.
- Cite only peer-reviewed sources in the report and README. Never cite LLMs.

## Git workflow
- **Claude Code never runs git commands that change the repository: no `git add`, `git commit`, `git push`, `git merge`, `git rebase`, branch creation or PRs.** All commits and pushes are done by the user. Read-only commands (`git status`, `git diff`, `git log`) are fine.
- When a piece of work is done, say which files changed and, if useful, suggest a commit message. Don't run the commit.
- Team convention: one branch per feature, and a PR reviewed by another teammate before merging to `main`. `main` must always pass the dev-config smoke run.
