"""Imputation, encoding and scaling, as one unfitted ColumnTransformer (the Pipeline's "prep" step)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

_SCALERS = {
    "none": lambda: "passthrough",
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}
# Both tolerate categories never seen in training (test has new commercial zones).
# TODO: add "target" (TargetEncoder) once it uses group-aware internal folds.
_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """Numeric columns: impute, then scale. Categorical columns: impute, then encode."""
    prep_cfg = cfg["preprocessing"]
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=prep_cfg["numeric_imputer"])),
        ("scale", _SCALERS[prep_cfg["scaler"]]()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=prep_cfg["categorical_imputer"])),
        ("encode", _ENCODERS[prep_cfg["encoder"]]()),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, cfg["features"]["numeric"]),
        ("categorical", categorical_pipeline, cfg["features"]["categorical"] + cfg["data"]["clientes_columns"]),
    ])
