"""
Preprocessing and Data Pipeline Module for Employee Performance Prediction.
Builds robust ColumnTransformers, validates schemas, and handles missing/categorical data.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
    TARGET_SCORE_COLUMN,
    TARGET_CATEGORY_COLUMN,
    score_to_category,
)


def create_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for preprocessing numerical
    and categorical features with imputation and scaling.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def validate_dataframe(df: pd.DataFrame, require_target: bool = False) -> Tuple[bool, List[str]]:
    """
    Validates whether the dataframe contains required columns and correct formats.
    Returns (is_valid, list_of_errors_or_warnings).
    """
    errors = []
    if df is None or df.empty:
        return False, ["Dataset is empty."]

    missing_features = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {', '.join(missing_features)}")

    if require_target and TARGET_SCORE_COLUMN not in df.columns and TARGET_CATEGORY_COLUMN not in df.columns:
        errors.append(
            f"Target column missing. Expected either '{TARGET_SCORE_COLUMN}' (continuous) or '{TARGET_CATEGORY_COLUMN}' (High/Medium/Low)."
        )

    return len(errors) == 0, errors


def clean_and_prepare_data(
    df: pd.DataFrame,
    is_training: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Cleans raw DataFrame, handles missing columns by adding defaults if missing,
    derives target category if only score exists (or vice versa), and returns (X, y_category, y_score).
    """
    clean_df = df.copy()

    # Drop duplicate rows based on ID if present
    if "Employee_ID" in clean_df.columns:
        clean_df = clean_df.drop_duplicates(subset=["Employee_ID"]).reset_index(drop=True)
    else:
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    # Coerce numeric types
    for col in NUMERICAL_FEATURES:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        else:
            clean_df[col] = np.nan

    # Coerce categorical types
    for col in CATEGORICAL_FEATURES:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.strip()
        else:
            clean_df[col] = "Unknown"

    X = clean_df[ALL_FEATURE_COLUMNS].copy()

    y_category = None
    y_score = None

    if is_training:
        if TARGET_SCORE_COLUMN in clean_df.columns:
            clean_df[TARGET_SCORE_COLUMN] = pd.to_numeric(clean_df[TARGET_SCORE_COLUMN], errors="coerce")
            y_score = clean_df[TARGET_SCORE_COLUMN]

            if TARGET_CATEGORY_COLUMN not in clean_df.columns:
                y_category = y_score.apply(lambda s: score_to_category(float(s)) if pd.notnull(s) else np.nan)
            else:
                y_category = clean_df[TARGET_CATEGORY_COLUMN].astype(str)
        elif TARGET_CATEGORY_COLUMN in clean_df.columns:
            y_category = clean_df[TARGET_CATEGORY_COLUMN].astype(str)
            # Impute dummy scores for visualization if needed
            score_map = {"High": 90.0, "Medium": 70.0, "Low": 45.0}
            y_score = y_category.map(score_map).fillna(60.0)

    return X, y_category, y_score


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts transformed feature names from fitted ColumnTransformer.
    """
    feature_names = []
    try:
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        # Fallback manual reconstruction
        num_names = NUMERICAL_FEATURES
        cat_names = []
        try:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        except Exception:
            for cat_col in CATEGORICAL_FEATURES:
                cat_names.append(f"{cat_col}_encoded")
        return num_names + cat_names
