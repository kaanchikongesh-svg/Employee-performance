"""
Model Training and Comparison Module for Employee Performance Prediction.
Trains Random Forest, Logistic Regression, and Gradient Boosting models,
evaluates metrics (Accuracy, Precision, Recall, F1), generates confusion matrices,
selects the best model, and saves pipelines.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.config import (
    MODEL_FILE_PATH,
    SAMPLE_DATA_PATH,
    CATEGORIES,
    DATA_DIR,
    MODELS_DIR,
    score_to_category,
    GENDER_OPTIONS,
    DEPARTMENT_OPTIONS,
    JOB_ROLE_OPTIONS,
    PROMOTION_OPTIONS,
    ALL_FEATURE_COLUMNS,
    TARGET_SCORE_COLUMN,
    TARGET_CATEGORY_COLUMN,
)
from src.preprocessing import create_preprocessor, clean_and_prepare_data, get_feature_names


def generate_synthetic_dataset(num_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic HR dataset for Employee Performance Prediction.
    Labeled as 'Synthetic Demo Dataset'.
    """
    np.random.seed(random_state)

    departments = DEPARTMENT_OPTIONS
    genders = GENDER_OPTIONS
    gender_weights = [0.48, 0.46, 0.04, 0.02]

    records = []

    for i in range(1, num_samples + 1):
        emp_id = f"EMP-{i:04d}"
        gender = np.random.choice(genders, p=gender_weights)
        dept = np.random.choice(departments)
        roles = JOB_ROLE_OPTIONS[dept]
        role = np.random.choice(roles)

        # Experience and Age correlated
        exp = float(np.round(np.clip(np.random.gamma(shape=2.5, scale=2.5), 0.5, 30.0), 1))
        age = int(np.clip(21 + int(exp * 1.1) + np.random.randint(-2, 5), 21, 62))

        # Job level correlates with experience
        job_level = int(np.clip(1 + int(exp / 5) + np.random.choice([0, 1], p=[0.7, 0.3]), 1, 5))

        # Salary correlated with job level & experience
        base_salary = 3000 + (job_level * 2200) + (exp * 350) + np.random.normal(0, 500)
        monthly_income = float(np.round(np.clip(base_salary, 2500, 24000), -1))

        job_satisfaction = int(np.clip(np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.15, 0.35, 0.28, 0.14]), 1, 5))
        work_life_balance = int(np.clip(np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.20, 0.38, 0.22, 0.10]), 1, 5))

        # Hours & Attendance
        working_hours = float(np.round(np.clip(np.random.normal(41.5, 5.0), 28.0, 65.0), 1))
        overtime_hours = float(np.round(np.clip(max(0.0, (working_hours - 40.0) * 4.2 + np.random.normal(2, 3)), 0.0, 55.0), 1))

        attendance_pct = float(np.round(np.clip(np.random.beta(a=18, b=1.5) * 100, 62.0, 100.0), 1))

        # Training
        training_hours = float(np.round(np.clip(np.random.gamma(shape=3.0, scale=12.0), 5.0, 120.0), 1))
        num_trainings = int(np.clip(int(training_hours / 14.0) + np.random.randint(-1, 2), 0, 9))

        projects_completed = int(np.clip(int(exp * 0.8) + np.random.randint(1, 6), 1, 25))

        prev_perf = float(np.round(np.clip(np.random.normal(74.0, 11.0), 38.0, 99.0), 1))

        promotion_hist = np.random.choice(PROMOTION_OPTIONS, p=[0.65, 0.35] if exp > 3 else [0.9, 0.1])
        years_since_promo = float(np.round(np.clip(np.random.uniform(0.2, min(exp, 8.0)), 0.1, 12.0), 1))
        team_size = int(np.clip(np.random.poisson(lam=7) + 2, 2, 25))

        # Performance score generation with realistic domain logic
        # Positive drivers: Previous performance, Attendance, Training hours, Job satisfaction, Projects
        # Negative / drag drivers: Excessive overtime burnout (over 30 hrs), poor work life balance, low attendance
        score_base = (
            (prev_perf * 0.38)
            + ((attendance_pct - 60) * 0.35)
            + (min(training_hours, 80) * 0.16)
            + (job_satisfaction * 3.2)
            + (work_life_balance * 1.5)
            + (min(projects_completed, 15) * 0.8)
            - (max(0, overtime_hours - 25) * 0.35)
            + (1.5 if promotion_hist == "Yes" else 0.0)
            + np.random.normal(0, 3.8)
        )

        score = float(np.round(np.clip(score_base, 35.0, 98.5), 1))
        category = score_to_category(score)

        records.append({
            "Employee_ID": emp_id,
            "Age": age,
            "Gender": gender,
            "Department": dept,
            "Job_Role": role,
            "Years_of_Experience": exp,
            "Monthly_Income": monthly_income,
            "Job_Level": job_level,
            "Job_Satisfaction": job_satisfaction,
            "Work_Life_Balance": work_life_balance,
            "Working_Hours_per_Week": working_hours,
            "Attendance_Percentage": attendance_pct,
            "Training_Hours": training_hours,
            "Number_of_Trainings": num_trainings,
            "Projects_Completed": projects_completed,
            "Overtime_Hours": overtime_hours,
            "Previous_Performance_Score": prev_perf,
            "Promotion_History": promotion_hist,
            "Years_Since_Last_Promotion": years_since_promo,
            "Team_Size": team_size,
            "Performance_Score": score,
            "Performance_Category": category
        })

    df = pd.DataFrame(records)
    return df


def ensure_sample_dataset() -> pd.DataFrame:
    """Ensures sample dataset exists on disk; creates it if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_DATA_PATH.exists():
        df = generate_synthetic_dataset()
        df.to_csv(SAMPLE_DATA_PATH, index=False)
        return df
    return pd.read_csv(SAMPLE_DATA_PATH)


def train_and_compare_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains and benchmarks Random Forest, Logistic Regression, and Gradient Boosting.
    Calculates Accuracy, Precision, Recall, F1, and Confusion Matrices.
    Automatically identifies and returns the best model.
    """
    X, y_cat, y_score = clean_and_prepare_data(df, is_training=True)

    if y_cat is None or y_cat.dropna().empty:
        raise ValueError("Cannot train model without target category or score column.")

    # Drop NaNs in target
    valid_idx = y_cat.notnull()
    X = X[valid_idx]
    y_cat = y_cat[valid_idx]
    if y_score is not None:
        y_score = y_score[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=test_size, random_state=random_state, stratify=y_cat
    )

    models_config = {
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=10, min_samples_split=4, random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=4, random_state=random_state
        )
    }

    results = {}
    fitted_pipelines = {}

    for name, clf in models_config.items():
        preprocessor = create_preprocessor()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "labels": CATEGORIES
        }
        fitted_pipelines[name] = pipeline

    # Also train score regressor for continuous score prediction
    score_regressor = None
    if y_score is not None:
        X_train_s, X_test_s, ys_train, ys_test = train_test_split(
            X, y_score, test_size=test_size, random_state=random_state
        )
        score_reg_pipe = Pipeline(steps=[
            ("preprocessor", create_preprocessor()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state))
        ])
        score_reg_pipe.fit(X_train_s, ys_train)
        score_regressor = score_reg_pipe

    # Select best model based on F1 Score, then Accuracy
    best_model_name = max(
        results.keys(),
        key=lambda k: (results[k]["f1_score"], results[k]["accuracy"])
    )

    best_pipeline = fitted_pipelines[best_model_name]

    # Compute feature importance for best model
    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"])
    clf_step = best_pipeline.named_steps["classifier"]

    feature_importances = {}
    if hasattr(clf_step, "feature_importances_"):
        raw_importances = clf_step.feature_importances_
        for feat, imp in zip(feature_names, raw_importances):
            feature_importances[feat] = float(imp)
    elif hasattr(clf_step, "coef_"):
        # Mean absolute coefficient across classes
        raw_importances = np.mean(np.abs(clf_step.coef_), axis=0)
        for feat, imp in zip(feature_names, raw_importances):
            feature_importances[feat] = float(imp)

    # Sort feature importances descending
    sorted_importances = dict(
        sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    )

    payload = {
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "all_pipelines": fitted_pipelines,
        "metrics": results,
        "feature_importances": sorted_importances,
        "score_regressor": score_regressor,
        "trained_on_samples": len(X),
        "test_samples": len(X_test)
    }

    return payload


def save_model_package(payload: Dict[str, Any], filepath: Optional[os.PathLike] = None) -> Path:
    """Saves the trained model pipeline and metadata to disk using joblib."""
    target_path = Path(filepath) if filepath else MODEL_FILE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, target_path)
    return target_path


def load_model_package(filepath: Optional[os.PathLike] = None) -> Optional[Dict[str, Any]]:
    """Loads saved model package from disk."""
    target_path = Path(filepath) if filepath else MODEL_FILE_PATH
    if not target_path.exists():
        return None
    try:
        return joblib.load(target_path)
    except Exception as e:
        print(f"Error loading model from {target_path}: {e}")
        return None
