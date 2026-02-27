from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data import get_feature_columns, get_target_column


def build_models(random_state: int) -> Dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "K-Nearest Neighbors": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=3,
                        weights="distance",
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    model_defs: Dict[str, Any], X: pd.DataFrame, y: np.ndarray, random_state: int
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, float]] = []
    notes: List[str] = [
        "Leakage-safe evaluation enabled: grouped by full symptom pattern to prevent duplicate leakage."
    ]
    all_labels = np.unique(y)
    groups = X.astype(str).agg("|".join, axis=1).values
    unique_group_count = len(np.unique(groups))

    holdout_split = None
    if unique_group_count >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        holdout_split = next(gss.split(X, y, groups=groups))
    else:
        notes.append("Holdout evaluation skipped: fewer than 2 unique symptom-pattern groups.")

    cv_splits = min(5, unique_group_count)
    if cv_splits < 2:
        notes.append("Cross-validation skipped: fewer than 2 unique symptom-pattern groups.")

    for name, base_model in model_defs.items():
        train_model = clone(base_model)
        train_model.fit(X, y)
        train_acc = accuracy_score(y, train_model.predict(X))

        holdout_acc = np.nan
        holdout_f1 = np.nan
        if holdout_split is not None:
            train_idx, test_idx = holdout_split
            test_model = clone(base_model)
            test_model.fit(X.iloc[train_idx], y[train_idx])
            y_pred = test_model.predict(X.iloc[test_idx])
            holdout_acc = accuracy_score(y[test_idx], y_pred)
            holdout_f1 = f1_score(
                y[test_idx],
                y_pred,
                average="macro",
                labels=all_labels,
                zero_division=0,
            )

        cv_acc_mean = np.nan
        cv_f1_mean = np.nan
        if cv_splits >= 2:
            cv = GroupKFold(n_splits=cv_splits)
            cv_scores = cross_validate(
                clone(base_model),
                X,
                y,
                cv=cv.split(X, y, groups=groups),
                scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            )
            cv_acc_mean = float(np.mean(cv_scores["test_accuracy"]))
            cv_f1_mean = float(np.mean(cv_scores["test_f1_macro"]))

        rows.append(
            {
                "Model": name,
                "Training Accuracy": float(train_acc),
                "Holdout Accuracy": float(holdout_acc) if not np.isnan(holdout_acc) else np.nan,
                "Holdout Macro F1": float(holdout_f1) if not np.isnan(holdout_f1) else np.nan,
                "CV Accuracy (mean)": float(cv_acc_mean) if not np.isnan(cv_acc_mean) else np.nan,
                "CV Macro F1 (mean)": float(cv_f1_mean) if not np.isnan(cv_f1_mean) else np.nan,
            }
        )

    return pd.DataFrame(rows), notes


def train_models(df: pd.DataFrame, random_state: int) -> Tuple[
    pd.DataFrame,
    List[str],
    str,
    LabelEncoder,
    Dict[str, Any],
    LimeTabularExplainer,
    pd.DataFrame,
    List[str],
]:
    feature_cols = get_feature_columns(df)
    target_col = get_target_column(df)
    X = df[feature_cols].astype(int)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col].astype(str))

    models = build_models(random_state)
    eval_df, eval_notes = evaluate_models(models, X, y, random_state)

    fitted_models: Dict[str, Any] = {}
    for name, model in models.items():
        model.fit(X, y)
        fitted_models[name] = model

    lime_explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=feature_cols,
        class_names=label_encoder.classes_.tolist(),
        mode="classification",
        discretize_continuous=False,
    )

    return (
        X,
        feature_cols,
        target_col,
        label_encoder,
        fitted_models,
        lime_explainer,
        eval_df,
        eval_notes,
    )
