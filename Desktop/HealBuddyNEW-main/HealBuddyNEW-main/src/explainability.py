from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def get_shap_values_for_class(
    model: Any, X_background: pd.DataFrame, x_row: np.ndarray, class_idx: int
) -> np.ndarray:
    if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_row.reshape(1, -1))
    else:
        # For non-tree models (including Pipelines), operate on raw numpy arrays
        # to avoid compatibility issues with feature name handling.
        background_df = shap.sample(
            X_background, min(20, len(X_background)), random_state=42
        )
        background = background_df.values

        def predict_proba_fn(data):
            return model.predict_proba(np.asarray(data))

        explainer = shap.KernelExplainer(predict_proba_fn, background)
        shap_values = explainer.shap_values(
            x_row.reshape(1, -1).astype(float), nsamples=100
        )

    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx]).reshape(-1)

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[0, :, class_idx]
    if arr.ndim == 2:
        return arr[0]
    return arr.reshape(-1)


def plot_explanation_bar(names: List[str], values: np.ndarray, title: str):
    order = np.argsort(np.abs(values))[-10:]
    names_top = [names[i] for i in order]
    values_top = values[order]
    colors = ["#f97316" if v > 0 else "#3b82f6" for v in values_top]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(names_top, values_top, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Contribution")
    ax.set_ylabel("Symptom")
    fig.tight_layout()
    return fig
