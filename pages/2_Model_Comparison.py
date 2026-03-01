from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from src.app_services import load_data_cached, train_models_cached
from src.config import DATA_PATH, RANDOM_STATE
from src.data import make_input_vector

st.set_page_config(page_title="HealBuddy | Model Comparison", layout="wide")


def _holdout_predictions(
    model_defs: Dict[str, Any], X: pd.DataFrame, y: np.ndarray, random_state: int
) -> Dict[str, Dict[str, Any]]:
    groups = X.astype(str).agg("|".join, axis=1).values
    unique_group_count = len(np.unique(groups))
    if unique_group_count < 2:
        return {}

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    result: Dict[str, Dict[str, Any]] = {}
    for name, base_model in model_defs.items():
        model = clone(base_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result[name] = {"y_true": y_test, "y_pred": y_pred}
    return result


def _compute_shap_importance(
    X: pd.DataFrame,
    fitted_models: Dict[str, Any],
    feature_cols: List[str],
) -> pd.DataFrame:
    sample = X.sample(min(10, len(X)), random_state=RANDOM_STATE)
    background = shap.sample(X, min(20, len(X)), random_state=RANDOM_STATE)
    rows = []

    for model_name, model in fitted_models.items():
        if model_name == "Random Forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(sample, nsamples=80)

        if isinstance(shap_values, list):
            arr = np.stack([np.abs(np.asarray(sv)) for sv in shap_values], axis=0)
            mean_abs = arr.mean(axis=(0, 1))
        else:
            arr = np.abs(np.asarray(shap_values))
            if arr.ndim == 3:
                mean_abs = arr.mean(axis=(0, 2))
            else:
                mean_abs = arr.mean(axis=0)

        for feature, value in zip(feature_cols, mean_abs):
            rows.append({"Model": model_name, "Feature": feature, "Mean |SHAP|": float(value)})

    return pd.DataFrame(rows)


def main() -> None:
    st.title("Model Comparison")

    try:
        df = load_data_cached(DATA_PATH)
        (
            X,
            feature_cols,
            target_col,
            le,
            fitted_models,
            _lime_explainer,
            eval_df,
            eval_notes,
        ) = train_models_cached(df)
    except Exception as exc:
        st.error(f"Failed to initialize model comparison: {exc}")
        st.stop()

    model_names = list(fitted_models.keys())
    st.subheader("Accuracy Snapshot")
    st.dataframe(
        eval_df[
            [
                "Model",
                "Training Accuracy",
                "Holdout Accuracy",
                "Holdout Macro F1",
                "CV Accuracy (mean)",
                "CV Macro F1 (mean)",
            ]
        ].style.format(
            {
                "Training Accuracy": "{:.2%}",
                "Holdout Accuracy": "{:.2%}",
                "Holdout Macro F1": "{:.2%}",
                "CV Accuracy (mean)": "{:.2%}",
                "CV Macro F1 (mean)": "{:.2%}",
            }
        ),
        use_container_width=True,
    )
    for note in eval_notes:
        st.caption(note)

    st.subheader("Probability Charts")
    selected_symptoms = st.multiselect("Choose symptoms for comparison input", options=feature_cols, default=[])
    if selected_symptoms:
        x_user = make_input_vector(feature_cols, selected_symptoms)
        prob_chart_rows = []
        for model_name in model_names:
            model = fitted_models[model_name]
            probs = model.predict_proba(x_user.reshape(1, -1))[0]
            pred_idx = int(np.argmax(probs))
            prob_chart_rows.append(
                {
                    "Model": model_name,
                    "Predicted Disease": le.inverse_transform([pred_idx])[0],
                    "Probability": float(probs[pred_idx]),
                }
            )
        prob_chart_df = pd.DataFrame(prob_chart_rows)
        st.dataframe(prob_chart_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)
        st.bar_chart(prob_chart_df.set_index("Model")["Probability"])
    else:
        st.info("Select at least one symptom to generate model probability charts.")

    st.subheader("Confusion Matrix (Holdout)")
    holdout = _holdout_predictions(
        fitted_models,
        X,
        le.transform(df[target_col].astype(str)),
        RANDOM_STATE,
    )
    if not holdout:
        st.warning("Not enough unique grouped samples to compute a holdout confusion matrix.")
    else:
        fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
        axes = np.atleast_1d(axes)
        labels = np.arange(len(le.classes_))

        for idx, model_name in enumerate(model_names):
            data = holdout[model_name]
            cm = confusion_matrix(data["y_true"], data["y_pred"], labels=labels, normalize="true")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(ax=axes[idx], colorbar=False, xticks_rotation=45, values_format=".2f")
            axes[idx].set_title(model_name)
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("SHAP Summary Comparison")
    with st.spinner("Computing SHAP summary across models..."):
        shap_df = _compute_shap_importance(X, fitted_models, feature_cols)
    for model_name in model_names:
        top = shap_df[shap_df["Model"] == model_name].nlargest(10, "Mean |SHAP|")
        st.markdown(f"**{model_name}**")
        st.bar_chart(top.set_index("Feature")["Mean |SHAP|"])


if __name__ == "__main__":
    main()
