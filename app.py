import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATA_PATH,
    MIN_REQUIRED_SYMPTOMS,
    PROBABILITY_THRESHOLD,
    RANDOM_STATE,
    TOP_N_PREDICTIONS,
)
from src.data import load_data as _load_data
from src.data import make_input_vector
from src.data import resolve_text_symptoms
from src.data import suggest_closest_symptoms
from src.explainability import get_shap_values_for_class, plot_explanation_bar
from src.models import train_models as _train_models

warnings.filterwarnings("ignore")

st.set_page_config(page_title="HealBuddy Symptom Checker", layout="wide")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return _load_data(path)


@st.cache_resource
def train_models(df: pd.DataFrame):
    return _train_models(df, random_state=RANDOM_STATE)


def render_model_output(
    model_name: str,
    model: Any,
    x_user: np.ndarray,
    le: LabelEncoder,
    X: pd.DataFrame,
    feature_cols: List[str],
    lime_explainer: LimeTabularExplainer,
):
    probs = model.predict_proba(x_user.reshape(1, -1))[0]
    ranked_idx = np.argsort(probs)[::-1]
    pred_idx = ranked_idx[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    st.success(f"{model_name} prediction: **{pred_label}**")

    top_rows: List[Tuple[str, float]] = []
    for idx in ranked_idx:
        prob = float(probs[idx])
        if prob < PROBABILITY_THRESHOLD:
            continue
        top_rows.append((le.inverse_transform([idx])[0], prob))
        if len(top_rows) >= TOP_N_PREDICTIONS:
            break

    if not top_rows:
        top_rows.append((pred_label, float(probs[pred_idx])))
        st.info("Only low-probability alternatives found. Showing best available prediction.")

    prob_df = pd.DataFrame(top_rows, columns=["Disease", "Probability"])
    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

    tab1, tab2 = st.tabs(["SHAP Explanation", "LIME Explanation"])

    with tab1:
        try:
            shap_vals = get_shap_values_for_class(model, X, x_user, pred_idx)
            shap_fig = plot_explanation_bar(
                feature_cols,
                shap_vals,
                f"SHAP contributions toward '{pred_label}'",
            )
            st.pyplot(shap_fig)
            st.caption("Orange bars push prediction toward the class, blue bars push away.")
        except Exception as e:
            st.error(f"Could not generate SHAP explanation: {e}")

    with tab2:
        try:
            lime_exp = lime_explainer.explain_instance(
                x_user.astype(float),
                model.predict_proba,
                top_labels=1,
                num_features=min(10, len(feature_cols)),
            )
            lime_values = lime_exp.as_list(label=pred_idx)
            names = [n for n, _ in lime_values]
            vals = np.array([v for _, v in lime_values], dtype=float)

            lime_fig = plot_explanation_bar(names, vals, f"LIME local explanation for '{pred_label}'")
            st.pyplot(lime_fig)
        except Exception as e:
            st.error(f"Could not generate LIME explanation: {e}")


def main():
    st.title("HealBuddy Symptom Checker")
    st.caption("Educational demo only. Not a medical diagnosis tool.")

    try:
        df = load_data(DATA_PATH)
        (
            X,
            feature_cols,
            _target_col,
            le,
            fitted_models,
            lime_explainer,
            eval_df,
            eval_notes,
        ) = train_models(df)
    except Exception as e:
        st.error(f"Failed to initialize app: {e}")
        st.stop()

    st.subheader("Suggested Models")
    cols = st.columns(3)
    model_names = list(fitted_models.keys())
    eval_by_model = eval_df.set_index("Model").to_dict(orient="index")

    for i, model_name in enumerate(model_names):
        with cols[i]:
            model_eval = eval_by_model.get(model_name, {})
            holdout_acc = model_eval.get("Holdout Accuracy", np.nan)
            holdout_f1 = model_eval.get("Holdout Macro F1", np.nan)
            cv_f1 = model_eval.get("CV Macro F1 (mean)", np.nan)
            st.markdown(f"**{model_name}**")
            st.write(
                "Holdout accuracy:",
                f"`{holdout_acc:.2%}`" if not pd.isna(holdout_acc) else "`N/A`",
            )
            st.write(
                "Holdout macro F1:",
                f"`{holdout_f1:.2%}`" if not pd.isna(holdout_f1) else "`N/A`",
            )
            st.write(
                "CV macro F1 (mean):",
                f"`{cv_f1:.2%}`" if not pd.isna(cv_f1) else "`N/A`",
            )

    st.markdown("**Select one or more models to run**")
    selected_model_names: List[str] = []
    model_pick_cols = st.columns(3)
    for i, model_name in enumerate(model_names):
        with model_pick_cols[i]:
            if st.checkbox(model_name, value=(i == 0), key=f"model_check_{model_name}"):
                selected_model_names.append(model_name)

    st.subheader("Select Symptoms")
    typed_symptoms = st.text_input(
        "Type symptoms (comma-separated)",
        placeholder="e.g., stomach pain, cough, fatigue",
        help="Typed terms are matched to known symptoms using normalization and aliases.",
    )
    if st.button("Reset Prediction"):
        for key in ["prediction_ready", "x_user", "selected_model_names", "active_model_idx"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if st.button("Predict Disease", type="primary"):
        if not selected_model_names:
            st.error("Please select at least one model.")
        else:
            resolved_symptoms, unmatched_symptoms = resolve_text_symptoms(typed_symptoms, feature_cols)
            if resolved_symptoms:
                st.info(f"Recognized symptoms: {', '.join(resolved_symptoms)}")
            if unmatched_symptoms:
                st.warning(f"Unrecognized typed symptoms: {', '.join(unmatched_symptoms)}")
                suggestions = suggest_closest_symptoms(unmatched_symptoms, feature_cols)
                if suggestions:
                    suggestion_text = ", ".join(
                        [f"'{bad}' -> '{good}'" for bad, good in suggestions.items()]
                    )
                    st.caption(f"Closest matches: {suggestion_text}")
            if len(resolved_symptoms) < MIN_REQUIRED_SYMPTOMS:
                st.error(
                    f"Please type at least {MIN_REQUIRED_SYMPTOMS} recognized symptom(s)."
                )
                return

            st.session_state["prediction_ready"] = True
            st.session_state["x_user"] = make_input_vector(feature_cols, resolved_symptoms).tolist()
            st.session_state["selected_model_names"] = selected_model_names
            st.session_state["active_model_idx"] = 0

    if st.session_state.get("prediction_ready", False):
        selected_models = st.session_state.get("selected_model_names", [])
        if not selected_models:
            st.warning("No models selected for display. Predict again after selecting models.")
        else:
            if st.session_state.get("active_model_idx", 0) >= len(selected_models):
                st.session_state["active_model_idx"] = 0

            nav_left, nav_center, nav_right = st.columns([1, 2, 1])
            with nav_left:
                if st.button("Previous Model", key="prev_model_btn"):
                    st.session_state["active_model_idx"] = (
                        st.session_state["active_model_idx"] - 1
                    ) % len(selected_models)
            with nav_right:
                if st.button("Next Model", key="next_model_btn"):
                    st.session_state["active_model_idx"] = (
                        st.session_state["active_model_idx"] + 1
                    ) % len(selected_models)

            current_name = selected_models[st.session_state["active_model_idx"]]
            x_user = np.array(st.session_state["x_user"], dtype=int)
            with nav_center:
                st.markdown(
                    f"### Model {st.session_state['active_model_idx'] + 1}/{len(selected_models)}: {current_name}"
                )

            render_model_output(
                model_name=current_name,
                model=fitted_models[current_name],
                x_user=x_user,
                le=le,
                X=X,
                feature_cols=feature_cols,
                lime_explainer=lime_explainer,
            )

if __name__ == "__main__":
    main()
