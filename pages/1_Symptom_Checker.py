from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder

from src.app_services import (
    get_confidence_and_risk,
    load_data_cached,
    record_prediction,
    softmax_probabilities,
    train_models_cached,
)
from src.config import DATA_PATH, MIN_REQUIRED_SYMPTOMS, PROBABILITY_THRESHOLD, TOP_N_PREDICTIONS
from src.data import make_input_vector, resolve_text_symptoms, suggest_closest_symptoms
from src.explainability import get_shap_values_for_class, plot_explanation_bar

st.set_page_config(page_title="HealBuddy | Symptom Checker", layout="wide")


def render_model_output(
    model_name: str,
    model: Any,
    x_user: np.ndarray,
    le: LabelEncoder,
    X: pd.DataFrame,
    feature_cols: List[str],
    lime_explainer: LimeTabularExplainer,
    selected_symptoms: List[str],
) -> None:
    probs = softmax_probabilities(model, x_user)
    ranked_idx = np.argsort(probs)[::-1]
    pred_idx = ranked_idx[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    pred_prob = float(probs[pred_idx])
    confidence_label, risk_label = get_confidence_and_risk(pred_prob)

    st.success(f"{model_name} prediction: **{pred_label}**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{pred_prob:.2%}")
    m2.metric("Confidence Level", confidence_label)
    m3.metric("Risk Category", risk_label)
    st.progress(min(max(pred_prob, 0.0), 1.0))

    top_rows: List[Tuple[str, float]] = []
    for idx in ranked_idx:
        prob = float(probs[idx])
        if prob < PROBABILITY_THRESHOLD:
            continue
        top_rows.append((le.inverse_transform([idx])[0], prob))
        if len(top_rows) >= TOP_N_PREDICTIONS:
            break

    if not top_rows:
        top_rows.append((pred_label, pred_prob))
        st.info("Only low-probability alternatives found. Showing best available prediction.")

    prob_df = pd.DataFrame(top_rows, columns=["Disease", "Probability"])
    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

    st.session_state["selected_disease"] = pred_label
    if st.button("Learn more", key=f"learn_more_{model_name}"):
        st.switch_page("pages/3_Disease_Information.py")

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
        except Exception as exc:
            st.error(f"Could not generate SHAP explanation: {exc}")

    with tab2:
        try:
            lime_exp = lime_explainer.explain_instance(
                x_user.astype(float),
                model.predict_proba,
                top_labels=1,
                num_features=min(10, len(feature_cols)),
            )
            lime_values = lime_exp.as_list(label=pred_idx)
            names = [name for name, _ in lime_values]
            vals = np.array([value for _, value in lime_values], dtype=float)
            lime_fig = plot_explanation_bar(names, vals, f"LIME local explanation for '{pred_label}'")
            st.pyplot(lime_fig)
        except Exception as exc:
            st.error(f"Could not generate LIME explanation: {exc}")

    record_prediction(model_name, pred_label, pred_prob, selected_symptoms)


def main() -> None:
    st.title("Symptom Checker")
    st.caption("Educational demo only. Not a medical diagnosis tool.")

    try:
        df = load_data_cached(DATA_PATH)
        (
            X,
            feature_cols,
            _target_col,
            le,
            fitted_models,
            lime_explainer,
            eval_df,
            _eval_notes,
        ) = train_models_cached(df)
    except Exception as exc:
        st.error(f"Failed to initialize app: {exc}")
        st.stop()

    st.subheader("Model Selection")
    model_names = list(fitted_models.keys())
    eval_by_model = eval_df.set_index("Model").to_dict(orient="index")
    selected_model_names: List[str] = st.multiselect(
        "Select one or more models",
        options=model_names,
        default=[model_names[0]],
    )

    row = st.columns(3)
    for idx, model_name in enumerate(model_names):
        with row[idx]:
            model_eval = eval_by_model.get(model_name, {})
            holdout_acc = model_eval.get("Holdout Accuracy", np.nan)
            st.metric(model_name, f"{holdout_acc:.2%}" if not pd.isna(holdout_acc) else "N/A")

    st.subheader("Symptom Selection")
    picked_symptoms = st.multiselect(
        "Choose symptoms from list",
        options=feature_cols,
        default=[],
    )
    typed_symptoms = st.text_input(
        "Or type symptoms (comma-separated)",
        placeholder="e.g., stomach pain, cough, fatigue",
        help="Typed terms are matched to known symptoms using normalization and aliases.",
    )

    if st.button("Reset Prediction"):
        for key in ["prediction_ready", "x_user", "selected_model_names", "active_model_idx", "selected_symptoms"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if st.button("Predict Disease", type="primary"):
        if not selected_model_names:
            st.error("Please select at least one model.")
            return

        resolved_symptoms, unmatched_symptoms = resolve_text_symptoms(typed_symptoms, feature_cols)
        merged_symptoms = sorted(set(picked_symptoms).union(set(resolved_symptoms)))

        if merged_symptoms:
            st.info(f"Recognized symptoms: {', '.join(merged_symptoms)}")
        if unmatched_symptoms:
            st.warning(f"Unrecognized typed symptoms: {', '.join(unmatched_symptoms)}")
            suggestions = suggest_closest_symptoms(unmatched_symptoms, feature_cols)
            if suggestions:
                suggestion_text = ", ".join([f"'{bad}' -> '{good}'" for bad, good in suggestions.items()])
                st.caption(f"Closest matches: {suggestion_text}")
        if len(merged_symptoms) < MIN_REQUIRED_SYMPTOMS:
            st.error(f"Please provide at least {MIN_REQUIRED_SYMPTOMS} recognized symptom(s).")
            return

        st.session_state["prediction_ready"] = True
        st.session_state["x_user"] = make_input_vector(feature_cols, merged_symptoms).tolist()
        st.session_state["selected_model_names"] = selected_model_names
        st.session_state["active_model_idx"] = 0
        st.session_state["selected_symptoms"] = merged_symptoms

    if st.session_state.get("prediction_ready", False):
        selected_models = st.session_state.get("selected_model_names", [])
        if not selected_models:
            st.warning("No models selected for display. Predict again after selecting models.")
            return

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
            selected_symptoms=st.session_state.get("selected_symptoms", []),
        )


if __name__ == "__main__":
    main()
