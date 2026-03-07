from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from src.app_services import (
    compute_similarity_probabilities,
    consensus_probabilities,
    get_confidence_and_risk,
    hybrid_probabilities,
    load_data_cached,
    record_prediction,
    train_models_cached,
)
from src.auth import render_auth_sidebar, require_roles
from src.config import (
    DATA_PATH,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_REQUIRED_SYMPTOMS,
    MODEL_PROBABILITY_WEIGHT,
    TOP_N_PREDICTIONS,
)
from src.data import (
    make_input_vector,
    resolve_text_symptoms_with_spellcheck,
    suggest_closest_symptoms,
)
from src.explainability import get_shap_values_for_class, plot_explanation_bar

st.set_page_config(page_title="HealBuddy | Symptom Checker", layout="wide")


def compute_model_probabilities(
    model: Any,
    x_user: np.ndarray,
    n_classes: int,
    X: pd.DataFrame,
    selected_symptoms: List[str],
    y_encoded: np.ndarray,
) -> np.ndarray:
    model_hybrid_probs = hybrid_probabilities(
        model=model,
        x_row=x_user,
        X_train=X,
        y_train=y_encoded,
        n_classes=n_classes,
        model_weight=MODEL_PROBABILITY_WEIGHT,
    )
    similarity_probs = compute_similarity_probabilities(x_user, X, y_encoded, n_classes)
    probs = consensus_probabilities(
        model_probs=model_hybrid_probs,
        similarity_probs=similarity_probs,
        n_selected_symptoms=len(selected_symptoms),
        model_weight=MODEL_PROBABILITY_WEIGHT,
    )

    exact_pattern_mask = np.all(X.values == x_user.reshape(1, -1), axis=1)
    if np.any(exact_pattern_mask):
        exact_labels = y_encoded[exact_pattern_mask]
        exact_counts = np.bincount(exact_labels, minlength=n_classes).astype(float)
        exact_probs = exact_counts / exact_counts.sum()
        probs = 0.7 * exact_probs + 0.3 * probs

    return probs


def top_diseases_from_probs(probs: np.ndarray, le: LabelEncoder, top_n: int) -> List[Tuple[str, float]]:
    ranked_idx = np.argsort(probs)[::-1]
    rows: List[Tuple[str, float]] = []
    for idx in ranked_idx[:top_n]:
        rows.append((le.inverse_transform([idx])[0], float(probs[idx])))
    return rows


def main() -> None:
    require_roles(["user", "admin"])
    render_auth_sidebar()

    st.title("Symptom Checker")
    st.caption("Educational demo only. Not a medical diagnosis tool.")

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
            _eval_notes,
        ) = train_models_cached(df)
    except Exception as exc:
        st.error(f"Failed to initialize app: {exc}")
        st.stop()
    y_encoded = le.transform(df[target_col].astype(str))

    st.subheader("Model Selection")
    model_names = list(fitted_models.keys())
    best_model_row = eval_df.sort_values(
        by=["CV Macro F1 (mean)", "Holdout Macro F1"],
        ascending=False,
        na_position="last",
    ).iloc[0]
    default_model_name = str(best_model_row["Model"]) if not eval_df.empty else model_names[0]
    st.caption("Select one or more models")
    selected_model_names: List[str] = []
    model_cols = st.columns(len(model_names))
    for i, model_name in enumerate(model_names):
        with model_cols[i]:
            is_checked = st.checkbox(
                model_name,
                value=(model_name == default_model_name),
                key=f"model_checkbox_{model_name}",
            )
            if is_checked:
                selected_model_names.append(model_name)

    st.subheader("Symptom Selection")
    selected_symptoms = st.multiselect(
        "Select symptoms from the list",
        options=feature_cols,
        help="Select all currently observed symptoms. You can also add typed symptoms.",
    )
    typed_symptoms = st.text_input(
        "Or type symptoms (comma-separated)",
        placeholder="e.g., stomach pain, cough, fatigue",
        help="Typed terms are matched to known symptoms using normalization and aliases.",
    )

    if st.button("Reset Prediction"):
        for key in ["prediction_ready", "x_user", "selected_model_names", "selected_symptoms"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if st.button("Predict Disease", type="primary"):
        if not selected_model_names:
            st.error("Please select at least one model.")
            return

        resolved_symptoms, unmatched_symptoms, spelling_corrections = (
            resolve_text_symptoms_with_spellcheck(typed_symptoms, feature_cols)
        )
        merged_symptoms = sorted(set(selected_symptoms).union(resolved_symptoms))

        if merged_symptoms:
            st.info(f"Recognized symptoms: {', '.join(merged_symptoms)}")
        if spelling_corrections:
            correction_text = ", ".join([f"'{bad}' -> '{good}'" for bad, good in spelling_corrections.items()])
            st.caption(f"Dictionary spell-corrections applied: {correction_text}")
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
        st.session_state["selected_symptoms"] = merged_symptoms

    if st.session_state.get("prediction_ready", False):
        selected_models = st.session_state.get("selected_model_names", [])
        if not selected_models:
            st.warning("No models selected for display. Predict again after selecting models.")
            return

        x_user = np.array(st.session_state["x_user"], dtype=int)
        active_symptoms = st.session_state.get("selected_symptoms", [])

        st.subheader("Model Comparison")
        comparison_rows: List[Dict[str, Any]] = []
        per_model_probs: List[np.ndarray] = []
        model_top3: Dict[str, List[Tuple[str, float]]] = {}
        model_probs_map: Dict[str, np.ndarray] = {}
        for model_name in selected_models:
            probs = compute_model_probabilities(
                model=fitted_models[model_name],
                x_user=x_user,
                n_classes=len(le.classes_),
                X=X,
                selected_symptoms=active_symptoms,
                y_encoded=y_encoded,
            )
            per_model_probs.append(probs)
            model_probs_map[model_name] = probs
            top_rows = top_diseases_from_probs(probs, le, TOP_N_PREDICTIONS)
            model_top3[model_name] = top_rows
            top_disease, top_prob = top_rows[0]
            confidence_label, risk_label = get_confidence_and_risk(top_prob)

            comparison_rows.append(
                {
                    "Model": model_name,
                    "Top Prediction": top_disease,
                    "Confidence": top_prob,
                    "Confidence Level": confidence_label,
                    "Risk Category": risk_label,
                    "Top 3": ", ".join([f"{d} ({p:.1%})" for d, p in top_rows]),
                }
            )
            record_prediction(model_name, top_disease, top_prob, active_symptoms)

        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(
            comparison_df.style.format({"Confidence": "{:.2%}"}),
            use_container_width=True,
        )

        mean_probs = np.mean(np.vstack(per_model_probs), axis=0)
        final_top = top_diseases_from_probs(mean_probs, le, TOP_N_PREDICTIONS)
        final_label, final_prob = final_top[0]
        final_confidence_label, final_risk_label = get_confidence_and_risk(final_prob)

        st.subheader("Final Prediction (Model Consensus)")
        st.success(f"Most likely disease: **{final_label}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Consensus Confidence", f"{final_prob:.2%}")
        m2.metric("Confidence Level", final_confidence_label)
        m3.metric("Risk Category", final_risk_label)
        st.progress(min(max(final_prob, 0.0), 1.0))
        if final_prob < LOW_CONFIDENCE_THRESHOLD:
            st.warning("Consensus confidence is low. Add more symptoms for a stronger final prediction.")

        final_df = pd.DataFrame(final_top, columns=["Disease", "Probability"])
        st.dataframe(final_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

        support_rows: List[Dict[str, Any]] = []
        for disease, prob in final_top:
            supporting_models = [
                model_name
                for model_name, top3 in model_top3.items()
                if disease in [d for d, _ in top3]
            ]
            support_rows.append(
                {
                    "Disease": disease,
                    "Consensus Probability": prob,
                    "Models Supporting (Top 3)": len(supporting_models),
                    "Supporting Models": ", ".join(supporting_models) if supporting_models else "None",
                }
            )

        st.info(
            "Why these final top 3? The app averages probability distributions from your selected models "
            "to form a consensus score, then ranks diseases by that score."
        )
        st.dataframe(
            pd.DataFrame(support_rows).style.format({"Consensus Probability": "{:.2%}"}),
            use_container_width=True,
        )

        final_idx = int(le.transform([final_label])[0])
        explainer_model_name = max(
            selected_models,
            key=lambda name: float(model_probs_map[name][final_idx]),
        )
        explainer_model = fitted_models[explainer_model_name]

        st.subheader("Final Prediction Explainability")
        st.caption(
            f"SHAP and LIME are generated using **{explainer_model_name}**, "
            "the selected model with the strongest support for the final predicted disease."
        )
        exp_tab1, exp_tab2 = st.tabs(["SHAP Diagram", "LIME Diagram"])

        with exp_tab1:
            try:
                shap_vals = get_shap_values_for_class(explainer_model, X, x_user, final_idx)
                shap_fig = plot_explanation_bar(
                    feature_cols,
                    shap_vals,
                    f"SHAP contributions toward '{final_label}'",
                )
                st.pyplot(shap_fig)
                st.caption("Orange bars push prediction toward the class, blue bars push away.")
            except Exception as exc:
                st.error(f"Could not generate SHAP explanation: {exc}")

        with exp_tab2:
            try:
                lime_exp = _lime_explainer.explain_instance(
                    x_user.astype(float),
                    explainer_model.predict_proba,
                    labels=(final_idx,),
                    num_features=min(10, len(feature_cols)),
                )
                lime_values = lime_exp.as_list(label=final_idx)
                names = [name for name, _ in lime_values]
                vals = np.array([value for _, value in lime_values], dtype=float)
                lime_fig = plot_explanation_bar(names, vals, f"LIME local explanation for '{final_label}'")
                st.pyplot(lime_fig)
            except Exception as exc:
                st.error(f"Could not generate LIME explanation: {exc}")

        st.session_state["selected_disease"] = final_label
        if st.button("Learn more about final prediction", key="learn_more_final"):
            st.switch_page("pages/3_Disease_Information.py")


if __name__ == "__main__":
    main()
