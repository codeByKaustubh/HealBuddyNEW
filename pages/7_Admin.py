import io

import numpy as np
import pandas as pd
import streamlit as st

from src.app_services import dataset_overview, init_usage_log, load_data_cached, train_models_cached
from src.config import DATA_PATH

st.set_page_config(page_title="HealBuddy | Admin", layout="wide")


def validate_dataset(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    disease_cols = [c for c in df.columns if c.strip().lower() == "disease"]
    if len(disease_cols) != 1:
        issues.append("Dataset must include exactly one 'Disease' column.")
        return issues

    disease_col = disease_cols[0]
    if df[disease_col].isna().any():
        issues.append("Disease column contains missing values.")
    if df[disease_col].nunique() < 2:
        issues.append("At least two unique diseases are required.")

    feature_cols = [c for c in df.columns if c != disease_col]
    if not feature_cols:
        issues.append("No symptom feature columns found.")
        return issues

    for col in feature_cols:
        col_vals = df[col].dropna()
        if col_vals.empty:
            issues.append(f"Symptom column '{col}' is empty.")
            continue
        unique_vals = set(np.unique(col_vals))
        if not unique_vals.issubset({0, 1}):
            issues.append(f"Symptom column '{col}' must be binary (0/1).")
            break

    return issues


def main() -> None:
    st.title("Admin Console")
    st.caption("Manage datasets, retrain models, and monitor prediction usage.")

    init_usage_log()

    st.subheader("Upload New Dataset")
    upload = st.file_uploader("Upload CSV with symptom columns and disease target", type=["csv"])
    if upload is not None:
        try:
            uploaded_df = pd.read_csv(upload)
            validation_issues = validate_dataset(uploaded_df)
            if validation_issues:
                st.error("Dataset uploaded but validation failed.")
                for issue in validation_issues:
                    st.caption(f"- {issue}")
            else:
                st.success("Dataset uploaded and validated successfully.")
                st.session_state["admin_uploaded_df"] = uploaded_df
            st.dataframe(uploaded_df.head(20), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read uploaded dataset: {exc}")

    st.subheader("Retrain Models")
    if st.button("Retrain on Active Dataset", type="primary"):
        try:
            working_df = st.session_state.get("admin_uploaded_df", load_data_cached(DATA_PATH))
            validation_issues = validate_dataset(working_df)
            if validation_issues:
                st.error("Retraining blocked due to dataset validation issues.")
                for issue in validation_issues:
                    st.caption(f"- {issue}")
                return

            (_, _, _, _, _, _, eval_df, _) = train_models_cached(working_df)
            st.success("Retraining completed.")
            ranking_df = eval_df.sort_values(
                by=["CV Macro F1 (mean)", "Holdout Macro F1"],
                ascending=False,
                na_position="last",
            ).copy()
            st.dataframe(
                ranking_df.style.format(
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
        except Exception as exc:
            st.error(f"Retraining failed: {exc}")

    st.subheader("Prediction Usage Monitor")
    logs = st.session_state.get("prediction_logs", [])
    if not logs:
        st.info("No prediction usage logs yet.")
    else:
        logs_df = pd.DataFrame(logs)
        st.metric("Total Predictions", len(logs_df))
        st.dataframe(logs_df, use_container_width=True)
        st.bar_chart(logs_df["Model"].value_counts())

        buffer = io.StringIO()
        logs_df.to_csv(buffer, index=False)
        st.download_button(
            "Download Logs CSV",
            data=buffer.getvalue(),
            file_name="prediction_logs.csv",
            mime="text/csv",
        )

    st.subheader("Current Dataset Overview")
    try:
        current_df = st.session_state.get("admin_uploaded_df", load_data_cached(DATA_PATH))
        overview = dataset_overview(current_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", overview["num_rows"])
        c2.metric("Diseases", overview["num_diseases"])
        c3.metric("Symptoms", overview["num_symptoms"])
        st.caption(f"Example diseases: {', '.join(overview['disease_examples'])}")
    except Exception as exc:
        st.error(f"Could not compute dataset overview: {exc}")


if __name__ == "__main__":
    main()
