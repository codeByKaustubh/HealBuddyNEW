from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_services import dataset_overview, load_data_cached, train_models_cached
from src.auth import render_auth_sidebar, require_roles
from src.config import DATA_PATH

st.set_page_config(page_title="HealBuddy | Admin Panel", layout="wide")

FEEDBACK_FILE = Path("feedback_log.csv")


def main() -> None:
    require_roles(["admin"])
    render_auth_sidebar()

    st.title("Admin Panel")
    st.caption("Restricted access: dataset status, model evaluation, and feedback review.")

    st.subheader("Dataset Overview")
    try:
        df = load_data_cached(DATA_PATH)
        overview = dataset_overview(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", overview["num_rows"])
        c2.metric("Diseases", overview["num_diseases"])
        c3.metric("Symptoms", overview["num_symptoms"])
        st.caption(f"Sample diseases: {', '.join(overview['disease_examples'])}")
    except Exception as exc:
        st.error(f"Could not load dataset overview: {exc}")
        return

    st.subheader("Model Evaluation")
    try:
        (_, _, _, _, _, _, eval_df, eval_notes) = train_models_cached(df)
        ranking_df = eval_df.sort_values(
            by=["CV Macro F1 (mean)", "Holdout Macro F1"],
            ascending=False,
            na_position="last",
        )
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
        for note in eval_notes:
            st.caption(note)
    except Exception as exc:
        st.error(f"Could not evaluate models: {exc}")

    st.subheader("Feedback Inbox")
    if FEEDBACK_FILE.exists():
        try:
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            st.metric("Total Feedback Entries", len(feedback_df))
            st.dataframe(feedback_df, use_container_width=True)
            st.download_button(
                "Download Feedback CSV",
                data=feedback_df.to_csv(index=False),
                file_name="feedback_log.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not read feedback log: {exc}")
    else:
        st.info("No feedback entries yet.")


if __name__ == "__main__":
    main()
