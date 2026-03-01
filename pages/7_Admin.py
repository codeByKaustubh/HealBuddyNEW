import io

import pandas as pd
import streamlit as st

from src.app_services import dataset_overview, init_usage_log, load_data_cached, train_models_cached
from src.config import DATA_PATH

st.set_page_config(page_title="HealBuddy | Admin", layout="wide")


def main() -> None:
    st.title("Admin Page")
    st.caption("Optional advanced controls for dataset upload, retraining, logs, and usage monitoring.")

    init_usage_log()

    st.subheader("Upload New Dataset")
    upload = st.file_uploader("Upload CSV with symptom columns and disease target", type=["csv"])
    if upload is not None:
        try:
            uploaded_df = pd.read_csv(upload)
            st.success("Dataset uploaded successfully.")
            st.dataframe(uploaded_df.head(20), use_container_width=True)
            st.session_state["admin_uploaded_df"] = uploaded_df
        except Exception as exc:
            st.error(f"Could not read uploaded dataset: {exc}")

    st.subheader("Retrain Models")
    if st.button("Retrain on Active Dataset", type="primary"):
        try:
            working_df = st.session_state.get("admin_uploaded_df", load_data_cached(DATA_PATH))
            (_, _, _, _, _, _, eval_df, _) = train_models_cached(working_df)
            st.success("Retraining completed.")
            st.dataframe(eval_df, use_container_width=True)
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
        st.write(overview)
    except Exception as exc:
        st.error(f"Could not compute dataset overview: {exc}")


if __name__ == "__main__":
    main()
