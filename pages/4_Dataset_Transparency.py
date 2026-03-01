import pandas as pd
import streamlit as st

from src.app_services import dataset_overview, load_data_cached, train_models_cached
from src.config import DATA_PATH

st.set_page_config(page_title="HealBuddy | Dataset Transparency", layout="wide")


def main() -> None:
    st.title("Dataset Transparency")

    df = load_data_cached(DATA_PATH)
    overview = dataset_overview(df)
    (_, _, _, _, _, _, eval_df, eval_notes) = train_models_cached(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Number of Diseases", overview["num_diseases"])
    c2.metric("Number of Symptoms", overview["num_symptoms"])
    c3.metric("Dataset Rows", overview["num_rows"])

    st.subheader("Model Training Accuracy")
    st.dataframe(
        eval_df[["Model", "Training Accuracy", "Holdout Accuracy", "CV Accuracy (mean)"]].style.format(
            {"Training Accuracy": "{:.2%}", "Holdout Accuracy": "{:.2%}", "CV Accuracy (mean)": "{:.2%}"}
        ),
        use_container_width=True,
    )

    st.subheader("How the Dataset Was Built")
    st.write(
        "The app uses a structured symptom-to-disease dataset where each symptom column is binary "
        "(present/absent), and one target column stores disease labels. Duplicate symptom patterns are "
        "handled with grouped splitting during evaluation to reduce leakage."
    )

    st.subheader("Limitations")
    limitations = [
        "Small dataset size may limit generalization.",
        "Predictions depend on symptom wording and selected symptom coverage.",
        "No patient history, demographics, vitals, or lab reports are used.",
        "This tool is educational and not a medical diagnosis service.",
    ]
    st.write(pd.DataFrame({"Limitations": limitations}))

    for note in eval_notes:
        st.caption(note)


if __name__ == "__main__":
    main()
