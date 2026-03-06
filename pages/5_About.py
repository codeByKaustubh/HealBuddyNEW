import streamlit as st
from src.auth import render_auth_sidebar, require_roles

st.set_page_config(page_title="HealBuddy | About", layout="wide")


def main() -> None:
    require_roles(["user", "admin"])
    render_auth_sidebar()

    st.title("About HealBuddy")
    st.caption("Educational clinical decision-support demo")

    st.subheader("Brief Introduction")
    st.write(
        "HealBuddy is a symptom-checking web app that estimates possible conditions from user-reported "
        "symptoms using multiple machine learning models. It is designed for learning and early awareness, "
        "not for definitive diagnosis."
    )

    st.subheader("Purpose of the System")
    st.write(
        "This project was created to demonstrate how machine learning can support early health awareness "
        "through symptom-based risk estimation, model comparison, and explainable outputs."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Models Used", "3")
    c2.metric("Prediction Style", "Top 3 + Consensus")
    c3.metric("Explainability", "SHAP + LIME")

    st.subheader("How the Symptom Checker Works")
    st.markdown(
        "1. You enter symptoms.\n"
        "2. Each selected model predicts probable diseases.\n"
        "3. The app compares model outputs and generates a final consensus top-3 prediction.\n"
        "4. SHAP and LIME charts explain which symptoms influenced the final result."
    )

    st.subheader("Steps of the Process")
    st.markdown(
        "1. User enters symptoms\n"
        "2. System analyzes symptoms\n"
        "3. Possible diseases are matched\n"
        "4. Results and advice are displayed"
    )

    st.subheader("Key Features")
    st.markdown(
        "- Symptom input and analysis\n"
        "- Disease information database\n"
        "- Health tips and educational content\n"
        "- User-friendly interface with model comparison and explainability"
    )

    st.subheader("Technology Stack")
    st.write(
        "Frontend: Streamlit (Python-based UI)\n\n"
        "Backend: Python\n\n"
        "Database/Data Source: CSV medical symptom-disease dataset (`cleaned_dataset.csv`)\n\n"
        "APIs/Web Services: No external API dependency in the current version."
    )

    st.subheader("Dataset / Medical Information Source")
    st.write(
        "Predictions are generated from a structured symptom-disease dataset stored locally in the project. "
        "Disease descriptions and educational content are curated for demonstration purposes."
    )

    st.subheader("Limitations")
    st.info(
        "Predictions depend on dataset quality and symptom coverage. The app does not use medical history, "
        "lab tests, physical examination, or clinician judgment."
    )

    st.subheader("Medical Safety Disclaimer")
    st.warning(
        "HealBuddy does not provide diagnosis, treatment plans, or emergency triage. "
        "If symptoms are severe, worsening, or urgent, contact a licensed doctor or emergency services."
    )

    st.subheader("Project Team")
    st.write(
        "Developers / Team Members: HealBuddy Project Team\n\n"
        "College / Institution: RD and SH National College and SWA Science College"
    )


if __name__ == "__main__":
    main()
