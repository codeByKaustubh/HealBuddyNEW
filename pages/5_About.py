import streamlit as st

st.set_page_config(page_title="HealBuddy | About", layout="wide")


def main() -> None:
    st.title("About HealBuddy")

    st.subheader("Project Description")
    st.write(
        "HealBuddy is an AI-assisted symptom checker prototype designed to estimate probable diseases "
        "from user-reported symptoms and provide interpretable model outputs."
    )

    st.subheader("Technology Used")
    st.write(
        "The platform is built with Streamlit and Scikit-learn using three classifiers: "
        "Random Forest, Logistic Regression, and Naive Bayes. Explanations are generated using SHAP and LIME."
    )

    st.subheader("Research Motivation")
    st.write(
        "The app is intended to demonstrate transparent machine learning workflows in healthcare-style "
        "problem settings and to support educational experimentation with interpretable prediction systems."
    )

    st.subheader("Disclaimer")
    st.warning(
        "HealBuddy does not provide medical diagnosis, treatment, or emergency advice. "
        "Always consult qualified healthcare professionals for medical decisions."
    )


if __name__ == "__main__":
    main()
