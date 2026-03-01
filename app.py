import streamlit as st

st.set_page_config(page_title="HealBuddy | Home", layout="wide")


def main() -> None:
    st.title("AI-Based Disease Prediction System")
    st.write(
        "HealBuddy helps users check probable diseases from symptoms using trained machine "
        "learning models and transparent explanations."
    )

    st.subheader("How It Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Select Symptoms**")
        st.caption("Choose or type symptoms to build your input profile.")
    with c2:
        st.markdown("**2. Run Models**")
        st.caption("Compare predictions from Random Forest, Logistic Regression, and Naive Bayes.")
    with c3:
        st.markdown("**3. Review Explanations**")
        st.caption("Inspect confidence, risk level, SHAP, and LIME reasoning.")

    st.warning(
        "Disclaimer: This application is for educational support only and does not replace "
        "professional medical diagnosis or treatment."
    )

    if st.button("Start Symptom Check", type="primary"):
        st.switch_page("pages/1_Symptom_Checker.py")


if __name__ == "__main__":
    main()
