from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="HealBuddy | Contact & Feedback", layout="wide")

FEEDBACK_FILE = Path("feedback_log.csv")


def _append_feedback(feedback_row: dict) -> bool:
    row_df = pd.DataFrame([feedback_row])
    if FEEDBACK_FILE.exists():
        existing = pd.read_csv(FEEDBACK_FILE)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(FEEDBACK_FILE, index=False)
    return True


def main() -> None:
    st.title("Contact / Feedback")
    st.write("Use this page to report incorrect predictions, suggest improvements, or share general feedback.")

    with st.form("feedback_form", clear_on_submit=True):
        feedback_type = st.selectbox(
            "Feedback type",
            options=["Incorrect prediction", "Improvement suggestion", "General feedback"],
        )
        email = st.text_input("Email (optional)")
        message = st.text_area("Your feedback", height=180)
        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        if not message.strip():
            st.error("Please provide details before submitting.")
            return
        row = {
            "Feedback Type": feedback_type,
            "Email": email.strip(),
            "Message": message.strip(),
        }
        try:
            _append_feedback(row)
            st.success("Thanks. Your feedback has been recorded.")
        except Exception as exc:
            st.error(f"Could not save feedback locally: {exc}")
            st.write("You can still copy this feedback text and share it manually:")
            st.code(message.strip())


if __name__ == "__main__":
    main()
