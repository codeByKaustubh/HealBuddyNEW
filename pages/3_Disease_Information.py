import streamlit as st

from src.app_services import get_disease_content, load_data_cached
from src.config import DATA_PATH

st.set_page_config(page_title="HealBuddy | Disease Information", layout="wide")


def main() -> None:
    st.title("Disease Information")

    df = load_data_cached(DATA_PATH)
    disease_col = [c for c in df.columns if c.strip().lower() == "disease"][0]
    all_diseases = sorted(df[disease_col].astype(str).unique().tolist())
    default_disease = st.session_state.get("selected_disease", all_diseases[0])

    selected_disease = st.selectbox(
        "Select disease",
        options=all_diseases,
        index=all_diseases.index(default_disease) if default_disease in all_diseases else 0,
    )
    st.session_state["selected_disease"] = selected_disease

    content = get_disease_content(selected_disease)

    st.subheader(selected_disease)
    st.markdown("**Description**")
    st.write(content["description"])
    st.markdown("**Causes**")
    st.write(content["causes"])
    st.markdown("**Symptoms**")
    st.write(content["symptoms"])
    st.markdown("**When to see a doctor**")
    st.write(content["doctor"])
    st.markdown("**Prevention tips**")
    st.write(content["prevention"])


if __name__ == "__main__":
    main()
