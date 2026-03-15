import streamlit as st
from src.auth import authenticate, create_user_account, current_role, login_user, render_auth_sidebar

st.set_page_config(page_title="HealBuddy | Home", layout="wide")


def hide_sidebar_for_auth_pages() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login() -> None:
    st.title("HealBuddy Authentication")
    st.write("Choose an option: log in to an existing account or create a new user account.")

    login_tab, register_tab = st.tabs(["Log In", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")

        if submitted:
            role = authenticate(username.strip(), password)
            if role is None:
                st.error("Invalid credentials. Please try again.")
            else:
                login_user(username=username.strip(), role=role)
                st.success(f"Login successful. Redirecting to {role} panel...")
                if role == "admin":
                    st.switch_page("pages/7_Admin.py")
                st.switch_page("pages/1_Symptom_Checker.py")

    with register_tab:
        st.caption("New registrations create user accounts. Admin accounts are managed separately.")
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_submitted = st.form_submit_button("Create Account", type="primary")

        if register_submitted:
            if new_password != confirm_password:
                st.error("Password and confirm password do not match.")
            else:
                ok, message = create_user_account(new_username, new_password)
                if ok:
                    st.success(message)
                else:
                    st.error(message)


def render_user_home() -> None:
    st.title("HealBuddy Clinical Triage Assistant")
    st.subheader("Website Introduction")
    st.write(
        "HealBuddy is an educational health-tech web app that estimates likely conditions "
        "from symptoms using multiple machine learning models. It is designed to help users "
        "understand possible next steps, not to replace clinical diagnosis."
    )

    st.subheader("How the Symptom Checker Works")
    st.write(
        "You provide symptoms, the app compares patterns using selected models, and then "
        "shows model-wise predictions plus a final consensus with explainability charts."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Enter Symptoms**")
        st.caption("Select symptoms from the list and optionally type additional symptoms.")
    with c2:
        st.markdown("**2. Compare Models**")
        st.caption("Selected models generate top predictions with confidence and risk labels.")
    with c3:
        st.markdown("**3. Review Final Result**")
        st.caption("A consensus top-3 prediction is shown along with SHAP and LIME explanations.")

    st.subheader("Basic Health Tips")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.info("Stay hydrated, sleep 7-9 hours, and avoid skipping meals when unwell.")
    with t2:
        st.info("Monitor red-flag symptoms like chest pain, breathing difficulty, or confusion.")
    with t3:
        st.info("If symptoms persist, worsen, or feel severe, consult a licensed doctor promptly.")

    st.warning(
        "Disclaimer: This application is for educational support only and does not replace "
        "professional medical diagnosis, emergency triage, or treatment."
    )

    if st.button("Start Symptom Check", type="primary"):
        st.switch_page("pages/1_Symptom_Checker.py")


def main() -> None:
    if not st.session_state.get("is_authenticated", False):
        hide_sidebar_for_auth_pages()
        render_login()
        return

    render_auth_sidebar()
    role = current_role()
    if role == "admin":
        st.switch_page("pages/7_Admin.py")
    render_user_home()


if __name__ == "__main__":
    main()
