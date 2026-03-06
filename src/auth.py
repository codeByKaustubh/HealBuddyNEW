import hashlib
import json
import secrets
from pathlib import Path
from typing import Iterable, Optional, Tuple

import streamlit as st

from src.config import (
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USERNAME,
    DEFAULT_USER_PASSWORD,
    DEFAULT_USER_USERNAME,
)

USER_STORE_PATH = Path("user_accounts.json")
PBKDF2_ITERATIONS = 120_000


def _get_auth_config() -> dict:
    try:
        auth_cfg = st.secrets.get("auth", {})
        if isinstance(auth_cfg, dict):
            return auth_cfg
    except Exception:
        pass
    return {}


def _credential_store() -> dict:
    auth_cfg = _get_auth_config()
    return {
        "user": {
            "username": auth_cfg.get("user_username", DEFAULT_USER_USERNAME),
            "password": auth_cfg.get("user_password", DEFAULT_USER_PASSWORD),
        },
        "admin": {
            "username": auth_cfg.get("admin_username", DEFAULT_ADMIN_USERNAME),
            "password": auth_cfg.get("admin_password", DEFAULT_ADMIN_PASSWORD),
        },
    }


def _load_user_accounts() -> dict:
    if not USER_STORE_PATH.exists():
        return {}
    try:
        with USER_STORE_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw
    except Exception:
        return {}
    return {}


def _save_user_accounts(accounts: dict) -> None:
    with USER_STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=2)


def _hash_password(password: str, salt_hex: Optional[str] = None) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored_value: str) -> bool:
    try:
        salt_hex, expected_hex = stored_value.split("$", 1)
        computed = _hash_password(password, salt_hex=salt_hex)
        return computed.split("$", 1)[1] == expected_hex
    except Exception:
        return False


def create_user_account(username: str, password: str) -> Tuple[bool, str]:
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if " " in username:
        return False, "Username cannot contain spaces."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    credentials = _credential_store()
    reserved_usernames = {
        credentials["admin"]["username"],
        credentials["user"]["username"],
    }
    if username in reserved_usernames:
        return False, "This username is reserved. Please choose another."

    accounts = _load_user_accounts()
    if username in accounts:
        return False, "Username already exists. Please log in instead."

    accounts[username] = {
        "password_hash": _hash_password(password),
        "role": "user",
    }
    _save_user_accounts(accounts)
    return True, "Account created successfully. You can now log in."


def authenticate(username: str, password: str) -> Optional[str]:
    credentials = _credential_store()
    for role, creds in credentials.items():
        if username == creds["username"] and password == creds["password"]:
            return role

    accounts = _load_user_accounts()
    user_row = accounts.get(username)
    if user_row and _verify_password(password, user_row.get("password_hash", "")):
        return str(user_row.get("role", "user"))
    return None


def login_user(username: str, role: str) -> None:
    st.session_state["is_authenticated"] = True
    st.session_state["auth_username"] = username
    st.session_state["auth_role"] = role


def logout_user() -> None:
    for key in ["is_authenticated", "auth_username", "auth_role"]:
        st.session_state.pop(key, None)
    st.switch_page("app.py")


def current_role() -> Optional[str]:
    if not st.session_state.get("is_authenticated", False):
        return None
    return st.session_state.get("auth_role")


def render_auth_sidebar() -> None:
    if st.session_state.get("is_authenticated", False):
        username = st.session_state.get("auth_username", "unknown")
        role = st.session_state.get("auth_role", "unknown")

        if role == "user":
            st.sidebar.page_link("app.py", label="Home")
            st.sidebar.page_link("pages/1_Symptom_Checker.py", label="Symptom Checker")
            st.sidebar.page_link("pages/3_Disease_Information.py", label="Disease Information")
            st.sidebar.page_link("pages/5_About.py", label="About")
            st.sidebar.page_link("pages/6_Contact_Feedback.py", label="Contact Feedback")
        elif role == "admin":
            st.sidebar.page_link("pages/7_Admin.py", label="Admin Panel")

        st.sidebar.caption(f"Signed in as: {username} ({role})")
        if st.sidebar.button("Logout"):
            logout_user()


def require_roles(allowed_roles: Iterable[str]) -> None:
    if not st.session_state.get("is_authenticated", False):
        st.error("Authentication required. Please login first.")
        if st.button("Go to Login"):
            st.switch_page("app.py")
        st.stop()

    role = st.session_state.get("auth_role")
    if role not in set(allowed_roles):
        st.error("You do not have permission to access this page.")
        if st.button("Back to Login"):
            st.switch_page("app.py")
        st.stop()
