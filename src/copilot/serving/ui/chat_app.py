"""Streamlit chat UI calling the FastAPI backend."""

from __future__ import annotations

import os

import httpx
import streamlit as st

from copilot.branding import render_submission_header

API_URL = os.environ.get("COPILOT_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("COPILOT_API_KEY", "")

# Get current page name for highlighting active link.
_CURRENT_PAGE = "chat"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _send_message(message: str, session_id: str | None) -> dict:
    """Send a chat message to the backend and return the JSON response."""
    with httpx.Client(timeout=300) as client:
        resp = client.post(
            f"{API_URL}/chat",
            headers={"x-api-key": API_KEY},
            json={"message": message, "session_id": session_id},
        )
        resp.raise_for_status()
        return resp.json()


def _send_feedback(data: dict, rating: str) -> None:
    """Send feedback for a previous answer."""
    with httpx.Client(timeout=30) as client:
        client.post(
            f"{API_URL}/feedback",
            headers={"x-api-key": API_KEY},
            json={
                "session_id": data["session_id"],
                "query": st.session_state.last_query,
                "answer": data["answer"],
                "rating": rating,
            },
        )


def _nav_link(label: str, page: str, icon: str, is_active: bool) -> str:
    """Return an HTML sidebar nav link with active state styling."""
    if page == "dashboard":
        port = 8502
    elif page == "upload":
        port = 8503
    else:
        port = 8501
    active = "background-color: rgba(255,255,255,0.1); border-radius: 4px;" if is_active else ""
    return f'<div style="padding: 0.25rem 0.5rem; margin: 0.1rem 0; {active}"><a href="http://localhost:{port}" target="_self" style="text-decoration: none; color: inherit; font-size: 1rem;">{icon} {label}</a></div>'


# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Support Copilot",
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🤖 Copilot")
st.sidebar.markdown(
    _nav_link("Chat", "chat_app", "💬", True) +
    _nav_link("Upload KB", "upload", "📤", False) +
    _nav_link("Dashboard", "dashboard", "📊", False),
    unsafe_allow_html=True,
)
st.sidebar.divider()
st.sidebar.caption(
    "Upload your own support documents on the **Upload KB** page "
    "to build a custom knowledge base, then ask questions here."
)

# Render the SPARKIIT-required submission header.
render_submission_header()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about your account, billing, or how-to guides..."):
    # Display user message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    st.session_state.last_query = prompt

    try:
        data = _send_message(prompt, st.session_state.session_id)
    except httpx.HTTPError:
        st.error("Sorry, the server encountered an error. Please try again.")
        st.stop()

    st.session_state.session_id = data["session_id"]
    answer = data["answer"]

    # Append citations if present.
    if data["citations"]:
        cites = "  \\n".join(f"[{c['marker']}] {c['title']}" for c in data["citations"])
        answer += f"\\n\\n---\\n**Sources:**  \\n{cites}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Feedback buttons.
    col1, col2, col3 = st.columns([1, 1, 4])
    if col1.button("👍  Helpful", key=f"up_{len(st.session_state.messages)}"):
        _send_feedback(data, "up")
        st.success("Thanks for the feedback!")
    if col2.button("👎  Not helpful", key=f"down_{len(st.session_state.messages)}"):
        _send_feedback(data, "down")
        st.error("Thanks for the feedback — we'll use it to improve.")
