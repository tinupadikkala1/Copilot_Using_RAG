"""SPARKIIT submission branding — renders identity on landing pages."""

from __future__ import annotations

import streamlit as st

from copilot.config import get_settings

_settings = get_settings()

SUBMISSION = {
    "project_topic": _settings.project_topic,
    "full_name": _settings.full_name,
    "registered_email": _settings.registered_email,
}


def render_submission_header() -> None:
    """Render the SPARKIIT-required landing-page identity block.

    Must be the first Streamlit command called before any other UI element.
    """
    st.title(SUBMISSION["project_topic"])
    st.caption(
        f"**Submitted by:** {SUBMISSION['full_name']}  |  "
        f"**Registered Email:** {SUBMISSION['registered_email']}"
    )
    st.divider()


def get_landing_metadata() -> dict[str, str]:
    """Return identity metadata for non-Streamlit surfaces (FastAPI root)."""
    return dict(SUBMISSION)
