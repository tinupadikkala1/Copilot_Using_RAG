"""Streamlit metrics dashboard for resolution analytics."""

from __future__ import annotations

import os

import httpx
import streamlit as st

from copilot.branding import render_submission_header

API_URL = os.environ.get("COPILOT_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("COPILOT_API_KEY", "")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Copilot Dashboard",
    page_icon="📊",
    layout="wide",
)

render_submission_header()
st.title("📊 Resolution Metrics Dashboard")
st.caption("Live analytics from the support copilot. Data updates on each page refresh.")

# ---------------------------------------------------------------------------
# Fetch metrics from the API
# ---------------------------------------------------------------------------

try:
    with httpx.Client(timeout=15) as client:
        resp = client.get(
            f"{API_URL}/metrics",
            headers={"x-api-key": API_KEY},
        )
        resp.raise_for_status()
        metrics_data = resp.json()
except httpx.HTTPError:
    st.error(
        f"Cannot connect to the API at {API_URL}. "
        "Make sure the server is running and the API key is configured."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Display KPI cards
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    defl = metrics_data.get("deflection_rate", 0.0)
    st.metric(
        label="📈 Deflection Rate",
        value=f"{defl * 100:.1f}%",
        delta=None,
        help="Sessions resolved without human escalation",
    )

with col2:
    csat = metrics_data.get("csat", 0.0)
    st.metric(
        label="⭐ CSAT",
        value=f"{csat * 100:.1f}%",
        delta=None,
        help="👍 / (👍 + 👎)",
    )

with col3:
    p50 = metrics_data.get("p50", 0.0)
    st.metric(
        label="⚡ p50 Latency",
        value=f"{p50:.0f} ms",
        delta=None,
        help="Median response time",
    )

with col4:
    p95 = metrics_data.get("p95", 0.0)
    st.metric(
        label="🚀 p95 Latency",
        value=f"{p95:.0f} ms",
        delta=None,
        help="95th percentile response time",
    )

# ---------------------------------------------------------------------------
# Target comparison
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🎯 Targets vs Actual")

targets = {
    "Deflection Rate": (defl, 0.30),
    "CSAT": (csat, 0.75),
    "p95 Latency": (p95, 6000.0),
}

for label, (actual, target) in targets.items():
    col_a, col_b, col_c = st.columns([2, 2, 6])
    col_a.markdown(f"**{label}**")
    if isinstance(target, float) and target > 1:
        # Latency — lower is better.
        met = actual <= target
        col_b.markdown(f"{'✅' if met else '❌'} Actual: {actual:.0f} ms")
        col_c.markdown(f"Target: ≤ {target:.0f} ms")
    else:
        met = actual >= target
        col_b.markdown(f"{'✅' if met else '❌'} Actual: {actual:.1%}")
        col_c.markdown(f"Target: ≥ {target:.1%}")

st.divider()
st.caption("Data sourced from the analytics database. Refresh the page to update.")
