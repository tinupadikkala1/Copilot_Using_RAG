"""Unified Copilot UI — Chat, Upload KB, and Dashboard in one app."""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import streamlit as st

from copilot.branding import render_submission_header

API_URL = os.environ.get("COPILOT_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("COPILOT_API_KEY", "")

SUPPORTED_FORMATS: dict[str, str] = {
    ".md": "Markdown",
    ".html": "HTML",
    ".htm": "HTML",
    ".pdf": "PDF",
    ".txt": "Plain text",
    ".csv": "CSV",
}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Support Copilot",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# API helpers
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


def _fetch_metrics() -> dict | None:
    """Fetch dashboard metrics from the API."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{API_URL}/metrics",
                headers={"x-api-key": API_KEY},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError:
        return None





# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🤖 Copilot")

PAGES = {
    "💬 Chat": "chat",
    "📤 Upload KB": "upload",
    "📊 Dashboard": "dashboard",
}

if "page" not in st.session_state:
    st.session_state.page = "chat"

selected_label = st.sidebar.radio(
    "Navigate",
    options=list(PAGES.keys()),
    index=list(PAGES.keys()).index(
        next(k for k, v in PAGES.items() if v == st.session_state.page)
    ),
    label_visibility="collapsed",
)
st.session_state.page = PAGES[selected_label]

st.sidebar.divider()
st.sidebar.caption(
    "Upload your own support documents on the **Upload KB** page "
    "to build a custom knowledge base, then ask questions in **Chat**."
)
st.sidebar.divider()
st.sidebar.caption("Check the **Dashboard** for resolution metrics.")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None
if "index_result" not in st.session_state:
    st.session_state.index_result = None
if "build_running" not in st.session_state:
    st.session_state.build_running = False

# ---------------------------------------------------------------------------
#  Chat view
# ---------------------------------------------------------------------------


def _chat_view() -> None:
    render_submission_header()
    st.header("💬 Chat with Your Knowledge Base")
    st.caption(
        "Ask questions about your uploaded documents — "
        "the copilot answers using the vector index."
    )

    # Show processing indicator while another question is being answered
    if st.session_state.get("chat_processing", False):
        st.info("⏳ Your question is being processed... Please wait.")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your account, billing, or how-to guides..."
    ):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        st.session_state.last_query = prompt
        st.session_state.chat_processing = True

        try:
            data = _send_message(prompt, st.session_state.session_id)
        except httpx.HTTPError:
            st.error("Sorry, the server encountered an error. Please try again.")
            st.session_state.chat_processing = False
            st.stop()

        st.session_state.chat_processing = False
        st.session_state.session_id = data["session_id"]
        answer = data["answer"]

        # Append citations if present
        if data["citations"]:
            cites = "  \\n".join(
                f"[{c['marker']}] {c['title']}" for c in data["citations"]
            )
            answer += f"\\n\\n---\\n**Sources:**  \\n{cites}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Feedback buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        if col1.button("👍  Helpful", key=f"up_{len(st.session_state.messages)}"):
            _send_feedback(data, "up")
            st.success("Thanks for the feedback!")
        if col2.button("👎  Not helpful", key=f"down_{len(st.session_state.messages)}"):
            _send_feedback(data, "down")
            st.error("Thanks for the feedback — we'll use it to improve.")


# ---------------------------------------------------------------------------
#  Upload KB view
# ---------------------------------------------------------------------------


def _upload_view() -> None:
    render_submission_header()
    st.header("📤 Upload Knowledge Base Documents")
    st.caption(
        "Upload your own support documents (Markdown, PDF, HTML, TXT, CSV) to build a "
        "custom knowledge base. The system will automatically chunk, embed, and index them."
    )

    # Supported formats info
    format_str = ", ".join(
        f"`{ext}` ({name})" for ext, name in SUPPORTED_FORMATS.items()
    )
    st.markdown(f"**Supported formats:** {format_str}")
    st.markdown(f"**Max file size:** {MAX_FILE_SIZE_MB} MB per file")
    st.divider()

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files or drag & drop them here",
        type=list(SUPPORTED_FORMATS.keys()),
        accept_multiple_files=True,
        help=f"Select one or more files. Maximum {MAX_FILE_SIZE_MB} MB per file.",
    )

    if uploaded_files:
        # Validate file sizes
        oversized = [f for f in uploaded_files if len(f.getvalue()) > MAX_FILE_SIZE_BYTES]
        valid = [f for f in uploaded_files if len(f.getvalue()) <= MAX_FILE_SIZE_BYTES]

        if oversized:
            st.error(
                f"⚠️ {len(oversized)} file(s) exceed the {MAX_FILE_SIZE_MB} MB limit: "
                + ", ".join(f.name for f in oversized)
            )

        if valid:
            with st.status("📤 Uploading files...", expanded=True) as status:
                saved_paths = []
                for file in valid:
                    save_dir = Path("data/kb_raw")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / file.name

                    with open(save_path, "wb") as f:
                        f.write(file.getvalue())

                    saved_paths.append(str(save_path))
                    st.write(f"✅ Saved `{file.name}` ({len(file.getvalue()) / 1024:.1f} KB)")

                status.update(
                    label=f"✅ {len(saved_paths)} file(s) uploaded successfully!",
                    state="complete",
                )

            st.session_state.upload_status = {
                "count": len(saved_paths),
                "files": saved_paths,
            }

    # Current file inventory
    st.divider()
    st.subheader("📂 Current Knowledge Base")

    kb_path = Path("data/kb_raw")
    existing_docs: list[Path] = []
    if kb_path.exists():
        kb_files = sorted(kb_path.iterdir())
        existing_docs = [
            f
            for f in kb_files
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
        ]

        if existing_docs:
            st.markdown(f"**{len(existing_docs)} document(s)** in `data/kb_raw/`:")
            for f in existing_docs:
                size_kb = f.stat().st_size / 1024
                fmt = SUPPORTED_FORMATS.get(f.suffix.lower(), "Unknown")
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.markdown(f"📄 `{f.name}`")
                col2.caption(f"{fmt}")
                col3.caption(f"{size_kb:.1f} KB")
        else:
            st.info("No documents yet. Upload files above to get started.")
    else:
        st.info("No documents yet. Upload files above to get started.")

    # Build / rebuild index button
    st.divider()

    col_a, col_b = st.columns([3, 2])
    with col_a:
        build_clicked = st.button(
            "🚀 Build/Refresh Vector Index",
            type="primary",
            use_container_width=True,
            disabled=not bool(existing_docs) or st.session_state.build_running,
            help="Chunk, embed, and index all documents in data/kb_raw/",
        )

    with col_b:
        if st.button("🗑️ Clear All Uploads", use_container_width=True):
            if kb_path.exists():
                for f in kb_path.iterdir():
                    if f.is_file():
                        f.unlink()
            st.session_state.upload_status = None
            st.session_state.index_result = None
            st.rerun()

    # ------------------------------------------------------------------
    # Index build — start in background + poll real-time progress
    # ------------------------------------------------------------------
    if build_clicked and API_KEY:
        st.session_state.build_running = True
        st.rerun()

    if st.session_state.build_running:
        _render_build_progress(API_KEY)

    # Show last index results (only when build is NOT running)
    if st.session_state.index_result and not st.session_state.build_running:
        result = st.session_state.index_result
        if "error" in result:
            st.error(f"Last build failed: {result['error']}")
        else:
            st.success(
                f"✅ **{result.get('chunks_indexed', 0)} chunks** indexed from "
                f"**{result.get('documents_loaded', 0)} documents** (last build)"
            )


def _render_build_progress(api_key: str) -> None:
    """Poll the build progress endpoint and render a real-time progress bar."""
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    phase_text = st.empty()

    with httpx.Client(timeout=10) as client:
        # Start the build
        try:
            resp = client.post(
                f"{API_URL}/upload/build",
                headers={"x-api-key": api_key},
            )
            resp.raise_for_status()
            start_data = resp.json()
        except httpx.HTTPError as e:
            status_placeholder.error(f"❌ Failed to start build: {e}")
            st.session_state.build_running = False
            return

        if start_data.get("status") == "already_running":
            status_placeholder.warning("⚠️ A build is already running. Please wait.")
            # Poll for the already-running build to finish
        elif start_data.get("status") != "started":
            if "chunks_indexed" in start_data:
                # Build completed synchronously (no changes detected)
                st.session_state.index_result = start_data
                st.session_state.build_running = False
                progress_bar.progress(1.0)
                phase_text.success(
                    f"✅ **{start_data.get('chunks_indexed', 0)} chunks** indexed from "
                    f"**{start_data.get('documents_loaded', 0)} documents**"
                )
                st.rerun()
                return
            else:
                status_placeholder.error(f"❌ Unexpected response: {start_data}")
                st.session_state.build_running = False
                return

        # Poll progress until completion
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                resp = client.get(
                    f"{API_URL}/upload/build/progress",
                    headers={"x-api-key": api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "idle")

                if status in ("starting", "running"):
                    current = data.get("current", 0)
                    total = data.get("total", 0)
                    phase = data.get("phase", "")

                    if total > 0:
                        pct = min(current / total, 1.0)
                        progress_bar.progress(pct)
                        bar_chars = int(pct * 20)
                        visual_bar = "█" * bar_chars + "░" * (20 - bar_chars)
                        phase_text.markdown(
                            f"**Batch {current}/{total}** — {phase}  "
                            f"`{visual_bar} {int(pct * 100)}%`"
                        )
                    else:
                        phase_text.markdown(f"⏳ {phase}")

                    status_placeholder.info("🧠 Building vector index...")

                elif status == "completed":
                    progress_bar.progress(1.0)
                    result = data.get("result", {})
                    st.session_state.index_result = result
                    chunks = result.get("chunks_indexed", 0)
                    docs = result.get("documents_loaded", 0)
                    phase_text.success(
                        f"✅ **{chunks} chunks** indexed from **{docs} documents**"
                    )
                    status_placeholder.empty()
                    st.session_state.build_running = False
                    st.rerun()
                    return

                elif status == "error":
                    error_msg = data.get("error", "Unknown error")
                    progress_bar.progress(0.0)
                    phase_text.error(f"❌ Build failed: {error_msg}")
                    status_placeholder.empty()
                    st.session_state.index_result = {"error": error_msg}
                    st.session_state.build_running = False
                    return

            except httpx.HTTPError:
                pass

            time.sleep(1.5)

        # Timed out
        progress_bar.progress(0.0)
        phase_text.error("❌ Build timed out after 15 minutes")
        status_placeholder.empty()
        st.session_state.build_running = False


# ---------------------------------------------------------------------------
#  Dashboard view
# ---------------------------------------------------------------------------


def _dashboard_view() -> None:
    render_submission_header()
    st.header("📊 Resolution Metrics Dashboard")
    st.caption("Live analytics from the support copilot. Data updates on each page refresh.")

    metrics = _fetch_metrics()

    if metrics is None:
        st.error(
            f"Cannot connect to the API at {API_URL}. "
            "Make sure the server is running and the API key is configured."
        )
        st.stop()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        defl = metrics.get("deflection_rate", 0.0)
        st.metric(
            label="📈 Deflection Rate",
            value=f"{defl * 100:.1f}%",
            delta=None,
            help="Sessions resolved without human escalation",
        )

    with col2:
        csat = metrics.get("csat", 0.0)
        st.metric(
            label="⭐ CSAT",
            value=f"{csat * 100:.1f}%",
            delta=None,
            help="👍 / (👍 + 👎)",
        )

    with col3:
        p50 = metrics.get("p50", 0.0)
        st.metric(
            label="⚡ p50 Latency",
            value=f"{p50:.0f} ms",
            delta=None,
            help="Median response time",
        )

    with col4:
        p95 = metrics.get("p95", 0.0)
        st.metric(
            label="🚀 p95 Latency",
            value=f"{p95:.0f} ms",
            delta=None,
            help="95th percentile response time",
        )

    # Target comparison
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
            # Latency — lower is better
            met = actual <= target
            col_b.markdown(f"{'✅' if met else '❌'} Actual: {actual:.0f} ms")
            col_c.markdown(f"Target: ≤ {target:.0f} ms")
        else:
            met = actual >= target
            col_b.markdown(f"{'✅' if met else '❌'} Actual: {actual:.1%}")
            col_c.markdown(f"Target: ≥ {target:.1%}")

    st.divider()
    st.caption("Data sourced from the analytics database. Refresh the page to update.")


# ---------------------------------------------------------------------------
#  Render the selected page
# ---------------------------------------------------------------------------

if st.session_state.page == "chat":
    _chat_view()
elif st.session_state.page == "upload":
    _upload_view()
elif st.session_state.page == "dashboard":
    _dashboard_view()
