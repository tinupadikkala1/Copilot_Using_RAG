"""Streamlit page for uploading custom KB documents.

Users can drag-and-drop files, see the current file inventory,
and trigger an index rebuild — all without touching the terminal.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st

from copilot.branding import render_submission_header

API_URL = os.environ.get("COPILOT_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("COPILOT_API_KEY", "")

SUPPORTED_FORMATS = {
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
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Upload KB Documents",
    page_icon="📤",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🤖 Copilot")
st.sidebar.page_link("chat_app.py", label="💬 Chat", icon="💬")
st.sidebar.page_link("upload.py", label="📤 Upload KB", icon="📤")
st.sidebar.page_link("dashboard.py", label="📊 Dashboard", icon="📊")

render_submission_header()
st.title("📤 Upload Knowledge Base Documents")
st.caption(
    "Upload your own support documents (Markdown, PDF, HTML, TXT, CSV) to build a "
    "custom knowledge base. The system will automatically chunk, embed, and index them."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "upload_status" not in st.session_state:
    st.session_state.upload_status = None
if "index_result" not in st.session_state:
    st.session_state.index_result = None

# ---------------------------------------------------------------------------
# Supported formats info
# ---------------------------------------------------------------------------

format_str = ", ".join(f"`{ext}` ({name})" for ext, name in SUPPORTED_FORMATS.items())
st.markdown(f"**Supported formats:** {format_str}")
st.markdown(f"**Max file size:** {MAX_FILE_SIZE_MB} MB per file")
st.divider()

# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------

uploaded_files = st.file_uploader(
    "Choose files or drag & drop them here",
    type=list(SUPPORTED_FORMATS.keys()),
    accept_multiple_files=True,
    help=f"Select one or more files. Maximum {MAX_FILE_SIZE_MB} MB per file.",
)

if uploaded_files:
    # Validate file sizes.
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

# ---------------------------------------------------------------------------
# Current file inventory
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📂 Current Knowledge Base")

kb_path = Path("data/kb_raw")
if kb_path.exists():
    kb_files = sorted(kb_path.iterdir())
    existing_docs = [f for f in kb_files if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS]

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

# ---------------------------------------------------------------------------
# Build / rebuild index button
# ---------------------------------------------------------------------------

st.divider()

col_a, col_b = st.columns([3, 2])
with col_a:
    build_clicked = st.button(
        "🚀 Build/Refresh Vector Index",
        type="primary",
        use_container_width=True,
        disabled=not (existing_docs if kb_path.exists() else False),
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

# ---------------------------------------------------------------------------
# Index build status
# ---------------------------------------------------------------------------

if build_clicked and API_KEY:
    with st.status("🧠 Building vector index...", expanded=True) as status:
        try:
            with httpx.Client(timeout=300) as client:
                resp = client.post(
                    f"{API_URL}/upload/build",
                    headers={"x-api-key": API_KEY},
                )
                resp.raise_for_status()
                data = resp.json()

            st.session_state.index_result = data
            status.update(
                label=f"✅ Index built: {data.get('chunks_indexed', 0)} chunks from "
                f"{data.get('documents_loaded', 0)} documents",
                state="complete",
            )
        except httpx.HTTPError as e:
            st.session_state.index_result = {"error": str(e)}
            status.update(
                label="❌ Index build failed",
                state="error",
            )

    st.rerun()
elif build_clicked and not API_KEY:
    st.error("API key not configured. Set COPILOT_API_KEY environment variable.")

# ---------------------------------------------------------------------------
# Show last index results
# ---------------------------------------------------------------------------

if st.session_state.index_result and not build_clicked:
    result = st.session_state.index_result
    if "error" in result:
        st.error(f"Last build failed: {result['error']}")
    else:
        st.success(
            f"✅ **{result.get('chunks_indexed', 0)} chunks** indexed from "
            f"**{result.get('documents_loaded', 0)} documents** "
            f"(last build)"
        )
