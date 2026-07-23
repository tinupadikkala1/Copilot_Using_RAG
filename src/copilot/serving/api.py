"""FastAPI serving layer. All state-changing endpoints require an API key."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Request, UploadFile
from pydantic import BaseModel, Field

from copilot.analytics import metrics, reporting
from copilot.analytics.db import init_db
from copilot.branding import get_landing_metadata
from copilot.feedback.store import record_feedback
from copilot.indexing.embedder import Embedder
from copilot.indexing.index_builder import build_index
from copilot.indexing.vector_store import ChromaStore
from copilot.schemas import ChatResponse
from copilot.serving.deps import get_pipeline
from copilot.serving.security import RateLimiter, require_api_key

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    # Take only the basename (strip directories)
    name = Path(name).name
    # Remove any remaining path separators and null bytes
    name = re.sub(r'[/\\\x00]', '', name)
    # Reject empty or dot-only names
    if not name or name.startswith('.'):
        return ''
    return name


_limiter = RateLimiter()

# Upload configuration (module-level for testability).
SUPPORTED_UPLOAD_EXTENSIONS: set[str] = {
    ".md", ".html", ".htm", ".pdf", ".txt", ".csv"
}
KB_RAW = Path("data/kb_raw")
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# ---------------------------------------------------------------------------
# Concurrency guards
# ---------------------------------------------------------------------------

# Prevents concurrent /chat requests (local LLM cannot serve two at once).
_chat_lock = threading.Lock()

# Prevents concurrent /upload/build requests.
_build_lock = threading.Lock()

# Shared build progress — written by the background thread, read by the
# progress-polling endpoint.
_build_progress_lock = threading.Lock()
_build_progress: dict = {
    "status": "idle",        # idle | running | completed | error
    "current": 0,
    "total": 0,
    "phase": "",
    "result": None,
    "error": None,
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    """User feedback on a previous answer."""

    session_id: str
    query: str
    answer: str
    rating: str = Field(pattern="^(up|down)$")
    correction: str | None = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and return a configured FastAPI application."""
    app = FastAPI(
        title="Autonomous Customer Support Copilot",
        version="1.0.0",
        description=(
            "RAG-based customer support system using local Ollama models. "
            "Resolves KB-backed queries and auto-escalates when needed."
        ),
    )
    init_db()

    # --- Landing / health ---

    @app.get("/")
    def landing() -> dict[str, str]:
        """Root route — return project identity (SPARKIIT requirement)."""
        return get_landing_metadata()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Simple health-check endpoint."""
        return {"status": "ok"}

    @app.post(
        "/chat",
        response_model=ChatResponse,
        dependencies=[Depends(require_api_key)],
    )
    def chat(req: ChatRequest, request: Request) -> ChatResponse:
        """Answer a customer support query.

        Requires ``X-API-Key`` header. Returns a grounded, cited answer
        or an escalation ticket if the query cannot be confidently resolved.

        Only one question is processed at a time. If another request arrives
        while one is in progress it will wait in line.
        """
        _limiter.check(request)
        session_id = req.session_id or uuid.uuid4().hex
        pipeline = get_pipeline()

        with _chat_lock:
            resp = pipeline.answer_query(req.message, session_id)

        return resp

    @app.post(
        "/feedback",
        dependencies=[Depends(require_api_key)],
    )
    def feedback(req: FeedbackRequest) -> dict[str, str]:
        """Record user feedback (👍/👎 + optional correction).

        Requires ``X-API-Key`` header.
        """
        record_feedback(
            req.session_id,
            req.query,
            req.answer,
            req.rating,  # type: ignore[arg-type]
            req.correction,
        )
        return {"status": "recorded"}

    @app.get(
        "/metrics",
        dependencies=[Depends(require_api_key)],
    )
    def get_metrics() -> dict[str, float]:
        """Return aggregate resolution metrics.

        Requires ``X-API-Key`` header.
        """
        return {
            "deflection_rate": reporting.deflection_rate(),
            "csat": reporting.csat(),
            **metrics.latency_percentiles(),
        }

    # --- Upload & index building ---

    @app.post(
        "/upload",
        dependencies=[Depends(require_api_key)],
    )
    async def upload_files(files: list[UploadFile] = File(...)) -> dict:
        """Upload one or more KB documents and save them to data/kb_raw/.

        Requires ``X-API-Key`` header. Accepted formats:
        .md, .html, .htm, .pdf, .txt, .csv (max 20 MB per file).
        """
        KB_RAW.mkdir(parents=True, exist_ok=True)

        saved = []
        errors = []

        for file in files:
            ext = (
                Path(file.filename or "").suffix.lower()
                if file.filename
                else ""
            )

            if not file.filename:
                errors.append({"file": "unknown", "error": "Empty filename"})
                continue

            safe_name = _sanitize_filename(file.filename)
            if not safe_name:
                errors.append({"file": file.filename, "error": "Invalid filename"})
                continue

            if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
                errors.append(
                    {"file": file.filename, "error": f"Unsupported format '{ext}'"}
                )
                continue

            content = await file.read()

            if len(content) == 0:
                errors.append({"file": file.filename, "error": "Empty file"})
                continue

            if len(content) > MAX_UPLOAD_SIZE_BYTES:
                size_mb = len(content) / (1024 * 1024)
                errors.append(
                    {
                        "file": file.filename,
                        "error": f"File exceeds 20 MB limit ({size_mb:.1f} MB)",
                    }
                )
                continue

            save_path = KB_RAW / safe_name
            with open(save_path, "wb") as f:
                f.write(content)

            saved.append(
                {
                    "filename": safe_name,
                    "size_bytes": len(content),
                    "path": str(save_path),
                }
            )

        return {
            "status": "ok",
            "saved": saved,
            "errors": errors,
            "total_saved": len(saved),
            "total_errors": len(errors),
        }

    def _run_build_in_background() -> None:
        """Run the index build in a background thread, updating _build_progress."""
        global _build_progress

        def _progress(current: int, total: int, phase: str) -> None:
            with _build_progress_lock:
                _build_progress.update(
                    status="running", current=current, total=total, phase=phase
                )
            logger.debug("Build progress: %d/%d — %s", current, total, phase)

        try:
            embedder = Embedder(timeout=600.0)
            store = ChromaStore(persist_dir="data/chroma")

            from copilot.serving.deps import get_pipeline

            pipeline = get_pipeline()
            retriever = pipeline._retriever if hasattr(pipeline, "_retriever") else None

            chunks_count = build_index(
                kb_root=KB_RAW,
                store=store,
                embedder=embedder,
                retriever=retriever,
                progress_callback=_progress,
            )
            docs_count = len(list(KB_RAW.iterdir()))

            with _build_progress_lock:
                _build_progress.update(
                    status="completed",
                    current=_build_progress["total"],
                    result={
                        "status": "ok",
                        "documents_loaded": docs_count,
                        "chunks_indexed": chunks_count,
                    },
                )
        except Exception as exc:
            logger.exception("Background index build failed")
            with _build_progress_lock:
                _build_progress.update(
                    status="error",
                    error=str(exc),
                    result={"status": "error", "error": str(exc)},
                )
        finally:
            _build_lock.release()

    @app.post(
        "/upload/build",
        dependencies=[Depends(require_api_key)],
    )
    def build_kb_index() -> dict:
        """Start building/refreshing the vector index in a background thread.

        Requires ``X-API-Key`` header. Returns immediately with a status
        indicator. Poll ``GET /upload/build/progress`` to track progress.

        If a build is already running, returns 409 Conflict.
        If no documents exist, returns 200 with zero counts.
        """
        if not KB_RAW.exists() or not any(KB_RAW.iterdir()):
            return {"status": "ok", "documents_loaded": 0, "chunks_indexed": 0}

        if not _build_lock.acquire(blocking=False):
            return {"status": "already_running"}

        # Reset progress and launch background thread.
        with _build_progress_lock:
            _build_progress.clear()
            _build_progress.update(status="starting", current=0, total=0, phase="Initialising…")

        thread = threading.Thread(target=_run_build_in_background, daemon=True)
        thread.start()

        return {"status": "started"}

    @app.get(
        "/upload/build/progress",
        dependencies=[Depends(require_api_key)],
    )
    def get_build_progress() -> dict:
        """Return current index-build progress.

        Requires ``X-API-Key`` header. Returns:
        - ``status``: "idle" | "starting" | "running" | "completed" | "error"
        - ``current``: Batch number currently being processed (0-based).
        - ``total``: Total number of batches (0 if unknown).
        - ``phase``: Human-readable phase description.
        - ``result``: Final build result (only present when status is "completed").
        - ``error``: Error message (only present when status is "error").
        """
        with _build_progress_lock:
            return dict(_build_progress)

    return app


# ASGI entrypoint — used by uvicorn.
app = create_app()
