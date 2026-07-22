"""FastAPI serving layer. All state-changing endpoints require an API key."""

from __future__ import annotations

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

_limiter = RateLimiter()

# Upload configuration (module-level for testability).
SUPPORTED_UPLOAD_EXTENSIONS: set[str] = {
    ".md", ".html", ".htm", ".pdf", ".txt", ".csv"
}
KB_RAW = Path("data/kb_raw")
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


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
        """
        _limiter.check(request)
        session_id = req.session_id or uuid.uuid4().hex
        pipeline = get_pipeline()
        started = time.perf_counter()
        resp = pipeline.answer_query(req.message, session_id)
        metrics.record_turn(resp, (time.perf_counter() - started) * 1000)
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

            save_path = KB_RAW / file.filename
            with open(save_path, "wb") as f:
                f.write(content)

            saved.append(
                {
                    "filename": file.filename,
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

    @app.post(
        "/upload/build",
        dependencies=[Depends(require_api_key)],
    )
    def build_kb_index() -> dict:
        """Build/refresh the vector index from all files in data/kb_raw/.

        Requires ``X-API-Key`` header. Returns the number of documents
        loaded and chunks indexed. Also refreshes the BM25 index for
        hybrid search.
        """
        if not KB_RAW.exists() or not any(KB_RAW.iterdir()):
            return {"status": "ok", "documents_loaded": 0, "chunks_indexed": 0}

        embedder = Embedder()
        store = ChromaStore(persist_dir="data/chroma")
        # Get the cached retriever (if any) so BM25 is updated after build.
        from copilot.serving.deps import get_pipeline

        pipeline = get_pipeline()
        retriever = pipeline._retriever if hasattr(pipeline, "_retriever") else None

        chunks_count = build_index(
            kb_root=KB_RAW,
            store=store,
            embedder=embedder,
            retriever=retriever,
        )
        docs_count = len(list(KB_RAW.iterdir()))

        return {
            "status": "ok",
            "documents_loaded": docs_count,
            "chunks_indexed": chunks_count,
        }

    return app


# ASGI entrypoint — used by uvicorn.
app = create_app()
