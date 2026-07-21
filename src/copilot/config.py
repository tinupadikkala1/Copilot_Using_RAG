"""Typed settings loaded from environment variables and TOML config."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — override any field via environment variable or .env file."""

    # --- Ollama backend ---
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen3:latest"
    llm_backend: str = "ollama"

    # --- Retrieval ---
    retrieval_k: int = 5
    min_retrieval_score: float = 0.35

    # --- Groundedness ---
    min_groundedness: float = 0.60

    # --- Intent ---
    min_intent_confidence: float = 0.35

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Paths ---
    kb_root: str = "data/kb_raw"
    persist_dir: str = "data/chroma"
    db_path: str = "data/metrics.db"

    # --- Serving ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""

    # --- Branding / Submission ---
    project_topic: str = "Autonomous Customer Support Copilot"
    full_name: str = "TINU THOMAS P"
    registered_email: str = "tinupadikkala1@gmail.com"

    model_config = SettingsConfigDict(
        env_prefix="COPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Return a singleton Settings instance."""
    return Settings()
