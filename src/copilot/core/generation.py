"""LLM generation via Ollama's /api/chat endpoint.

Uses the user's existing local models (qwen3:latest, qwen-local)
instead of any paid or remote API.
"""

from __future__ import annotations

import logging

import httpx

from copilot.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin client for Ollama's chat endpoint.

    Args:
        model: The Ollama model to use for generation.
        base_url: Ollama server base URL.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.llm_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout
        logger.info(
            "LLMClient initialised: model=%s url=%s",
            self._model,
            self._base_url,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
    ) -> str:
        """Send a chat completion request to Ollama.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If the Ollama API call fails.
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.exception("Ollama chat API call failed")
            raise RuntimeError(f"LLM generation failed for model {self._model}: {exc}") from exc

        content = data.get("message", {}).get("content", "")
        return content.strip()
