"""LLM generation via Ollama's /api/chat endpoint.

Uses the user's existing local models (qwen3:latest, qwen-local)
instead of any paid or remote API.
"""

from __future__ import annotations

import logging
import time

import httpx

from copilot.config import get_settings

logger = logging.getLogger(__name__)

# Retry configuration for low-end hardware where model loading may be slow.
_MAX_RETRIES = 3
_RETRY_DELAY_S = 5.0


class LLMClient:
    """Thin client for Ollama's chat endpoint.

    Includes retry logic for reliability on low-end hardware where
    the LLM may be slow to respond or timeout during model loading.

    Args:
        model: The Ollama model to use for generation.
        base_url: Ollama server base URL.
        timeout: Request timeout in seconds (default 600s for slow hardware).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.llm_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=self._timeout)
        logger.info(
            "LLMClient initialised: model=%s url=%s timeout=%.0fs",
            self._model,
            self._base_url,
            self._timeout,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
    ) -> str:
        """Send a chat completion request to Ollama with retry on failure.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If the Ollama API call fails after all retries.
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return content.strip()
            except httpx.HTTPError as exc:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAY_S * (attempt + 1)
                    logger.warning(
                        "LLM API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    logger.exception("Ollama chat API call failed after %d retries", _MAX_RETRIES)
                    raise RuntimeError(
                        f"LLM generation failed for model {self._model} after {_MAX_RETRIES} retries: {exc}"
                    ) from exc

        # Should never reach here, but satisfy type checker.
        return ""
