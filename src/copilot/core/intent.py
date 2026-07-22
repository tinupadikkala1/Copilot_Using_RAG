"""Intent classifier: nearest-centroid over labeled example embeddings.

Deterministic, fast, and free (no extra model). Falls back to 'unknown'
when the best cosine similarity is below ``min_confidence``, which the
router treats as a signal to escalate.
"""

from __future__ import annotations

import logging

import numpy as np

from copilot.indexing.embedder import Embedder

logger = logging.getLogger(__name__)

# Seed examples per intent (extend from the feedback loop over time).
INTENT_EXAMPLES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice",
        "How do I get a refund",
        "update my credit card",
        "why was I charged",
    ],
    "technical": [
        "the app crashes on login",
        "API returns 500",
        "reset my password",
        "error message on screen",
    ],
    "account": [
        "change my email address",
        "delete my account",
        "upgrade my plan",
        "update my profile",
    ],
    "how_to": [
        "how do I export data",
        "where is the settings page",
        "how to invite a teammate",
        "how to change my password",
    ],
    "greeting": [
        "hello",
        "hi there",
        "good morning",
        "hey",
    ],
    "human_agent": [
        "I want to talk to a human",
        "connect me to an agent",
        "this is urgent",
        "speak to a representative",
    ],
}

SENSITIVE_INTENTS = frozenset({"human_agent", "billing"})


class IntentClassifier:
    """Classifies a query into one of the known intents.

    Uses nearest-centroid over pre-computed embedding centroids for
    each intent. Returns ``"unknown"`` if no centroid is close enough.
    """

    def __init__(self, embedder: Embedder, min_confidence: float = 0.35) -> None:
        self._embedder = embedder
        self._min_confidence = min_confidence
        self._labels: list[str] = list(INTENT_EXAMPLES.keys())

        # Pre-compute normalised centroid vectors for each intent.
        centroids = []
        for label in self._labels:
            vecs = self._embedder.encode(INTENT_EXAMPLES[label])
            centroids.append(vecs.mean(axis=0))
        mat = np.vstack(centroids).astype(np.float32)
        # Re-normalise so dot product == cosine similarity.
        self._centroids = mat / np.linalg.norm(mat, axis=1, keepdims=True)

        logger.info(
            "IntentClassifier initialised: %d intents, min_confidence=%.2f",
            len(self._labels),
            min_confidence,
        )

    def predict(self, query: str) -> tuple[str, float]:
        """Return ``(intent_label, confidence)`` with confidence in [0, 1].

        Args:
            query: The user's input text.

        Returns:
            Tuple of (label, confidence). Label is ``"unknown"`` when
            confidence is below the threshold.
        """
        vec = self._embedder.encode([query])[0]
        sims = self._centroids @ vec  # cosine (both L2-normalised)
        idx = int(np.argmax(sims))
        # Map cosine [-1, 1] -> confidence [0, 1].
        confidence = float((sims[idx] + 1.0) / 2.0)
        if confidence < self._min_confidence:
            logger.debug(
                "Low-confidence intent (%.3f < %.3f); returning 'unknown'",
                confidence,
                self._min_confidence,
            )
            return "unknown", confidence
        return self._labels[idx], confidence
