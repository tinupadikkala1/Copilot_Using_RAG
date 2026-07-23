"""Intent classifier: nearest-centroid over labeled example embeddings.

Deterministic, fast, and free (no extra model). Falls back to 'unknown'
when the best cosine similarity is below ``min_confidence``, which the
router treats as a signal to escalate.
"""

from __future__ import annotations

import logging
import threading

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
        "I need to dispute a charge",
        "what is your refund policy",
        "can I get a partial refund",
        "when will my refund be processed",
        "I was overcharged for my subscription",
        "cancel my subscription and refund me",
        "there is an unauthorized charge on my account",
        "how do I update my billing information",
        "my invoice is incorrect",
        "do you offer discounts for annual plans",
        "I need a receipt for my purchase",
    ],
    "technical": [
        "the app crashes on login",
        "API returns 500",
        "error message on screen",
        "I can't log in to my account",
        "the page is not loading properly",
        "I keep getting a timeout error",
        "my dashboard is showing wrong data",
        "the mobile app freezes when I open it",
        "I get a white screen after updating",
        "two-factor authentication code not received",
        "browser extension is not working",
        "file upload keeps failing with error 413",
        "integration with Slack is broken",
        "webhook is not triggering",
        "my account was hacked, suspicious activity detected",
    ],
    "account": [
        "change my email address",
        "delete my account",
        "upgrade my plan",
        "update my profile",
        "how do I change my username",
        "I want to merge two accounts",
        "transfer ownership of my account",
        "close my account permanently",
        "change my notification preferences",
        "update my profile picture",
        "I forgot my username",
        "how do I set up two-factor authentication",
        "add a team member to my account",
        "remove a user from my organization",
        "change my account from personal to business",
    ],
    "how_to": [
        "how do I export data",
        "where is the settings page",
        "how to invite a teammate",
        "how to change my password",
        "steps to reset my password",
        "guide to setting up API keys",
        "how to create a new project",
        "tutorial for importing contacts",
        "instructions to enable dark mode",
        "how to schedule a report",
        "how to share a document with external users",
        "walk me through setting up a webhook",
        "how to connect my calendar",
        "steps to duplicate a dashboard",
        "how to add custom fields to a form",
    ],
    "greeting": [
        "hello",
        "hi there",
        "good morning",
        "hey",
        "good afternoon",
        "good evening",
        "what's up",
        "howdy",
        "hey there",
        "yo",
        "greetings",
        "hi",
        "heyo",
        "morning",
        "sup",
    ],
    "human_agent": [
        "I want to talk to a human",
        "connect me to an agent",
        "this is urgent",
        "speak to a representative",
        "get me a real person please",
        "I need human support",
        "can I talk to someone",
        "I demand to speak to a manager",
        "this is an emergency",
        "your chatbot is not helping",
        "connect me to customer service",
        "I want to file a complaint with a person",
        "is there a support phone number",
        "I need immediate assistance from a human",
        "please escalate my issue to a supervisor",
    ],
}

SENSITIVE_INTENTS = frozenset({"human_agent", "billing"})


class IntentClassifier:
    """Classifies a query into one of the known intents.

    Uses nearest-centroid over pre-computed embedding centroids for
    each intent. Returns ``"unknown"`` if no centroid is close enough.

    The ``last_query_vector`` property exposes the embedding vector from
    the most recent ``predict()`` call so downstream components (e.g.
    the retriever) can reuse it without a duplicate embed API call.
    """

    def __init__(self, embedder: Embedder, min_confidence: float = 0.35) -> None:
        self._embedder = embedder
        self._min_confidence = min_confidence
        self._labels: list[str] = list(INTENT_EXAMPLES.keys())
        self._local = threading.local()

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

    @property
    def last_query_vector(self) -> list[float] | None:
        """Return the embedding vector from the most recent predict() call."""
        return getattr(self._local, 'last_query_vector', None)

    def predict(self, query: str, *, llm_fallback=None) -> tuple[str, float]:
        """Return ``(intent_label, confidence)`` with confidence in [0, 1].

        Two-stage classification:
        1. Fast centroid classifier (always runs).
        2. If confidence is below threshold AND an LLM client is provided
           via ``llm_fallback``, ask the LLM to classify as a second pass.

        Args:
            query: The user's input text.
            llm_fallback: Optional LLM client for second-pass classification.

        Returns:
            Tuple of (label, confidence). Label is ``"unknown"`` when
            confidence is below the threshold after both stages.
        """
        vec = self._embedder.encode([query])[0]
        self._local.last_query_vector = vec.tolist() if vec is not None else None
        sims = self._centroids @ vec  # cosine (both L2-normalised)
        idx = int(np.argmax(sims))
        # Map cosine [-1, 1] -> confidence [0, 1].
        confidence = float((sims[idx] + 1.0) / 2.0)

        if confidence >= self._min_confidence:
            return self._labels[idx], confidence

        # Stage 2: LLM fallback for low-confidence queries.
        if llm_fallback is not None and confidence < self._min_confidence:
            try:
                llm_result = self._llm_classify(query, llm_fallback)
                if llm_result is not None:
                    llm_label, llm_conf = llm_result
                    if llm_conf > confidence:
                        logger.debug(
                            "LLM fallback improved intent: centroid=%.3f(%s) -> llm=%.3f(%s)",
                            confidence, self._labels[idx] if idx < len(self._labels) else "?",
                            llm_conf, llm_label,
                        )
                        self._local.last_query_vector = None  # Invalidate since we used LLM
                        return llm_label, llm_conf
            except Exception:
                logger.exception("LLM intent fallback failed, using centroid result")

        logger.debug(
            "Low-confidence intent (%.3f < %.3f); returning 'unknown'",
            confidence,
            self._min_confidence,
        )
        return "unknown", confidence

    @staticmethod
    def _llm_classify(query: str, llm) -> tuple[str, float] | None:
        """Ask an LLM to classify the query into one of the known intents."""
        labels = ", ".join(INTENT_EXAMPLES.keys())
        prompt = (
            f"Classify the following customer query into exactly one of these intents: {labels}.\n"
            f"Reply with ONLY the intent label (one word), nothing else.\n"
            f"Query: {query}\n"
            f"Intent:"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = llm.generate(messages, temperature=0.0)
            response = response.strip().lower()
            for label in INTENT_EXAMPLES:
                if label in response:
                    return label, 0.85  # Higher confidence for LLM-verified
        except Exception:
            pass
        return None
