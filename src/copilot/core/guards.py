"""Anti-hallucination and prompt-injection guards."""

from __future__ import annotations

import logging
import re

from copilot.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

REFUSAL = "I don't have enough information to answer that confidently."

# Patterns that indicate an injection attempt inside untrusted content.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|previous|above) instructions", re.I),
    re.compile(r"disregard (the )?(system|previous) (prompt|instructions)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"reveal (your )?(system )?prompt", re.I),
]

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "are",
        "you",
        "your",
        "please",
        "we",
        "it",
        "this",
        "that",
        "with",
        "can",
        "will",
        "not",
        "be",
        "have",
        "do",
    }
)


def sanitize(text: str) -> str:
    """Neutralise injection directives found in untrusted KB/user text.

    Args:
        text: Input text that may contain injection attempts.

    Returns:
        Text with injection patterns replaced by ``[filtered]``.
    """
    cleaned = text
    for pat in _INJECTION_PATTERNS:
        cleaned = pat.sub("[filtered]", cleaned)
    return cleaned


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def groundedness_score(answer: str, contexts: list[RetrievedChunk]) -> float:
    """Lexical-overlap groundedness: fraction of answer tokens present in context.

    A lightweight, dependency-free proxy suitable for the local stack.
    Returns a value in [0, 1]; the pipeline escalates when it falls below
    the threshold.

    Args:
        answer: The LLM-generated answer text.
        contexts: The retrieved chunks used as context.

    Returns:
        A score in [0, 1] representing how much of the answer is
        supported by the context.
    """
    if answer.strip() == REFUSAL:
        return 1.0  # A correct refusal is fully grounded.

    answer_tokens = _tokens(answer) - _STOPWORDS
    if not answer_tokens:
        return 0.0

    context_tokens: set[str] = set()
    for rc in contexts:
        context_tokens |= _tokens(rc.chunk.text)

    supported = answer_tokens & context_tokens
    return len(supported) / len(answer_tokens)
