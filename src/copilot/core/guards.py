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


# Synonym groups for relaxing token-matching in groundedness scoring.
_SYNONYM_GROUPS: list[set[str]] = [
    {"refund", "money", "back", "reimburse", "return"},
    {"password", "login", "signin", "sign", "in", "log", "authentication"},
    {"account", "profile", "user", "username"},
    {"email", "mail", "inbox", "gmail", "outlook"},
    {"delete", "remove", "close", "cancel", "terminate"},
    {"upgrade", "downgrade", "change", "switch", "plan"},
    {"error", "bug", "crash", "freeze", "fail", "broken"},
    {"help", "support", "assist", "guide", "tutorial"},
]

# Pre-build a synonym-expanded lookup.
_SYNONYM_MAP: dict[str, set[str]] = {}
for group in _SYNONYM_GROUPS:
    for word in group:
        _SYNONYM_MAP.setdefault(word, set()).update(group - {word})


def _expand_with_synonyms(tokens: set[str]) -> set[str]:
    """Expand a set of tokens with their synonyms from the predefined groups."""
    expanded = set(tokens)
    for t in tokens:
        if t in _SYNONYM_MAP:
            expanded.update(_SYNONYM_MAP[t])
    return expanded


def groundedness_score(
    answer: str,
    contexts: list[RetrievedChunk],
    use_synonyms: bool = True,
) -> float:
    """Lexical-overlap groundedness with optional synonym-awareness.

    A lightweight, dependency-free proxy suitable for the local stack.
    Returns a value in [0, 1]; the pipeline escalates when it falls below
    the threshold.

    When ``use_synonyms`` is True, the scoring also checks for semantically
    equivalent terms (e.g. "refund" ↔ "money back") using predefined
    synonym groups.

    Args:
        answer: The LLM-generated answer text.
        contexts: The retrieved chunks used as context.
        use_synonyms: Whether to relax matching with synonym groups.

    Returns:
        A score in [0, 1] representing how much of the answer is
        supported by the context.
    """
    if answer.strip() == REFUSAL:
        return 1.0  # A correct refusal is fully grounded.

    answer_tokens = _tokens(answer) - _STOPWORDS
    if not answer_tokens:
        return 0.0

    # Expand answer tokens with synonyms if enabled.
    expanded_answer = _expand_with_synonyms(answer_tokens) if use_synonyms else answer_tokens

    context_tokens: set[str] = set()
    for rc in contexts:
        context_tokens |= _tokens(rc.chunk.text)

    expanded_context = _expand_with_synonyms(context_tokens) if use_synonyms else context_tokens

    supported = expanded_answer & expanded_context

    # Also check bigram overlap for phrase-level grounding.
    answer_bigrams = set(zip(list(expanded_answer), list(expanded_answer)[1:])) if len(expanded_answer) > 1 else set()
    context_bigrams: set[tuple[str, str]] = set()
    for rc in contexts:
        ctx_tokens_list = list(_tokens(rc.chunk.text))
        context_bigrams.update(zip(ctx_tokens_list, ctx_tokens_list[1:]))

    bigram_overlap = len(answer_bigrams & context_bigrams) / max(len(answer_bigrams), 1)

    # Combined score: weight unigrams and bigrams.
    unigram_score = min(1.0, len(supported) / len(answer_tokens))
    return max(unigram_score, bigram_overlap)  # Take best signal
