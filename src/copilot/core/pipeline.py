"""Online request pipeline: intent -> route -> retrieve -> generate -> guard -> escalate."""

from __future__ import annotations

import logging
import re
import time

from copilot.analytics import metrics
from copilot.analytics.db import init_db
from copilot.core.cache import ResponseCache
from copilot.core.escalation import should_escalate
from copilot.core.generation import LLMClient
from copilot.core.guards import groundedness_score, sanitize
from copilot.core.intent import IntentClassifier
from copilot.core.prompt import build_rag_prompt
from copilot.core.retriever import Retriever
from copilot.core.router import Route, route
from copilot.schemas import ChatResponse, Citation

logger = logging.getLogger(__name__)

_CITE = re.compile(r"\[(\d+)\]")

GREETING_REPLY = "Hi! I'm your support assistant. What can I help you with today?"

# Minimum retrieval score to proceed with LLM generation.
# Set very low so the LLM always gets a chance to answer.
_MIN_RETRIEVAL_FOR_LLM = 0.05

# Fallback answer when retrieval is too weak — still tries LLM but
# acknowledges uncertainty. No ticket is created since there's no
# human queue to handle it.
_LOW_CONFIDENCE_ANSWER = (
    "I don't have enough information in my knowledge base to give you a "
    "confident answer to that. Here's what I can tell you based on "
    "what I found:\n\n"
)

# Response cache for identical queries.
_response_cache = ResponseCache(maxsize=128, ttl_s=300.0)


class SupportPipeline:
    """Orchestrates the full online request path.

    Accepts a user query and returns a ChatResponse by running it through:
    sanitisation -> intent classification -> routing -> retrieval ->
    LLM generation -> groundedness check -> optional escalation.

    Optimization: if retrieval scores are too weak we short-circuit before
    calling the LLM. Pre-computed query vectors from intent classification
    are reused for retrieval, saving one embed call.
    """

    def __init__(
        self,
        retriever: Retriever,
        intent_clf: IntentClassifier,
        llm: LLMClient,
        k: int = 5,
        min_groundedness: float = 0.60,
    ) -> None:
        self._retriever = retriever
        self._intent = intent_clf
        self._llm = llm
        self._k = k
        self._min_groundedness = min_groundedness
        init_db()

    def answer_query(self, query: str, session_id: str) -> ChatResponse:
        """Process a user query end-to-end and return a response.

        Args:
            query: The user's input message.
            session_id: A unique identifier for this conversation session.

        Returns:
            A ChatResponse with the answer, citations, intent, and
            escalation status.
        """
        started = time.perf_counter()
        query = sanitize(query.strip())

        # --- Check response cache ---
        cache_key = query.lower().strip()
        cached = _response_cache.get(cache_key)
        if cached is not None:
            logger.debug("Returning cached response for key=%s", cache_key[:40])
            cached_resp = ChatResponse(**cached)
            cached_resp.session_id = session_id
            return cached_resp

        # --- Intent classification (two-stage: centroid + optional LLM fallback) ---
        intent, intent_conf = self._intent.predict(query, llm_fallback=self._llm)

        # --- Smalltalk / greeting shortcut ---
        route_decision = route(intent, intent_conf, 1.0)
        if route_decision == Route.SMALLTALK:
            resp = ChatResponse(
                answer=GREETING_REPLY,
                intent=intent,
                confidence=intent_conf,
                session_id=session_id,
            )
            metrics.record_turn(resp, (time.perf_counter() - started) * 1000)
            return resp

        # --- HyDE: generate a hypothetical answer to improve retrieval ---
        contexts = self._retrieve_with_hyde(query, intent, intent_conf)
        retrieval_top = Retriever.top_score(contexts)

        # --- Low-retrieval path: still try the LLM, but with a caveat ---
        if retrieval_top < _MIN_RETRIEVAL_FOR_LLM:
            if not contexts:
                answer = "I don't have enough information in my knowledge base to answer that."
                citations: list[Citation] = []
            else:
                messages = build_rag_prompt(query, contexts, use_cot=False)
                llm_answer = self._llm.generate(messages) or ""
                # Avoid double "I don't have enough information" when the
                # LLM follows its instruction to output the exact refusal.
                if llm_answer.strip().startswith(
                    "I don't have enough information"
                ):
                    answer = llm_answer
                else:
                    answer = _LOW_CONFIDENCE_ANSWER + llm_answer
                citations = self._extract_citations(answer, contexts)
            resp = ChatResponse(
                answer=answer,
                intent=intent,
                citations=citations,
                escalated=False,
                confidence=min(retrieval_top * 2, 0.5),
                session_id=session_id,
            )
            metrics.record_turn(resp, (time.perf_counter() - started) * 1000)
            return resp

        # --- Generate grounded answer (with CoT for better accuracy) ---
        messages = build_rag_prompt(query, contexts, use_cot=True)
        answer = self._llm.generate(messages)
        grounded = groundedness_score(answer, contexts, use_synonyms=True)

        # --- Check escalation ---
        decision = should_escalate(
            intent,
            intent_conf,
            retrieval_top,
            grounded,
            min_groundedness=self._min_groundedness,
        )

        # Always extract citations — we return the answer regardless of escalation.
        citations = self._extract_citations(answer, contexts)

        if decision.escalate:
            # No human queue exists — give the best answer we have instead.
            resp = ChatResponse(
                answer=answer,
                citations=citations,
                intent=intent,
                escalated=False,
                confidence=min(intent_conf, grounded, 0.5),
                session_id=session_id,
            )
        else:
            resp = ChatResponse(
                answer=answer,
                citations=citations,
                intent=intent,
                confidence=grounded,
                session_id=session_id,
            )
            _response_cache.put(cache_key, resp.model_dump())

        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "answered session=%s intent=%s escalated=%s latency_ms=%.0f",
            session_id,
            intent,
            resp.escalated,
            latency_ms,
        )
        metrics.record_turn(resp, latency_ms)
        return resp

    # --- HyDE: Hypothetical Document Embeddings for better retrieval ---

    def _retrieve_with_hyde(
        self,
        query: str,
        intent: str,
        intent_conf: float,
    ) -> list:
        """Retrieve context, optionally using HyDE (Hypothetical Document
        Embeddings) when confidence is high enough.

        HyDE generates a hypothetical answer first, then embeds THAT for
        retrieval instead of the raw query. This bridges the lexical gap
        between short user queries and long KB documents.

        For low-confidence or simple intents, falls back to normal retrieval
        reusing the intent classifier's query vector (batch embed).
        """
        query_vec = self._intent.last_query_vector

        # Use HyDE for non-trivial queries to improve retrieval.
        use_hyde = intent_conf >= 0.5 and intent not in ("greeting", "human_agent")

        if use_hyde and self._llm is not None:
            try:
                hyde_prompt = (
                    f"Write a brief hypothetical support article that answers this question:\n"
                    f"{query}\n\n"
                    f"Use a helpful, factual tone. Write 2-3 sentences."
                )
                hyde_messages = [
                    {"role": "system", "content": "You are a knowledge base writer."},
                    {"role": "user", "content": hyde_prompt},
                ]
                hyde_answer = self._llm.generate(hyde_messages, temperature=0.3)
                if hyde_answer and len(hyde_answer) > 20:
                    logger.debug(
                        "HyDE generated hypothetical doc (%d chars) for query",
                        len(hyde_answer),
                    )
                    # Embed the hypothetical doc and use that for retrieval.
                    hyde_vec = self._retriever.embedder.encode([hyde_answer])[0].tolist()
                    contexts = self._retriever.retrieve(query, self._k, query_vector=hyde_vec)
                    logger.debug(
                        "HyDE retrieval returned %d chunks (score=%.3f)",
                        len(contexts),
                        Retriever.top_score(contexts),
                    )
                    # If HyDE returned good results, use them.
                    if contexts and Retriever.top_score(contexts) > 0.15:
                        return contexts
                    # Otherwise fall through to normal retrieval.
            except Exception:
                logger.debug("HyDE generation failed, using direct query", exc_info=True)

        # Normal retrieval (reuse intent embed vector).
        contexts = self._retriever.retrieve(query, self._k, query_vector=query_vec)
        return contexts

    @staticmethod
    def _extract_citations(
        answer: str,
        contexts: list,
    ) -> list[Citation]:
        """Parse inline [n] markers from the answer into Citation objects."""
        markers = sorted({int(m) for m in _CITE.findall(answer)})
        citations: list[Citation] = []
        for marker in markers:
            if 1 <= marker <= len(contexts):
                rc = contexts[marker - 1]
                citations.append(
                    Citation(
                        marker=marker,
                        chunk_id=rc.chunk.chunk_id,
                        title=rc.chunk.title,
                        source_path=rc.chunk.source_path,
                    )
                )
        return citations
