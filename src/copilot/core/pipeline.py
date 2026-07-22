"""Online request pipeline: intent -> route -> retrieve -> generate -> guard -> escalate."""

from __future__ import annotations

import logging
import re
import time

from copilot.analytics import metrics
from copilot.analytics.db import init_db
from copilot.core.cache import ResponseCache
from copilot.core.escalation import create_ticket, should_escalate
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
# Below this threshold we escalate immediately, saving the expensive LLM call.
_MIN_RETRIEVAL_FOR_LLM = 0.25

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
        if cached is not None and cached["session_id"] == session_id:
            logger.debug("Returning cached response for key=%s", cache_key[:40])
            return ChatResponse(**cached)

        intent, intent_conf = self._intent.predict(query)

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

        # --- Retrieve relevant context (reuse query vector from intent) ---
        query_vec = self._intent.last_query_vector
        contexts = self._retriever.retrieve(query, self._k, query_vector=query_vec)
        retrieval_top = Retriever.top_score(contexts)

        # --- Early exit: skip LLM if retrieval is too weak ---
        if retrieval_top < _MIN_RETRIEVAL_FOR_LLM:
            ticket_id = create_ticket(session_id, query, "low_retrieval")
            answer = (
                "I couldn't find relevant information in our knowledge base to answer that. "
                f"I've created ticket {ticket_id} and a human agent will follow up shortly."
            )
            resp = ChatResponse(
                answer=answer,
                intent=intent,
                escalated=True,
                confidence=retrieval_top,
                session_id=session_id,
            )
            metrics.record_turn(resp, (time.perf_counter() - started) * 1000)
            return resp

        # --- Generate grounded answer ---
        messages = build_rag_prompt(query, contexts)
        answer = self._llm.generate(messages)
        grounded = groundedness_score(answer, contexts)

        # --- Check escalation ---
        decision = should_escalate(
            intent,
            intent_conf,
            retrieval_top,
            grounded,
            min_groundedness=self._min_groundedness,
        )

        if decision.escalate:
            ticket_id = create_ticket(session_id, query, decision.reason)
            answer = (
                "This looks like it needs a specialist. I've created ticket "
                f"{ticket_id} and a human agent will follow up shortly."
            )
            resp = ChatResponse(
                answer=answer,
                intent=intent,
                escalated=True,
                confidence=min(intent_conf, grounded),
                session_id=session_id,
            )
        else:
            citations = self._extract_citations(answer, contexts)
            resp = ChatResponse(
                answer=answer,
                citations=citations,
                intent=intent,
                confidence=grounded,
                session_id=session_id,
            )
            # Only cache successful (non-escalated) responses.
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
