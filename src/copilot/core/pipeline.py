"""Online request pipeline: intent -> route -> retrieve -> generate -> guard -> escalate."""

from __future__ import annotations

import logging
import re
import time

from copilot.analytics import metrics
from copilot.analytics.db import init_db
from copilot.core.cache import ResponseCache
from copilot.core.escalation import should_escalate, create_ticket
from copilot.core.generation import LLMClient
from copilot.core.guards import REFUSAL, groundedness_score, sanitize
from copilot.core.intent import IntentClassifier
from copilot.core.prompt import build_force_prompt, build_rag_prompt
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


def _is_refusal(text: str) -> bool:
    """Check if the LLM's output is a refusal to answer."""
    stripped = text.strip()
    # Check for the exact refusal phrase
    if stripped == REFUSAL:
        return True
    # Also check for common refusal variations
    if stripped.startswith("I don't have enough information"):
        return True
    if stripped.startswith("I don't know"):
        return True
    if stripped.startswith("I cannot answer"):
        return True
    return False


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

    def _generate_with_retry(
        self,
        query: str,
        contexts: list,
        use_cot: bool = True,
    ) -> str:
        """Generate an answer, retrying with a force prompt if the LLM refuses.

        Many local LLMs are fine-tuned to be overly cautious and will say
        "I don't have enough information..." even when relevant context
        is provided. This method:

        1. Tries the standard RAG prompt first.
        2. If the LLM outputs a refusal, retries ONCE with the force-answer
           prompt that explicitly commands the LLM to answer.
        3. Logs whether a retry was needed.
        """
        # First attempt: standard RAG prompt
        messages = build_rag_prompt(query, contexts, use_cot=use_cot)
        answer = self._llm.generate(messages) or ""

        if not _is_refusal(answer):
            return answer

        # Second attempt: force-answer prompt (stronger directive)
        logger.info("LLM refused with standard prompt, retrying with force prompt")
        force_messages = build_force_prompt(query, contexts)
        force_answer = self._llm.generate(force_messages) or ""

        if not _is_refusal(force_answer):
            logger.info("Force prompt succeeded")
            return force_answer

        # Both attempts failed — return the better of the two
        logger.warning("Both standard and force prompts failed — LLM still refusing")
        # The force answer might at least contain some useful text even if
        # it starts with a refusal preamble. Return whichever is longer.
        return answer if len(answer) >= len(force_answer) else force_answer

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
            cached_data = {**cached, "session_id": session_id}
            cached_resp = ChatResponse(**cached_data)
            return cached_resp

        # --- Intent classification (two-stage: centroid + optional LLM fallback) ---
        intent, intent_conf = self._intent.predict(query, llm_fallback=self._llm)

        # --- Smalltalk / greeting shortcut ---
        # Pre-retrieval routing: only catches greeting and human_agent.
        # Actual retrieval quality is checked later via should_escalate().
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
                llm_answer = self._generate_with_retry(query, contexts, use_cot=False)
                if _is_refusal(llm_answer):
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

        # --- Generate grounded answer (with retry on refusal) ---
        answer = self._generate_with_retry(query, contexts, use_cot=True)
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
            if intent == 'human_agent':
                ticket_id = create_ticket(session_id, query, decision.reason)
                resp = ChatResponse(
                    answer=f"Your query has been escalated to a human agent. A support specialist will follow up on ticket {ticket_id}. In the meantime, here's what I found:\n\n{answer}",
                    citations=citations,
                    intent=intent,
                    escalated=True,
                    confidence=min(intent_conf, grounded, 0.5),
                    session_id=session_id,
                )
            else:
                # Still provide the best answer for non-human-agent escalations
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

    def _retrieve_with_hyde(self, query: str, intent: str, intent_conf: float) -> list:
        """Retrieve context. Tries normal retrieval first, then HyDE as fallback
        when the initial retrieval score is low and the query type benefits from it."""
        query_vec = self._intent.last_query_vector

        # Step 1: Always try normal retrieval first (fast).
        contexts = self._retriever.retrieve(query, self._k, query_vector=query_vec)
        top_score = Retriever.top_score(contexts)

        # Step 2: If retrieval is good enough, return immediately.
        if top_score >= 0.5:
            return contexts

        # Step 3: Only use HyDE for specific intents where it helps.
        hyde_intents = ('how_to', 'technical', 'billing', 'account')
        use_hyde = (intent_conf >= 0.6 and intent in hyde_intents and self._llm is not None)

        if not use_hyde:
            return contexts

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
                hyde_vec = self._retriever.embedder.encode([hyde_answer])[0].tolist()
                hyde_contexts = self._retriever.retrieve(query, self._k, query_vector=hyde_vec)
                # Only use HyDE results if they're better than direct retrieval.
                if Retriever.top_score(hyde_contexts) > top_score:
                    logger.debug("HyDE improved retrieval: %.3f -> %.3f", top_score, Retriever.top_score(hyde_contexts))
                    return hyde_contexts
        except Exception:
            logger.debug("HyDE generation failed", exc_info=True)

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
