"""
RAG Orchestration, Prompt Guardrails, Grounded Generation, and LLM Engine.
Enforces strict hallucination guardrails, structured JSON outputs,
and explicit section-level source citations.
"""

import json
import logging
import re
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from app.core.config import get_settings
from app.models.schemas import DocumentChunk, QueryRequest, QueryResponse, SourceCitation
from app.services.retriever import HybridRetriever, get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

FALLBACK_MESSAGE = "Information not available in the knowledge base."

SYSTEM_PROMPT = """You are FinBase's Principal Financial Assistant, a strictly grounded AI customer support specialist.

CRITICAL OPERATIONAL RULES:
1. Answer ONLY using the facts explicitly stated in the provided context snippets below.
2. Do NOT extrapolate, speculate, or introduce external banking or financial knowledge.
3. If the context does NOT contain the factual answer to the customer's question, or if the question is unrelated to the provided FinBase policies, you MUST respond EXACTLY with:
   "Information not available in the knowledge base."
4. Every factual assertion must be attributed to a specific source in the 'sources' array.
5. Your output MUST be valid JSON adhering strictly to this schema:
{
  "answer": "Grounded answer text (or 'Information not available in the knowledge base.')",
  "sources": [
    {
      "document": "Document Title",
      "section": "Section Heading",
      "page": 1,
      "snippet": "Verbatim quote from context"
    }
  ],
  "confidence_score": 0.95
}
"""


class RAGChain:
    """End-to-end RAG orchestrator with retrieval, guardrailed prompting, and synthesis."""

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or get_retriever()

    def _build_context_prompt(self, candidates: List[Tuple[DocumentChunk, float]]) -> str:
        """Formats retrieved chunks into numbered context blocks for prompt injection."""
        context_blocks = []
        for idx, (chunk, score) in enumerate(candidates, start=1):
            block = (
                f"[Source {idx}]\n"
                f"Document: {chunk.doc_title}\n"
                f"Section: {chunk.section}\n"
                f"Page: {chunk.page_no}\n"
                f"Relevance Score: {score:.2f}\n"
                f"Content:\n{chunk.content}\n"
            )
            context_blocks.append(block)
        return "\n--------------------\n".join(context_blocks)

    def _call_openai(self, system_msg: str, user_msg: str) -> Dict:
        """Executes completion via OpenAI API with forced JSON response format."""
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": settings.TEMPERATURE,
            "response_format": {"type": "json_object"},
            "max_tokens": settings.MAX_OUTPUT_TOKENS,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"]
            return json.loads(raw_text)

    def _call_gemini(self, system_msg: str, user_msg: str) -> Dict:
        """Executes completion via Google Gemini API with JSON output mode."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent"
            f"?key={settings.GEMINI_API_KEY}"
        )
        combined_prompt = f"{system_msg}\n\nUser Question:\n{user_msg}"
        payload = {
            "contents": [{"parts": [{"text": combined_prompt}]}],
            "generationConfig": {
                "temperature": settings.TEMPERATURE,
                "responseMimeType": "application/json",
                "maxOutputTokens": settings.MAX_OUTPUT_TOKENS,
            },
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            text_part = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text_part)

    def _local_grounded_synthesis(
        self,
        query: str,
        candidates: List[Tuple[DocumentChunk, float]],
    ) -> Dict:
        """
        Deterministic, zero-hallucination local extraction engine.
        Synthesizes answers directly from top reranked chunks without requiring an external LLM.
        Strictly falls back to 'Information not available in the knowledge base.' if query relevance is low.
        """
        if not candidates or candidates[0][1] < settings.MIN_CONFIDENCE_THRESHOLD:
            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [],
                "confidence_score": 0.0,
            }

        top_chunk, top_score = candidates[0]
        query_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        content = top_chunk.content

        # 1. If chunk is a pre-formatted QA pair, verify high query-question alignment
        if top_chunk.metadata.get("is_qa_pair") and "Official Policy Answer:" in content:
            q_meta = top_chunk.metadata.get("question", "").lower()
            q_terms = [t for t in re.findall(r"\w+", q_meta) if len(t) > 2]
            common_q_terms = set(query_terms).intersection(set(q_terms))

            # Must match at least 2 key terms or 40% of query
            if len(common_q_terms) >= 2 or (len(query_terms) > 0 and len(common_q_terms) / len(query_terms) >= 0.40):
                parts = content.split("Official Policy Answer:")
                if len(parts) > 1:
                    ans_text = parts[1].strip()
                    return {
                        "answer": f"According to {top_chunk.section} of {top_chunk.doc_title}, {ans_text}",
                        "sources": [
                            {
                                "document": top_chunk.doc_title,
                                "section": top_chunk.section,
                                "page": top_chunk.page_no,
                                "snippet": ans_text[:200],
                            }
                        ],
                        "confidence_score": round(min(0.98, top_score), 2),
                    }

        # 2. Extract best matching sentence from document content
        sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", content) if len(s.strip()) > 15]
        best_sentence = ""
        best_overlap = 0

        for sentence in sentences:
            s_terms = set(re.findall(r"\w+", sentence.lower()))
            overlap = len(set(query_terms).intersection(s_terms))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

        # Require significant term overlap with sentence
        if best_sentence and best_overlap >= 3:
            answer = f"According to {top_chunk.section} ({top_chunk.doc_title}), {best_sentence}"
            sources = [
                {
                    "document": top_chunk.doc_title,
                    "section": top_chunk.section,
                    "page": top_chunk.page_no,
                    "snippet": best_sentence[:200],
                }
            ]
            if len(candidates) > 1 and candidates[1][1] >= 0.65:
                c2, _ = candidates[1]
                sources.append(
                    {
                        "document": c2.doc_title,
                        "section": c2.section,
                        "page": c2.page_no,
                        "snippet": c2.content[:150],
                    }
                )

            return {
                "answer": answer,
                "sources": sources,
                "confidence_score": round(min(0.95, top_score), 2),
            }

        # If overlap is too weak, trigger strict fallback
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "confidence_score": 0.0,
        }

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Executes end-to-end RAG query flow:
        1. Retrieval & Reranking
        2. Confidence Gate & Guardrails
        3. LLM Generation
        4. Structured Output Validation
        """
        start_time = time.time()

        # Step 1: Hybrid Retrieval + Cross-Encoder Reranking
        candidates, expanded_query = self.retriever.retrieve(
            query=request.query,
            history=request.history,
            top_k=request.top_k or settings.TOP_K_RERANK,
        )

        latency_base = (time.time() - start_time) * 1000.0

        # Step 2: Guardrail - Check minimum relevance threshold
        top_score = candidates[0][1] if candidates else 0.0
        if not candidates or top_score < settings.MIN_CONFIDENCE_THRESHOLD:
            logger.info("Low retrieval confidence (%.2f < %.2f) for query: '%s'. Returning fallback.",
                        top_score, settings.MIN_CONFIDENCE_THRESHOLD, request.query)
            return QueryResponse(
                answer=FALLBACK_MESSAGE,
                sources=[],
                confidence_score=0.0,
                latency_ms=round(latency_base, 2),
                query_expanded=expanded_query if expanded_query != request.query else None,
            )

        # Step 3: Synthesis via configured provider
        provider = settings.LLM_PROVIDER.lower()
        context_str = self._build_context_prompt(candidates)
        user_prompt = f"Retrieved Financial Policy Context:\n{context_str}\n\nCustomer Inquiry: {request.query}"

        output_data: Optional[Dict] = None

        if provider == "openai" and settings.OPENAI_API_KEY:
            try:
                output_data = self._call_openai(SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                logger.error("OpenAI API call failed (%s). Falling back to local synthesis.", e)

        elif provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                output_data = self._call_gemini(SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                logger.error("Gemini API call failed (%s). Falling back to local synthesis.", e)

        # Fallback to local synthesis if LLM failed or in mock/offline mode
        if not output_data:
            output_data = self._local_grounded_synthesis(request.query, candidates)

        total_latency = (time.time() - start_time) * 1000.0

        # Parse citations
        sources_list: List[SourceCitation] = []
        for s in output_data.get("sources", []):
            try:
                sources_list.append(
                    SourceCitation(
                        document=s.get("document", "FinBase Policy"),
                        section=s.get("section", "General Section"),
                        page=int(s.get("page", 1)),
                        snippet=s.get("snippet", ""),
                    )
                )
            except Exception as e:
                logger.warning("Error parsing source citation: %s", e)

        answer_text = output_data.get("answer", FALLBACK_MESSAGE)
        conf_score = float(output_data.get("confidence_score", top_score))

        # Enforce zero-sources if fallback is outputted
        if answer_text == FALLBACK_MESSAGE:
            sources_list = []
            conf_score = 0.0

        return QueryResponse(
            answer=answer_text,
            sources=sources_list,
            confidence_score=conf_score,
            latency_ms=round(total_latency, 2),
            query_expanded=expanded_query if expanded_query != request.query else None,
        )

    async def query_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """Streams response tokens as Server-Sent Events (SSE)."""
        response = await self.query(request)
        answer = response.answer

        words = answer.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            chunk = {"type": "token", "content": token}
            yield f"data: {json.dumps(chunk)}\n\n"

        final_meta = {
            "type": "done",
            "sources": [s.model_dump() for s in response.sources],
            "confidence_score": response.confidence_score,
            "latency_ms": response.latency_ms,
            "query_expanded": response.query_expanded,
        }
        yield f"data: {json.dumps(final_meta)}\n\n"


_global_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Singleton getter for RAGChain."""
    global _global_rag_chain
    if _global_rag_chain is None:
        _global_rag_chain = RAGChain()
    return _global_rag_chain
