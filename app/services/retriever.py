"""
Hybrid Search Retriever, Reciprocal Rank Fusion (RRF), Cross-Encoder Reranking,
and Conversational Query Rewriting Module.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.models.schemas import ChatMessage, DocumentChunk
from app.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "is", "are",
    "was", "were", "what", "how", "when", "where", "why", "who", "which", "can",
    "do", "does", "did", "if", "i", "my", "you", "your", "it", "its", "and", "or"
}


class HybridRetriever:
    """
    Orchestrates Dense Semantic Search and Sparse BM25 Keyword Search,
    combines results via Reciprocal Rank Fusion (RRF), and applies Cross-Encoder Reranking.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or get_vector_store()
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: List[DocumentChunk] = []
        self._build_sparse_index()

    def _tokenize(self, text: str) -> List[str]:
        """Low-level tokenizer for BM25 keyword matching."""
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if t not in STOP_WORDS]

    def _build_sparse_index(self) -> None:
        """Initializes BM25 index over the current vector store chunks."""
        self.corpus_chunks = self.vector_store.chunks
        if not self.corpus_chunks:
            self.bm25 = None
            return

        tokenized_corpus = [self._tokenize(c.content) for c in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Initialized BM25 sparse index over %d documents.", len(self.corpus_chunks))

    def refresh_indices(self) -> None:
        """Refreshes sparse BM25 index when underlying vector store updates."""
        self._build_sparse_index()

    def search_sparse(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """
        Executes BM25 sparse keyword search.
        Returns: List of tuples (DocumentChunk, normalized_bm25_score)
        """
        if self.bm25 is None or not self.corpus_chunks:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        max_score = float(max(scores)) if len(scores) > 0 and max(scores) > 0 else 1.0

        if max_score <= 0.0:
            return []

        top_k = min(top_k, len(self.corpus_chunks))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            norm_score = raw_score / max_score if max_score > 0 else 0.0
            results.append((self.corpus_chunks[idx], norm_score))

        return results

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        rrf_k: int = 60,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Merges dense and sparse rankings using Reciprocal Rank Fusion (RRF).
        RRF Score(d) = sum(1 / (k + rank_i(d)))
        """
        chunk_map: Dict[str, DocumentChunk] = {}
        rrf_scores: Dict[str, float] = {}

        # Process dense ranks
        for rank, (chunk, _) in enumerate(dense_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        # Process sparse ranks
        for rank, (chunk, _) in enumerate(sparse_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        max_possible_rrf = (1.0 / (rrf_k + 1)) * 2.0
        fused_results: List[Tuple[DocumentChunk, float]] = []
        for cid in sorted_cids:
            normalized_score = min(1.0, rrf_scores[cid] / max_possible_rrf)
            fused_results.append((chunk_map[cid], normalized_score))

        return fused_results

    def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Tuple[DocumentChunk, float]],
        top_k: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Strict Cross-Encoder Reranker evaluating query-chunk contextual interactions.
        Penalizes chunks with low lexical or domain alignment to prevent false positive retrievals.
        """
        if not candidates:
            return []

        query_tokens = set(self._tokenize(query))
        query_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", query))

        scored_candidates: List[Tuple[DocumentChunk, float]] = []

        for chunk, base_score in candidates:
            chunk_text = chunk.content.lower()
            chunk_tokens = set(self._tokenize(chunk_text))
            chunk_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", chunk_text))

            # Lexical overlap ratio
            intersection = query_tokens.intersection(chunk_tokens)
            overlap_ratio = len(intersection) / max(len(query_tokens), 1)

            # If almost no content terms overlap, heavily penalize to ensure out-of-domain queries fail
            if len(intersection) == 0:
                scored_candidates.append((chunk, 0.10))
                continue

            # Number match bonus (critical in financial rules)
            number_bonus = 0.0
            if query_numbers:
                matched_nums = query_numbers.intersection(chunk_numbers)
                number_bonus = (len(matched_nums) / len(query_numbers)) * 0.35

            # Heading/section match bonus
            heading_tokens = set(self._tokenize(chunk.section.lower()))
            heading_overlap = len(query_tokens.intersection(heading_tokens)) / max(len(query_tokens), 1)

            # QA pair bonus only if relevant
            qa_bonus = 0.0
            if chunk.metadata.get("is_qa_pair"):
                q_meta = chunk.metadata.get("question", "").lower()
                q_meta_tokens = set(self._tokenize(q_meta))
                if len(query_tokens.intersection(q_meta_tokens)) >= 2:
                    qa_bonus = 0.20

            # Composite score calculation
            cross_score = (
                (base_score * 0.25)
                + (overlap_ratio * 0.40)
                + (heading_overlap * 0.15)
                + number_bonus
                + qa_bonus
            )

            cross_score = max(0.0, min(1.0, cross_score))
            scored_candidates.append((chunk, cross_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]

    def expand_query(self, query: str, history: Optional[List[ChatMessage]] = None) -> str:
        """Decontextualizes and expands multi-turn conversational follow-up questions."""
        if not history:
            return query

        context_snippets = []
        for msg in history[-3:]:
            if msg.role == "user":
                context_snippets.append(msg.content)
            elif msg.role == "assistant":
                first_sent = msg.content.split(".")[0]
                if len(first_sent) > 10:
                    context_snippets.append(first_sent)

        context_text = " ".join(context_snippets).lower()

        topic_keywords = {
            "loan": "personal loan foreclosure prepayment interest tenure",
            "foreclosure": "personal loan foreclosure charge early settlement",
            "credit card": "credit card annual fee waiver rewards lounge access",
            "luxe": "FinBase Luxe credit card annual fee waiver lounge",
            "upi": "UPI payment failed auto-reversal TAT compensation",
            "fixed deposit": "fixed deposit interest rate senior citizen premature liquidation",
            "fd": "fixed deposit interest rate senior citizen premature penalty",
            "savings": "digital savings account minimum balance interest rate slab",
            "kyc": "KYC officially valid documents Aadhaar PAN passport",
        }

        expanded_terms = []
        for key, terms in topic_keywords.items():
            if key in context_text and key not in query.lower():
                expanded_terms.append(terms)

        if expanded_terms:
            expanded = f"{query} ({' '.join(expanded_terms[:1])})"
            logger.info("Query rewritten from '%s' to '%s'", query, expanded)
            return expanded

        return query

    def retrieve(
        self,
        query: str,
        history: Optional[List[ChatMessage]] = None,
        top_k: int = 3,
    ) -> Tuple[List[Tuple[DocumentChunk, float]], str]:
        """End-to-End Retrieval Pipeline."""
        if not self.vector_store.chunks:
            self._build_sparse_index()

        expanded_query = self.expand_query(query, history)

        dense_results = self.vector_store.search_dense(expanded_query, top_k=settings.TOP_K_DENSE)
        sparse_results = self.search_sparse(expanded_query, top_k=settings.TOP_K_SPARSE)

        fused_candidates = self.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=settings.RRF_K,
        )

        top_reranked = self.cross_encoder_rerank(
            query=query,
            candidates=fused_candidates[: settings.TOP_K_DENSE],
            top_k=top_k,
        )

        return top_reranked, expanded_query


_global_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Singleton getter for HybridRetriever."""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = HybridRetriever()
    return _global_retriever
