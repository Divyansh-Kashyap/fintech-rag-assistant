"""
Comprehensive Pytest Integration and Unit Test Suite for FinTech RAG Assistant.
Tests document parsing, chunking, dense/sparse search, RRF fusion,
anti-hallucination guardrails, and FastAPI endpoints.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import DocumentChunk, QueryRequest
from app.services.document_processor import DocumentProcessor, RecursiveCharacterTextSplitter
from app.services.rag_chain import FALLBACK_MESSAGE, get_rag_chain
from app.services.retriever import get_retriever
from app.services.vector_store import DeterministicEmbeddingProvider, VectorStore

client = TestClient(app)


# ------------------------------------------------------------------------------
# 1. Document Processor & Chunking Tests
# ------------------------------------------------------------------------------

def test_clean_text_normalizes_artifacts():
    """Validates that text cleaning removes noisy artifacts, handles currency and whitespace."""
    raw = "ReportLab Generated PDF document\n\n\nLoan  Policy \xa0\xa0 ■500   fee.\r\nNext line."
    cleaned = DocumentProcessor.clean_text(raw)
    assert "ReportLab Generated PDF" not in cleaned
    assert "Rs 500" in cleaned
    assert "  " not in cleaned
    assert "\r" not in cleaned


def test_recursive_character_splitting():
    """Validates recursive splitting within maximum chunk bounds and with overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "Personal loan clause 1. Borrowers must be 21 years old. "
        "Section 2 details minimum salary requirement of Rs 25,000 per month. "
        "Section 3 covers foreclosure penalties applicable before 24 months."
    )
    chunks = splitter.split_text(sample_text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 120  # chunk_size + overlap


# ------------------------------------------------------------------------------
# 2. Vector Store & Dense Retrieval Tests
# ------------------------------------------------------------------------------

def test_deterministic_embedding_provider():
    """Validates that deterministic embeddings produce normalized 384-d vectors."""
    provider = DeterministicEmbeddingProvider(dimension=384)
    vec = provider.embed_query("foreclosure charges personal loan")
    assert vec.shape == (384,)
    import numpy as np
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4


def test_vector_store_indexing_and_search():
    """Validates vector store indexing and cosine similarity retrieval."""
    vs = VectorStore()
    chunk1 = DocumentChunk(
        chunk_id="c1",
        content="Foreclosure charge before 24 months is 3% plus GST.",
        doc_id="DOC1",
        doc_title="Personal Loan Policy",
        section="Section 4.2",
        page_no=7,
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        content="UPI failed transaction auto-reversal TAT is T+2 business days.",
        doc_id="DOC2",
        doc_title="UPI SOP",
        section="Section 1.1",
        page_no=2,
    )
    vs.index_chunks([chunk1, chunk2])
    assert vs.count() == 2

    results = vs.search_dense("foreclosure charge 18 months", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert score > 0.5


# ------------------------------------------------------------------------------
# 3. Hybrid Retriever & RRF Tests
# ------------------------------------------------------------------------------

def test_sparse_bm25_search():
    """Validates sparse BM25 retrieval ranking."""
    retriever = get_retriever()
    results = retriever.search_sparse("foreclosure charge", top_k=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "foreclosure" in top_chunk.content.lower() or "loan" in top_chunk.content.lower()


def test_reciprocal_rank_fusion_logic():
    """Validates RRF mathematical fusion combining two distinct ranking lists."""
    retriever = get_retriever()
    c1 = DocumentChunk(chunk_id="A", content="A", doc_id="1", doc_title="T", section="S", page_no=1)
    c2 = DocumentChunk(chunk_id="B", content="B", doc_id="1", doc_title="T", section="S", page_no=1)

    dense_ranks = [(c1, 0.9), (c2, 0.7)]
    sparse_ranks = [(c2, 0.8), (c1, 0.5)]

    fused = retriever.reciprocal_rank_fusion(dense_ranks, sparse_ranks, rrf_k=60)
    assert len(fused) == 2
    assert 0.0 <= fused[0][1] <= 1.0
    assert 0.0 <= fused[1][1] <= 1.0


# ------------------------------------------------------------------------------
# 4. RAG Chain & Anti-Hallucination Guardrail Tests
# ------------------------------------------------------------------------------

def test_in_domain_query_answering():
    """Validates that a verified financial query returns a factual answer with citations."""
    import asyncio
    chain = get_rag_chain()
    req = QueryRequest(query="What is the foreclosure charge if I close my personal loan in 18 months?")
    res = asyncio.run(chain.query(req))

    assert res.answer != FALLBACK_MESSAGE
    assert "3" in res.answer or "foreclosure" in res.answer.lower()
    assert res.confidence_score >= 0.50
    assert len(res.sources) >= 1
    assert res.sources[0].document != ""
    assert res.sources[0].section != ""
    assert res.sources[0].page >= 1


def test_out_of_domain_strict_fallback():
    """Validates that an irrelevant question strictly triggers zero-hallucination fallback."""
    import asyncio
    chain = get_rag_chain()
    req = QueryRequest(query="What is the warranty policy on an Apple MacBook Pro laptop?")
    res = asyncio.run(chain.query(req))

    assert res.answer == FALLBACK_MESSAGE
    assert res.confidence_score == 0.0
    assert len(res.sources) == 0


# ------------------------------------------------------------------------------
# 5. FastAPI Endpoints Integration Tests
# ------------------------------------------------------------------------------

def test_api_health():
    """Tests GET /api/v1/health."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["total_chunks_indexed"] > 0


def test_api_query_endpoint():
    """Tests POST /api/v1/query for a valid in-domain question."""
    payload = {
        "query": "What is the auto-reversal TAT for failed UPI transactions?",
        "top_k": 3,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "T+2" in data["answer"] or "revers" in data["answer"].lower()
    assert data["confidence_score"] > 0.0
    assert len(data["sources"]) > 0


def test_api_feedback_endpoint():
    """Tests POST /api/v1/feedback recording."""
    payload = {
        "query": "What is the foreclosure charge at 18 months?",
        "answer": "3% of outstanding principal balance",
        "feedback": "thumbs_up",
        "comments": "Accurate response matching FinBase policy.",
    }
    resp = client.post("/api/v1/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "recorded"
    assert "feedback_id" in data


def test_api_documents_endpoint():
    """Tests GET /api/v1/documents summary."""
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_documents"] >= 6
    assert data["total_chunks"] > 0
