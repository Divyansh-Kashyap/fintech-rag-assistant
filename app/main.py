"""
FastAPI Main Application Server.
Exposes endpoints for customer inquiries, streaming SSE responses,
knowledge base ingestion, user feedback collection, and health checks.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from app.core.config import get_settings
from app.core.security import check_rate_limit, verify_api_key
from app.models.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.rag_chain import get_rag_chain
from app.services.retriever import get_retriever
from app.services.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fintech-rag")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: auto-indexes sample data on boot if needed."""
    logger.info("Initializing FinTech RAG Assistant (%s)...", settings.ENVIRONMENT)
    vector_store = get_vector_store()

    # If vector store is empty, auto-ingest available raw data and sample dataset
    if vector_store.count() == 0:
        logger.info("Vector store is empty. Running initial knowledge base ingestion...")
        processor = DocumentProcessor()
        chunks = processor.process_all(
            raw_dir=settings.raw_data_path,
            sample_json_path=settings.sample_dataset_abs_path,
        )
        if chunks:
            vector_store.index_chunks(chunks)
            vector_store.save()
            get_retriever().refresh_indices()
            logger.info("Initial ingestion complete: %d chunks indexed.", vector_store.count())
        else:
            logger.warning("No initial documents found in %s.", settings.raw_data_path)

    yield
    logger.info("Shutting down FinTech RAG Assistant service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-Ready AI-Powered FinTech Customer Support Assistant using Hybrid RAG.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Appends X-Process-Time-Ms diagnostic header to all HTTP responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


@app.get("/", include_in_schema=False)
async def root():
    """Redirects root URL directly to Swagger API docs."""
    return RedirectResponse(url="/docs")


@app.get(f"{settings.API_V1_STR}/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns application health status, indexed chunk count, and model providers."""
    vector_store = get_vector_store()
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        total_chunks_indexed=vector_store.count(),
        llm_provider=settings.LLM_PROVIDER,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        vector_store_type=settings.VECTOR_STORE_TYPE,
    )


@app.post(
    f"{settings.API_V1_STR}/query",
    response_model=QueryResponse,
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
    tags=["RAG Engine"],
)
async def query_knowledge_base(request: QueryRequest):
    """
    Executes grounded RAG question answering:
    - Hybrid Dense + Sparse BM25 retrieval with Reciprocal Rank Fusion (RRF)
    - Cross-Encoder reranking
    - Strict anti-hallucination guardrail filtering
    - Section-level source citations and confidence scoring
    """
    rag_chain = get_rag_chain()
    try:
        response = await rag_chain.query(request)
        return response
    except Exception as e:
        logger.error("Error executing query: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the customer query: {str(e)}",
        )


@app.post(
    f"{settings.API_V1_STR}/query/stream",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
    tags=["RAG Engine"],
)
async def stream_query_knowledge_base(request: QueryRequest):
    """Streams answer tokens via Server-Sent Events (SSE) with real-time citations."""
    rag_chain = get_rag_chain()
    return StreamingResponse(
        rag_chain.query_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@app.post(
    f"{settings.API_V1_STR}/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
    tags=["Ingestion"],
)
async def ingest_knowledge_base(request: IngestRequest):
    """
    Triggers knowledge base ingestion:
    - Optionally pulls fresh documents from Google Drive
    - Ingests raw PDFs, CSVs, TXT files, and sample_dataset.json
    - Splits text recursively, enriches metadata, and rebuilds indices
    """
    processor = DocumentProcessor()

    # Sync from Google Drive if requested or if raw data folder is empty
    if request.force_download or not any(settings.raw_data_path.iterdir() if settings.raw_data_path.exists() else []):
        logger.info("Syncing documents from Google Drive folder: %s", settings.GDRIVE_FOLDER_ID)
        processor.sync_google_drive_folder(
            target_dir=settings.raw_data_path,
            folder_id=settings.GDRIVE_FOLDER_ID,
            force=request.force_download,
        )

    # Ingest and process all files
    chunks = processor.process_all(
        raw_dir=settings.raw_data_path,
        sample_json_path=settings.sample_dataset_abs_path,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid documents found to ingest.",
        )

    # Reindex vector store and BM25 sparse index
    vector_store = get_vector_store()
    vector_store.index_chunks(chunks)
    vector_store.save()
    get_retriever().refresh_indices()

    indexed_files = [f.name for f in settings.raw_data_path.iterdir() if f.is_file()]
    if settings.sample_dataset_abs_path.exists():
        indexed_files.append(settings.sample_dataset_abs_path.name)

    return IngestResponse(
        status="success",
        total_documents_ingested=len(indexed_files),
        total_chunks_created=len(chunks),
        indexed_files=indexed_files,
        message=f"Successfully indexed {len(chunks)} chunks across {len(indexed_files)} files.",
    )


@app.post(
    f"{settings.API_V1_STR}/feedback",
    response_model=FeedbackResponse,
    dependencies=[Depends(check_rate_limit)],
    tags=["Monitoring & Feedback"],
)
async def submit_feedback(request: FeedbackRequest):
    """Records customer feedback (thumbs up, thumbs down, hallucination flag) to disk."""
    feedback_id = str(uuid.uuid4())
    record = {
        "feedback_id": feedback_id,
        "timestamp": time.time(),
        "query_id": request.query_id,
        "query": request.query,
        "answer": request.answer,
        "feedback": request.feedback,
        "comments": request.comments,
    }

    feedback_file = settings.persist_path / "feedback.jsonl"
    settings.persist_path.mkdir(parents=True, exist_ok=True)
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return FeedbackResponse(
        status="recorded",
        feedback_id=feedback_id,
        message="Feedback successfully recorded for quality evaluation.",
    )


@app.get(f"{settings.API_V1_STR}/documents", tags=["Knowledge Base"])
async def list_indexed_documents() -> Dict[str, Any]:
    """Lists summary of all currently indexed documents, categories, and chunk distributions."""
    vector_store = get_vector_store()
    doc_summary: Dict[str, Dict[str, Any]] = {}

    for c in vector_store.chunks:
        if c.doc_title not in doc_summary:
            doc_summary[c.doc_title] = {
                "doc_id": c.doc_id,
                "sections": set(),
                "pages": set(),
                "chunk_count": 0,
            }
        doc_summary[c.doc_title]["sections"].add(c.section)
        doc_summary[c.doc_title]["pages"].add(c.page_no)
        doc_summary[c.doc_title]["chunk_count"] += 1

    formatted = []
    for title, info in doc_summary.items():
        formatted.append({
            "doc_title": title,
            "doc_id": info["doc_id"],
            "total_chunks": info["chunk_count"],
            "sections_count": len(info["sections"]),
            "pages_count": len(info["pages"]),
        })

    return {
        "total_documents": len(formatted),
        "total_chunks": vector_store.count(),
        "documents": formatted,
    }
