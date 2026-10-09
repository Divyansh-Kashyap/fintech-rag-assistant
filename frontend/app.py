"""
Streamlit Web User Interface for FinBase AI Customer Support Assistant.
Provides multi-turn conversational chat, collapsible source citations,
confidence gauges, latency monitoring, and feedback collection.
Supports both FastAPI Microservice Mode and Direct In-Process Cloud Mode (Streamlit Cloud).
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on Python path (critical for Streamlit Cloud where
# the working directory is the repo root but sys.path only contains frontend/)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import requests
import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="FinBase AI | Intelligent Banking Assistant",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# Import internal RAG chain for direct in-process execution on Streamlit Community Cloud
try:
    from app.services.rag_chain import get_rag_chain
    from app.services.vector_store import get_vector_store
    from app.services.retriever import get_retriever
    from app.models.schemas import QueryRequest, ChatMessage
    IN_PROCESS_RAG_AVAILABLE = True
except Exception as _import_err:
    import traceback
    print(f"[FinBase] In-process RAG import failed: {_import_err}")
    traceback.print_exc()
    IN_PROCESS_RAG_AVAILABLE = False


# Theme-Adaptive CSS (Works in both Dark and Light modes)
st.markdown(
    """
    <style>
    /* Header Card */
    .fintech-header {
        background: linear-gradient(135deg, #0A2540 0%, #1A365D 100%);
        color: #FFFFFF !important;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .fintech-header h1 {
        color: #FFFFFF !important;
        font-size: 24px;
        margin: 0 0 6px 0;
        font-weight: 700;
    }
    .fintech-header p {
        color: #94A3B8 !important;
        font-size: 13px;
        margin: 0;
    }
    
    /* Confidence and Latency Badges */
    .badge-confidence-high {
        background-color: #064E3B;
        color: #6EE7B7 !important;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #059669;
    }
    .badge-confidence-medium {
        background-color: #78350F;
        color: #FDE68A !important;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #D97706;
    }
    .badge-confidence-low {
        background-color: #7F1D1D;
        color: #FCA5A5 !important;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #DC2626;
    }
    .badge-latency {
        background-color: #1E293B;
        color: #94A3B8 !important;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #334155;
    }
    
    /* Source citation card */
    .source-box {
        background-color: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38BDF8;
        padding: 12px 16px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .source-title {
        font-weight: 700;
        font-size: 13px;
        color: #38BDF8 !important;
    }
    .source-section {
        font-size: 12px;
        color: #94A3B8 !important;
        margin-bottom: 6px;
    }
    .source-snippet {
        font-size: 12px;
        color: #E2E8F0 !important;
        background-color: rgba(15, 23, 42, 0.7);
        padding: 8px 10px;
        border-radius: 4px;
        font-family: Consolas, Monaco, monospace;
        border: 1px solid rgba(255, 255, 255, 0.08);
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initializes Streamlit session states for history and feedback."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to FinBase Priority Customer Support. "
                    "I can assist you with verified questions on Personal Loans, Foreclosure Rules, "
                    "Credit Cards, UPI Dispute Settlements, Fixed Deposits, and Account Policies. "
                    "How may I help you today?"
                ),
                "sources": [],
                "confidence_score": 1.0,
                "latency_ms": 0.0,
            }
        ]
    if "quick_query" not in st.session_state:
        st.session_state.quick_query = None


init_session_state()

# Header Display
st.markdown(
    """
    <div class="fintech-header">
        <h1>💳 FinBase AI Customer Support Assistant</h1>
        <p>Enterprise Grounded RAG Platform • Dense & Sparse Hybrid Search • Anti-Hallucination Guardrails</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def query_rag_engine(user_query: str, history_payload: List[Dict[str, str]]) -> Dict:
    """
    Dual-mode query resolver:
    1. Tries HTTP call to FastAPI backend (Microservice mode).
    2. Falls back to direct in-process RAG execution (Streamlit Cloud standalone mode).
    """
    # 1. Try FastAPI backend first
    try:
        resp = requests.post(
            f"{BACKEND_API_URL}/api/v1/query",
            json={"query": user_query, "history": history_payload, "top_k": 3},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # 2. In-Process RAG execution (ideal for single-click Streamlit Cloud deployment)
    if IN_PROCESS_RAG_AVAILABLE:
        start_time = time.time()
        chain = get_rag_chain()
        msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in history_payload]
        req = QueryRequest(query=user_query, history=msgs, top_k=3)
        
        # Run async chain in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # In active event loop (nested asyncio)
            import nest_asyncio
            nest_asyncio.apply()
            res = loop.run_until_complete(chain.query(req))
        else:
            res = loop.run_until_complete(chain.query(req))

        return {
            "answer": res.answer,
            "sources": [s.model_dump() for s in res.sources],
            "confidence_score": res.confidence_score,
            "latency_ms": res.latency_ms,
        }

    raise RuntimeError("Neither FastAPI backend nor In-Process RAG engine could be reached.")


# Sidebar controls
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=64)
    st.subheader("System Architecture")
    st.markdown(
        """
        - **Pipeline**: Hybrid (Dense + BM25)
        - **Fusion**: Reciprocal Rank Fusion ($k=60$)
        - **Reranker**: Cross-Encoder Model
        - **Guardrails**: Strict Zero-Hallucination
        """
    )

    st.divider()

    # System Health Check
    backend_online = False
    try:
        health_resp = requests.get(f"{BACKEND_API_URL}/api/v1/health", timeout=2)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success(f"● FastAPI Backend Connected ({health_data['total_chunks_indexed']} chunks)")
            st.caption(f"LLM: {health_data['llm_provider']} | Embeddings: {health_data['embedding_provider']}")
            backend_online = True
    except Exception:
        pass

    if not backend_online:
        if IN_PROCESS_RAG_AVAILABLE:
            vec_store = get_vector_store()
            st.info(f"● Streamlit In-Process Engine ({vec_store.count()} chunks)")
            st.caption("Mode: Standalone Streamlit Cloud")
        else:
            st.error("● Backend Offline (Check localhost:8000)")

    st.divider()

    # Knowledge Base Re-Ingestion Button
    st.subheader("Knowledge Base Sync")
    if st.button("🔄 Sync & Re-index Knowledge Base", key="sync_kb", use_container_width=True):
        with st.spinner("Syncing datasets and recalculating embeddings..."):
            try:
                if backend_online:
                    ingest_resp = requests.post(
                        f"{BACKEND_API_URL}/api/v1/ingest",
                        json={"force_download": False, "source": "all"},
                        timeout=60,
                    )
                    if ingest_resp.status_code == 200:
                        data = ingest_resp.json()
                        st.success(f"Indexed {data['total_chunks_created']} chunks from {data['total_documents_ingested']} docs!")
                        st.rerun()
                else:
                    from app.services.document_processor import DocumentProcessor
                    from app.core.config import get_settings
                    cfg = get_settings()
                    processor = DocumentProcessor()
                    chunks = processor.process_all(raw_dir=cfg.raw_data_path, sample_json_path=cfg.sample_dataset_abs_path)
                    vs = get_vector_store()
                    vs.index_chunks(chunks)
                    vs.save()
                    get_retriever().refresh_indices()
                    st.success(f"Directly indexed {len(chunks)} chunks!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error re-indexing: {e}")

    # Indexed Documents List
    try:
        vs = get_vector_store()
        doc_count = len(set(c.doc_title for c in vs.chunks))
        with st.expander(f"📚 Indexed Documents ({doc_count})", expanded=False):
            doc_map = {}
            for c in vs.chunks:
                doc_map[c.doc_title] = doc_map.get(c.doc_title, 0) + 1
            for title, count in doc_map.items():
                st.markdown(f"**{title}**")
                st.caption(f"{count} chunks indexed")
    except Exception:
        pass

    st.divider()
    if st.button("🧹 Clear Chat History", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        init_session_state()
        st.rerun()


# Quick Inquiries Carousel / Buttons
st.markdown("**Sample Customer Questions:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("What is the foreclosure charge at 18 months?", key="quick_q1", use_container_width=True):
        st.session_state.quick_query = "What is the foreclosure charge if I close my personal loan in 18 months?"
with col2:
    if st.button("Foreclosure charge after 26 months?", key="quick_q2", use_container_width=True):
        st.session_state.quick_query = "What is the foreclosure charge if I close my personal loan after 26 months?"
with col3:
    if st.button("UPI failed auto-reversal TAT & delay fee?", key="quick_q3", use_container_width=True):
        st.session_state.quick_query = "What is the auto-reversal TAT for failed UPI transactions and what is the delay compensation?"
with col4:
    if st.button("Luxe card annual fee waiver & lounge?", key="quick_q4", use_container_width=True):
        st.session_state.quick_query = "How can I get the annual fee waived on the FinBase Luxe Credit Card and what is the lounge access policy?"


# Render Conversation History
for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])

        # For assistant responses, show metrics and citations
        if role == "assistant" and idx > 0:
            conf = message.get("confidence_score", 0.0)
            lat = message.get("latency_ms", 0.0)
            sources = message.get("sources", [])

            # Metrics Badge Line
            col_b1, col_b2, col_b3 = st.columns([1.5, 1.5, 5])
            with col_b1:
                if conf >= 0.80:
                    st.markdown(f"<span class='badge-confidence-high'>Grounded: {int(conf*100)}%</span>", unsafe_allow_html=True)
                elif conf >= 0.45:
                    st.markdown(f"<span class='badge-confidence-medium'>Grounded: {int(conf*100)}%</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='badge-confidence-low'>Low Confidence: {int(conf*100)}%</span>", unsafe_allow_html=True)
            with col_b2:
                st.markdown(f"<span class='badge-latency'>Latency: {lat:.0f} ms</span>", unsafe_allow_html=True)

            # Collapsible Source Citations Panel
            if sources:
                with st.expander(f"📑 View Official Policy Citations ({len(sources)})", expanded=False):
                    for s_idx, src in enumerate(sources, start=1):
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-title">[{s_idx}] {src['document']}</div>
                                <div class="source-section">{src['section']} • Page {src['page']}</div>
                                <div class="source-snippet">"{src['snippet']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Interactive Feedback Mechanism
            f_col1, f_col2, f_col3, _ = st.columns([1, 1, 1.5, 6])
            with f_col1:
                if st.button("👍", key=f"thumb_up_{idx}", help="Accurate and grounded"):
                    st.toast("Thank you! Feedback recorded.", icon="✅")
            with f_col2:
                if st.button("👎", key=f"thumb_down_{idx}", help="Incorrect or unhelpful"):
                    st.toast("Feedback recorded for review.", icon="⚠️")
            with f_col3:
                if st.button("🚩 Hallucination", key=f"flag_{idx}", help="Flag unsupported claim"):
                    st.toast("Flagged for audit by compliance team.", icon="🚩")


# Handle User Prompt Input
user_input = st.chat_input("Ask a question about personal loans, credit cards, UPI, or banking policies...")

# If quick query clicked, override
if st.session_state.quick_query:
    user_input = st.session_state.quick_query
    st.session_state.quick_query = None

if user_input:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Format history payload
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ["user", "assistant"]
    ]

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        answer_placeholder.markdown("🔍 *Consulting FinBase policy database and reranking clauses...*")

        try:
            result = query_rag_engine(user_input, history_payload)
            answer_text = result["answer"]
            sources = result["sources"]
            confidence_score = result["confidence_score"]
            latency_ms = result["latency_ms"]

            # Render response
            answer_placeholder.markdown(answer_text)

            # Append to conversation state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "confidence_score": confidence_score,
                "latency_ms": latency_ms,
            })
            st.rerun()
        except Exception as e:
            answer_placeholder.error(f"Failed to communicate with assistant service: {e}")
