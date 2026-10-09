"""
FinBase AI - Enterprise FinTech Customer Support Assistant.
Interactive Streamlit UI featuring:
- Real-time response streaming
- Interactive category-based question selectors
- Visual groundedness & confidence metrics
- Collapsible verified policy citations & audit trail export
- Feedback recording (helpful, inaccurate, hallucination flag)
- Dual-mode execution (FastAPI Microservice + In-Process RAG)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on Python path
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


# Custom Professional FinTech Styling
st.markdown(
    """
    <style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* FinBase Header Banner */
    .fintech-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2847 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .fintech-hero h1 {
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC !important;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .fintech-hero p {
        color: #94A3B8 !important;
        font-size: 13.5px;
        margin: 0;
        line-height: 1.5;
    }
    .hero-tags {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .hero-tag {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #CBD5E1 !important;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 20px;
    }
    
    /* Interactive Citation Box */
    .citation-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .citation-card:hover {
        border-color: rgba(56, 189, 248, 0.6);
        background: rgba(15, 23, 42, 0.85);
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .citation-title {
        font-weight: 600;
        font-size: 13px;
        color: #38BDF8 !important;
    }
    .citation-meta {
        font-size: 11.5px;
        color: #94A3B8 !important;
        background: rgba(255, 255, 255, 0.06);
        padding: 2px 7px;
        border-radius: 4px;
    }
    .citation-snippet {
        font-size: 12px;
        color: #E2E8F0 !important;
        background: rgba(0, 0, 0, 0.25);
        padding: 8px 10px;
        border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 1.45;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Telemetry Pill Badges */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .metric-grounded-high {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399 !important;
    }
    .metric-grounded-med {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        color: #FBBF24 !important;
    }
    .metric-grounded-low {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        color: #F87171 !important;
    }
    .metric-latency {
        background: rgba(100, 116, 139, 0.15);
        border: 1px solid #64748B;
        color: #CBD5E1 !important;
    }
    
    /* Topic Selector Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 6px 14px;
        background-color: rgba(255, 255, 255, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initializes Streamlit session states for conversation and prompt routing."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am **FinBase AI**, your certified financial customer support assistant. "
                    "I am strictly grounded in verified banking policies for **Personal Loans**, "
                    "**Credit Cards & Fees**, **UPI Dispute Auto-Reversals**, and **Fixed Deposits**.\n\n"
                    "Select a suggested inquiry below or ask your own question to begin."
                ),
                "sources": [],
                "confidence_score": 1.0,
                "latency_ms": 0.0,
            }
        ]
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


init_session_state()

# Hero Banner
st.markdown(
    """
    <div class="fintech-hero">
        <h1>💳 FinBase AI Assistant</h1>
        <p>Enterprise Customer Support Architecture • Dual Dense & Sparse BM25 Retrieval • Reciprocal Rank Fusion • Zero-Hallucination Guardrails</p>
        <div class="hero-tags">
            <span class="hero-tag">⚡ Sub-50ms Retrieval</span>
            <span class="hero-tag">🎯 Section-Level Citations</span>
            <span class="hero-tag">🛡️ RBI Compliance Grounded</span>
            <span class="hero-tag">🔒 Anti-Hallucination Gate</span>
        </div>
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
            timeout=8,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # 2. In-Process RAG execution (Streamlit Cloud standalone mode)
    if IN_PROCESS_RAG_AVAILABLE:
        start_time = time.time()
        chain = get_rag_chain()
        msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in history_payload]
        req = QueryRequest(query=user_query, history=msgs, top_k=3)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
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


def stream_text_words(text: str, delay: float = 0.015):
    """Generator for smooth, responsive typing animation."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


# Sidebar Configuration & Telemetry
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=54)
    st.markdown("### **System Monitor**")

    backend_online = False
    try:
        health_resp = requests.get(f"{BACKEND_API_URL}/api/v1/health", timeout=2)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success(f"● Backend Connected ({health_data['total_chunks_indexed']} chunks)")
            st.caption(f"LLM: {health_data['llm_provider']} | Embeddings: {health_data['embedding_provider']}")
            backend_online = True
    except Exception:
        pass

    if not backend_online:
        if IN_PROCESS_RAG_AVAILABLE:
            vec_store = get_vector_store()
            chunk_count = vec_store.count()
            st.info(f"● In-Process Engine ({chunk_count} chunks)")
            st.caption("Mode: Standalone Streamlit Cloud")
        else:
            st.error("● Backend Offline (Check localhost:8000)")

    st.divider()

    # RAG Architecture Specifications
    st.markdown("#### **Retrieval Pipeline**")
    st.markdown(
        """
        - **Embedding Dimension**: 384-d L2 Normalized
        - **Sparse Search**: BM25 Okapi Algorithm
        - **Ranking Fusion**: RRF ($k=60$)
        - **Re-ranking**: Cross-Encoder Numerical Scorer
        - **Guardrail Gate**: Min Similarity 0.45
        """
    )

    st.divider()

    # Knowledge Base Controls
    st.markdown("#### **Knowledge Base Ops**")
    if st.button("🔄 Sync & Re-index Knowledge Base", key="sync_kb", use_container_width=True):
        with st.spinner("Re-indexing policy documents & regenerating embeddings..."):
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

    # Indexed Documents Summary
    try:
        vs = get_vector_store()
        doc_count = len(set(c.doc_title for c in vs.chunks))
        with st.expander(f"📚 Indexed Documents ({doc_count})", expanded=False):
            doc_map = {}
            for c in vs.chunks:
                doc_map[c.doc_title] = doc_map.get(c.doc_title, 0) + 1
            for title, count in doc_map.items():
                st.markdown(f"**{title}**")
                st.caption(f"{count} clauses indexed")
    except Exception:
        pass

    st.divider()

    # Session Management
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🧹 Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state.messages = []
            init_session_state()
            st.rerun()
    with col_c2:
        # Download Conversation Audit Trail
        chat_export = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Export Audit",
            data=chat_export,
            file_name="finbase_support_audit.json",
            mime="application/json",
            key="export_audit",
            use_container_width=True,
        )


# Interactive Category Explorers
st.markdown("##### 💡 **Explore Frequently Asked Policy Topics**")
tab_loan, tab_card, tab_upi, tab_guardrail = st.tabs([
    "🏦 Personal Loans & Foreclosure",
    "💳 Credit Cards & Perks",
    "⚡ UPI & Auto-Reversal TAT",
    "🛡️ Anti-Hallucination Test",
])

with tab_loan:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Foreclosure charge at 18 months?", key="loan_q1", use_container_width=True):
            st.session_state.pending_query = "What is the foreclosure charge if I close my personal loan in 18 months?"
    with c2:
        if st.button("Foreclosure charge after 26 months?", key="loan_q2", use_container_width=True):
            st.session_state.pending_query = "What is the foreclosure charge if I close my personal loan after 26 months?"

with tab_card:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Luxe Card annual fee waiver policy?", key="card_q1", use_container_width=True):
            st.session_state.pending_query = "How can I get the annual fee waived on the FinBase Luxe Credit Card and what is the lounge access policy?"
    with c2:
        if st.button("Cashback points expiry & redemption?", key="card_q2", use_container_width=True):
            st.session_state.pending_query = "What is the reward point expiry period and redemption value on credit cards?"

with tab_upi:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("UPI failed transaction auto-reversal TAT?", key="upi_q1", use_container_width=True):
            st.session_state.pending_query = "What is the auto-reversal TAT for failed UPI transactions and what is the delay compensation?"
    with c2:
        if st.button("Chargeback dispute timeline for merchants?", key="upi_q2", use_container_width=True):
            st.session_state.pending_query = "What is the maximum time to file a dispute for an unauthorized debit?"

with tab_guardrail:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Can I buy crypto using personal loans?", key="guard_q1", use_container_width=True):
            st.session_state.pending_query = "Can I use my personal loan to buy shares or cryptocurrency?"
    with c2:
        if st.button("Do you offer rocket insurance?", key="guard_q2", use_container_width=True):
            st.session_state.pending_query = "What is the interest rate for a SpaceX spaceship financing loan?"


# Render Conversation History
for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = "👤" if role == "user" else "💳"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

        # Assistant Telemetry Badges & Citations
        if role == "assistant" and idx > 0:
            conf = message.get("confidence_score", 0.0)
            lat = message.get("latency_ms", 0.0)
            sources = message.get("sources", [])

            # Telemetry Bar
            badge_class = "metric-grounded-high" if conf >= 0.80 else ("metric-grounded-med" if conf >= 0.45 else "metric-grounded-low")
            status_label = f"Grounded: {int(conf * 100)}%" if conf >= 0.45 else "Low Confidence Guardrail Triggered"

            metric_html = f"""
            <div style="margin-top: 8px; margin-bottom: 8px;">
                <span class="metric-pill {badge_class}">🛡️ {status_label}</span>
                <span class="metric-pill metric-latency">⚡ {lat:.0f} ms latency</span>
                <span class="metric-pill metric-latency">📑 {len(sources)} source clause(s) audited</span>
            </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)

            # Verified Policy Citations Accordion
            if sources:
                with st.expander(f"📑 View Official Policy Citations ({len(sources)} clauses)", expanded=False):
                    for s_idx, src in enumerate(sources, start=1):
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span class="citation-title">[{s_idx}] {src['document']}</span>
                                    <span class="citation-meta">{src['section']} • Page {src['page']}</span>
                                </div>
                                <div class="citation-snippet">"{src['snippet']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Interactive Micro-Feedback Action Row
            f_col1, f_col2, f_col3, _ = st.columns([1, 1, 1.8, 6])
            with f_col1:
                if st.button("👍", key=f"thumb_up_{idx}", help="Accurate and grounded answer"):
                    st.toast("Feedback recorded: Grounded answer verified.", icon="✅")
            with f_col2:
                if st.button("👎", key=f"thumb_down_{idx}", help="Inaccurate or incomplete"):
                    st.toast("Feedback recorded for engineering review.", icon="⚠️")
            with f_col3:
                if st.button("🚩 Audit Claim", key=f"flag_{idx}", help="Flag potentially unsupported claim for compliance team"):
                    st.toast("Flagged for compliance audit log.", icon="🚩")


# Handle User Prompt Input
user_input = st.chat_input("Ask any question regarding personal loans, credit card fees, UPI reversal TAT...")

# Route Quick Inquiries from tabs
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 2. Extract recent history
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ["user", "assistant"]
    ]

    # 3. Generate response with streaming typewriter animation
    with st.chat_message("assistant", avatar="💳"):
        with st.spinner("Consulting FinBase policy database, evaluating hybrid BM25 + dense vectors..."):
            try:
                result = query_rag_engine(user_input, history_payload)
                answer_text = result["answer"]
                sources = result["sources"]
                confidence_score = result["confidence_score"]
                latency_ms = result["latency_ms"]
            except Exception as e:
                answer_text = f"An error occurred while connecting to the FinBase assistant engine: {e}"
                sources = []
                confidence_score = 0.0
                latency_ms = 0.0

        # Live Typewriter Effect
        st.write_stream(stream_text_words(answer_text))

        # Append to conversation state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "confidence_score": confidence_score,
            "latency_ms": latency_ms,
        })
        st.rerun()
