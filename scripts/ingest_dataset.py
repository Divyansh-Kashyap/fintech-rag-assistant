"""
Automated Ingestion and Vector Store Syncing CLI Script.
Syncs documents from Google Drive (folder 1CX9szQJMiVBEYPqMnPxixGA8b3nOrWpc),
processes local raw PDFs/CSVs/JSONs, chunks them recursively, generates embeddings,
and persists indices to disk.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.document_processor import DocumentProcessor
from app.services.retriever import get_retriever
from app.services.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest-cli")


def main():
    parser = argparse.ArgumentParser(description="Ingest and Index Financial Knowledge Base Documents.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of files from Google Drive even if already cached locally.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to raw documents directory (defaults to settings.raw_data_path).",
    )
    parser.add_argument(
        "--sample-json",
        type=str,
        default=None,
        help="Path to sample dataset JSON (defaults to settings.sample_dataset_abs_path).",
    )

    args = parser.parse_args()
    settings = get_settings()

    raw_dir = Path(args.raw_dir) if args.raw_dir else settings.raw_data_path
    sample_json = Path(args.sample_json) if args.sample_json else settings.sample_dataset_abs_path

    print("==================================================================")
    print("  FinTech RAG Assistant - Knowledge Base Ingestion Engine        ")
    print("==================================================================")
    print(f"Target Raw Directory: {raw_dir}")
    print(f"Sample Dataset JSON : {sample_json}")
    print(f"Chunk Target Size   : {settings.CHUNK_SIZE} chars (overlap: {settings.CHUNK_OVERLAP} chars)")
    print(f"Vector Database     : {settings.VECTOR_STORE_TYPE}")
    print(f"Embedding Provider  : {settings.EMBEDDING_PROVIDER} ({settings.EMBEDDING_MODEL})")
    print("------------------------------------------------------------------")

    start_time = time.time()
    processor = DocumentProcessor()

    # 1. Sync from Google Drive
    print(f"[*] Checking Google Drive folder ({settings.GDRIVE_FOLDER_ID})...")
    downloaded = processor.sync_google_drive_folder(
        target_dir=raw_dir,
        folder_id=settings.GDRIVE_FOLDER_ID,
        force=args.force_download,
    )
    print(f"[*] Available raw document files: {len(list(raw_dir.glob('*')))} files")

    # 2. Process and Chunk Documents
    print("[*] Processing documents and generating recursive chunks...")
    chunks = processor.process_all(raw_dir=raw_dir, sample_json_path=sample_json)

    if not chunks:
        print("[!] Error: No text chunks extracted. Verify document folder contents.")
        sys.exit(1)

    print(f"[+] Successfully extracted {len(chunks)} chunks.")

    # 3. Vector Database Indexing
    print("[*] Generating embeddings and building Vector Store Index...")
    vector_store = get_vector_store()
    vector_store.index_chunks(chunks)

    # 4. Persistence
    print(f"[*] Persisting index and metadata to {settings.persist_path}...")
    vector_store.save()

    # 5. Refresh Hybrid Retriever indices
    print("[*] Initializing BM25 sparse keyword index...")
    retriever = get_retriever()
    retriever.refresh_indices()

    elapsed = time.time() - start_time
    print("------------------------------------------------------------------")
    print(f"[✓] INGESTION COMPLETED IN {elapsed:.2f} SECONDS")
    print(f"[✓] Total Chunks Indexed: {vector_store.count()}")
    print(f"[✓] Persist Directory   : {settings.persist_path.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    main()
