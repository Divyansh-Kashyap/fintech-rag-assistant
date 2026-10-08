"""
Document Preprocessing, Parsing, and Hierarchical Chunking Service.
Handles multi-format ingestion (PDF, CSV, JSON, TXT/MD), Google Drive syncing,
clean text extraction, and recursive character splitting with metadata enrichment.
"""

import csv
import hashlib
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdf

from app.core.config import get_settings
from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()

# Known Google Drive mappings for folder 1CX9szQJMiVBEYPqMnPxixGA8b3nOrWpc
KNOWN_GDRIVE_FILES: Dict[str, str] = {
    "sample_1.pdf": "1Q4-8Hcjm0QlPxu8q6bk6LCZ-Z2P34FPw",
    "sample_2.pdf": "1zjmIgc0SRmnQOKaoRNjNPNbxkZ5pHV0N",
    "sample_3.pdf": "1ICg9GLr1jrXeQoCjiljrOwl4JtZJ75LA",
    "sample_4.pdf": "18JQu-kcCzRA-WiK1w3BqCpawOe-2tWnS",
    "sample_5.pdf": "1Z98K7krGb2EoEweE_SDF5EXuPO4oLUTV",
    "sample_6.pdf": "1w0W_AMdBm1_u4gsABjlxXJR7kAbT9DfF",
}


class RecursiveCharacterTextSplitter:
    """
    Splits text recursively across semantic boundaries (paragraphs, sentences, clauses)
    targeting a maximum character size with overlap.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " "]

    def split_text(self, text: str) -> List[str]:
        """Splits a single body of text into overlapping chunks."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Internal recursive splitting logic along hierarchy of delimiters."""
        if not separators:
            # Base case: split hard at chunk_size
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        sep = separators[0]
        splits = text.split(sep)
        good_splits: List[str] = []
        current_chunk = ""

        for part in splits:
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    good_splits.append(current_chunk.strip())
                if len(part) > self.chunk_size:
                    # Recursively split large parts with deeper separators
                    sub_splits = self._recursive_split(part, separators[1:])
                    good_splits.extend(sub_splits)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            good_splits.append(current_chunk.strip())

        # Now apply overlap if chunks are discrete
        return self._merge_with_overlap(good_splits)

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Ensure overlapping character context across sequential chunk boundaries."""
        if len(pieces) <= 1:
            return pieces

        merged: List[str] = []
        for i, piece in enumerate(pieces):
            if i == 0:
                merged.append(piece)
            else:
                prev = merged[-1]
                overlap_len = min(self.chunk_overlap, len(prev))
                overlap_prefix = prev[-overlap_len:]
                # Avoid duplicating identical sentences
                combined = piece
                if not piece.startswith(overlap_prefix.strip()):
                    combined = f"...{overlap_prefix.strip()} {piece}"
                merged.append(combined[: self.chunk_size + self.chunk_overlap])
        return merged


class DocumentProcessor:
    """End-to-end document loader, normalizer, and metadata chunking engine."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Cleans noise artifacts, trailing page headers, and normalizes financial symbols."""
        if not raw_text:
            return ""

        # Remove ReportLab generated boilerplates
        text = re.sub(r"ReportLab Generated PDF.*?(?=\n|$)", "", raw_text, flags=re.I)
        # Normalize non-breaking spaces and special bullet points
        text = text.replace("\xa0", " ").replace("\177", "•").replace("■", "Rs ")
        # Standardize multiple newlines and carriage returns
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Standardize multiple spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Trim leading and trailing whitespace
        return text.strip()

    @staticmethod
    def sync_google_drive_folder(
        target_dir: Path,
        folder_id: Optional[str] = None,
        force: bool = False,
    ) -> List[Path]:
        """
        Pulls documents from the Google Drive folder if not already cached locally.
        Uses public direct export endpoints for the documented files.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        downloaded_paths: List[Path] = []

        for filename, file_id in KNOWN_GDRIVE_FILES.items():
            dest = target_dir / filename
            if dest.exists() and not force and dest.stat().st_size > 1000:
                logger.info("Found cached GDrive file: %s", dest)
                downloaded_paths.append(dest)
                continue

            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            try:
                logger.info("Downloading %s from Google Drive...", filename)
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f_out:
                    content = resp.read()
                    f_out.write(content)
                logger.info("Successfully saved %s (%d bytes)", filename, len(content))
                downloaded_paths.append(dest)
            except Exception as e:
                logger.error("Failed to download %s from GDrive: %s", filename, e)

        return downloaded_paths

    def parse_pdf(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """
        Extracts structured text per page from a PDF file.
        Returns: List of tuples: (page_no, section_name, cleaned_text)
        """
        results: List[Tuple[int, str, str]] = []
        try:
            reader = pypdf.PdfReader(str(file_path))
            current_section = "General Policy Information"

            for page_idx, page in enumerate(reader.pages):
                page_no = page_idx + 1
                page_text = page.extract_text() or ""
                cleaned = self.clean_text(page_text)
                if not cleaned:
                    continue

                # Detect Section header markers in page text
                section_match = re.search(r"(Section\s+\d+(\.\d+)?[^:\n]*:?[^\n]*)", cleaned, re.IGNORECASE)
                if section_match:
                    current_section = section_match.group(1).strip()

                results.append((page_no, current_section, cleaned))
        except Exception as e:
            logger.error("Error reading PDF %s: %s", file_path.name, e)

        return results

    def parse_csv(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """Parses structured CSV policy documents or QA files."""
        results: List[Tuple[int, str, str]] = []
        try:
            with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    row_lower = {k.lower(): v for k, v in row.items()}
                    content = (
                        row_lower.get("content")
                        or row_lower.get("context")
                        or row_lower.get("answer")
                        or " ".join(row.values())
                    )
                    section = (
                        row_lower.get("section")
                        or row_lower.get("category")
                        or f"Record {idx + 1}"
                    )
                    page_no = int(row_lower.get("page", 1)) if str(row_lower.get("page", "")).isdigit() else 1
                    cleaned = self.clean_text(content)
                    if cleaned:
                        results.append((page_no, section, cleaned))
        except Exception as e:
            logger.error("Error reading CSV %s: %s", file_path.name, e)

        return results

    def parse_json_dataset(self, file_path: Path) -> List[DocumentChunk]:
        """Parses pre-formatted sample_dataset.json into DocumentChunk instances."""
        chunks: List[DocumentChunk] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = data.get("documents", [])
            for doc in documents:
                doc_id = doc.get("doc_id", "DOC-GENERIC")
                doc_title = doc.get("doc_title", "Financial Knowledge Base")
                category = doc.get("category", "General")

                for sec in doc.get("sections", []):
                    section_heading = sec.get("heading", "General Section")
                    page_no = sec.get("page_no", 1)
                    content = self.clean_text(sec.get("content", ""))

                    # Add the primary section content chunk
                    if content:
                        for idx, text_slice in enumerate(self.splitter.split_text(content)):
                            chunk_id = hashlib.sha256(
                                f"{doc_id}_{section_heading}_{page_no}_{idx}_{text_slice[:50]}".encode()
                            ).hexdigest()[:16]

                            chunks.append(
                                DocumentChunk(
                                    chunk_id=chunk_id,
                                    content=text_slice,
                                    doc_id=doc_id,
                                    doc_title=doc_title,
                                    section=section_heading,
                                    page_no=page_no,
                                    metadata={"category": category, "chunk_index": idx},
                                )
                            )

                    # Also ingest the pre-formatted QA pairs as highly indexed context
                    for qa_idx, qa in enumerate(sec.get("qa_pairs", [])):
                        q = qa.get("question", "")
                        a = qa.get("answer", "")
                        combined_qa = f"Frequently Asked Question: {q}\nOfficial Policy Answer: {a}"
                        qa_chunk_id = hashlib.sha256(
                            f"{doc_id}_{section_heading}_qa_{qa_idx}_{q}".encode()
                        ).hexdigest()[:16]

                        chunks.append(
                            DocumentChunk(
                                chunk_id=qa_chunk_id,
                                content=combined_qa,
                                doc_id=doc_id,
                                doc_title=doc_title,
                                section=section_heading,
                                page_no=page_no,
                                metadata={
                                    "category": category,
                                    "is_qa_pair": True,
                                    "question": q,
                                },
                            )
                        )
        except Exception as e:
            logger.error("Error reading JSON dataset %s: %s", file_path.name, e)

        return chunks

    def process_file(self, file_path: Path) -> List[DocumentChunk]:
        """Processes an arbitrary file (PDF, CSV, TXT, MD) and splits into enriched DocumentChunks."""
        ext = file_path.suffix.lower()
        chunks: List[DocumentChunk] = []

        if ext == ".json":
            return self.parse_json_dataset(file_path)

        doc_title = file_path.stem.replace("_", " ").title()
        doc_id = f"DOC-{file_path.stem.upper()}"

        pages_data: List[Tuple[int, str, str]] = []
        if ext == ".pdf":
            pages_data = self.parse_pdf(file_path)
            # Refine doc_title from first page if present
            if pages_data:
                first_lines = pages_data[0][2].split("\n")
                if first_lines and len(first_lines[0].strip()) > 5:
                    doc_title = first_lines[0].strip()
        elif ext == ".csv":
            pages_data = self.parse_csv(file_path)
        elif ext in [".txt", ".md"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                cleaned = self.clean_text(raw)
                pages_data = [(1, "Document Section", cleaned)]
            except Exception as e:
                logger.error("Error reading text file %s: %s", file_path.name, e)

        for page_no, section_name, text in pages_data:
            text_splits = self.splitter.split_text(text)
            for idx, text_slice in enumerate(text_splits):
                chunk_id = hashlib.sha256(
                    f"{doc_id}_{page_no}_{section_name}_{idx}_{text_slice[:50]}".encode()
                ).hexdigest()[:16]

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        content=text_slice,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        section=section_name,
                        page_no=page_no,
                        metadata={
                            "source_file": file_path.name,
                            "chunk_index": idx,
                            "char_count": len(text_slice),
                        },
                    )
                )

        return chunks

    def process_all(
        self,
        raw_dir: Path,
        sample_json_path: Optional[Path] = None,
    ) -> List[DocumentChunk]:
        """Ingests all files from raw_dir plus the sample JSON dataset."""
        all_chunks: List[DocumentChunk] = []

        # 1. Process sample JSON dataset if provided
        if sample_json_path and sample_json_path.exists():
            logger.info("Processing structured JSON dataset: %s", sample_json_path)
            json_chunks = self.parse_json_dataset(sample_json_path)
            all_chunks.extend(json_chunks)

        # 2. Process all documents in raw_dir
        if raw_dir.exists():
            for p in sorted(raw_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in [".pdf", ".csv", ".txt", ".md"]:
                    logger.info("Processing raw document: %s", p.name)
                    doc_chunks = self.process_file(p)
                    all_chunks.extend(doc_chunks)

        logger.info("Document processing complete: %d total chunks extracted.", len(all_chunks))
        return all_chunks
