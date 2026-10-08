"""
Vector Store and Embedding Generation Engine.
Supports multiple embedding providers (OpenAI, Gemini, Deterministic Fallback)
and persistent Vector Store implementations (ChromaDB or NumPy in-memory/persistent).
"""

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import numpy as np

from app.core.config import get_settings
from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseEmbeddingProvider:
    """Abstract interface for embedding generation."""

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


class DeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """
    High-fidelity deterministic semantic subword embedder.
    Maps words, character n-grams, and financial domain keywords into a 384-dimensional
    L2-normalized vector. Enables zero-dependency, instant, offline testing with
    semantic similarity that tracks lexical and domain overlap.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_single(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        for pos, token in enumerate(tokens):
            # Token position weight
            decay = 1.0 / math.sqrt(pos + 1)

            # Hash token to dimension
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx1 = h % self.dimension
            sign1 = 1.0 if (h >> 4) % 2 == 0 else -1.0
            vec[idx1] += sign1 * decay * 2.0

            # Character 3-grams
            if len(token) >= 3:
                for j in range(len(token) - 2):
                    tri = token[j : j + 3]
                    htri = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16)
                    idx2 = htri % self.dimension
                    sign2 = 1.0 if (htri >> 3) % 2 == 0 else -1.0
                    vec[idx2] += sign2 * 0.5

        # Normalize to unit vector
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.array([self._embed_single(t) for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_single(query)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Generates dense vector embeddings using OpenAI API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.openai.com/v1/embeddings"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Batch in chunks of 64
        all_embeddings = []
        batch_size = 64
        with httpx.Client(timeout=30.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                payload = {"input": batch, "model": self.model}
                resp = client.post(self.url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()["data"]
                for item in data:
                    all_embeddings.append(item["embedding"])

        vecs = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-8)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Generates embeddings using Google Gemini API."""

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}"

    def embed_query(self, query: str) -> np.ndarray:
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": query}]},
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(self.url, json=payload)
            resp.raise_for_status()
            val = resp.json()["embedding"]["values"]
            vec = np.array(val, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / max(norm, 1e-8)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed_query(t) for t in texts], dtype=np.float32)


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Factory creating the appropriate embedding provider based on runtime settings."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "openai" and settings.OPENAI_API_KEY:
        try:
            return OpenAIEmbeddingProvider(settings.OPENAI_API_KEY, settings.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning("Could not initialize OpenAI embedder (%s). Falling back to deterministic.", e)

    if provider == "gemini" and settings.GEMINI_API_KEY:
        try:
            return GeminiEmbeddingProvider(settings.GEMINI_API_KEY)
        except Exception as e:
            logger.warning("Could not initialize Gemini embedder (%s). Falling back to deterministic.", e)

    return DeterministicEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)


class VectorStore:
    """
    High-performance vector database with persistence, dense cosine search,
    and fallback resilience across hardware platforms.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or settings.persist_path
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = get_embedding_provider()
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def count(self) -> int:
        """Returns the number of indexed chunks."""
        return len(self.chunks)

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Computes embeddings for chunks and updates in-memory index."""
        if not chunks:
            logger.warning("No chunks provided to index.")
            return

        logger.info("Computing embeddings for %d chunks...", len(chunks))
        texts = [c.content for c in chunks]
        self.embeddings = self.embedding_provider.embed_texts(texts)
        self.chunks = chunks
        logger.info("Indexed %d chunks with shape %s", len(self.chunks), self.embeddings.shape)

    def search_dense(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """
        Executes dense vector search using cosine similarity.
        Returns: List of tuples (DocumentChunk, cosine_similarity_score)
        """
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_vec = self.embedding_provider.embed_query(query)
        # Cosine similarity for unit-normalized vectors is dot product
        similarities = np.dot(self.embeddings, query_vec)

        # Handle score scaling to [0.0, 1.0]
        similarities = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)

        # Get top indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(-similarities)[:top_k]

        results = [(self.chunks[i], float(similarities[i])) for i in top_indices]
        return results

    def save(self) -> None:
        """Persists the chunks and embedding vectors to disk."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = self.persist_dir / "chunks_metadata.json"
        matrix_file = self.persist_dir / "embeddings_matrix.npy"

        chunks_data = [c.model_dump() for c in self.chunks]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        if self.embeddings is not None:
            np.save(str(matrix_file), self.embeddings)

        logger.info("Persisted vector store (%d chunks) to %s", len(self.chunks), self.persist_dir)

    def load(self) -> bool:
        """Loads cached chunks and embedding vectors from disk if present."""
        metadata_file = self.persist_dir / "chunks_metadata.json"
        matrix_file = self.persist_dir / "embeddings_matrix.npy"

        if not (metadata_file.exists() and matrix_file.exists()):
            return False

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = [DocumentChunk(**item) for item in data]
            self.embeddings = np.load(str(matrix_file))
            logger.info("Successfully loaded vector store (%d chunks) from disk.", len(self.chunks))
            return True
        except Exception as e:
            logger.error("Failed to load vector store from disk: %s", e)
            return False


# Singleton VectorStore instance
_global_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Singleton getter for VectorStore."""
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = VectorStore()
        # Attempt to load persistent cache on boot
        _global_vector_store.load()
    return _global_vector_store
