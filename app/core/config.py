"""
Configuration module for FinTech RAG Assistant.
Loads environment variables, handles type validation, and provides application settings.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings schema backed by pydantic-settings and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Base Application Metadata
    PROJECT_NAME: str = Field(default="FinTech RAG Assistant", description="Application service name")
    ENVIRONMENT: str = Field(default="development", description="Runtime environment: development|production|testing")
    API_V1_STR: str = Field(default="/api/v1", description="API route prefix for v1 endpoints")
    HOST: str = Field(default="0.0.0.0", description="Host interface binding")
    PORT: int = Field(default=8000, description="Listening TCP port")

    # Security & Rate Limiting
    API_KEY_SECRET: str = Field(default="fintech_secret_key_demo_2026", description="Static secret for API key authentication")
    ENABLE_AUTH: bool = Field(default=False, description="Whether to require X-API-Key on protected endpoints")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Maximum requests per client per minute")

    # LLM Provider Configuration
    LLM_PROVIDER: str = Field(default="mock", description="LLM provider: openai | gemini | mock")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API authentication key")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI model identifier")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API authentication key")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", description="Google Gemini model identifier")
    TEMPERATURE: float = Field(default=0.0, description="Sampling temperature for strict deterministic answers")
    MAX_OUTPUT_TOKENS: int = Field(default=1024, description="Maximum tokens generated in response")

    # Embedding Provider Configuration
    EMBEDDING_PROVIDER: str = Field(default="deterministic", description="Embedding engine: openai | gemini | deterministic")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Model name for embeddings")
    EMBEDDING_DIMENSION: int = Field(default=384, description="Vector dimension size")

    # Vector Store & Storage Paths
    VECTOR_STORE_TYPE: str = Field(default="memory", description="Vector database engine: memory | chroma")
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    PERSIST_DIRECTORY: Path = Field(default=Path("data/processed"))
    RAW_DATA_DIRECTORY: Path = Field(default=Path("data/raw"))
    SAMPLE_DATASET_PATH: Path = Field(default=Path("data/sample_dataset.json"))
    EVAL_DATASET_PATH: Path = Field(default=Path("evaluation/eval_dataset.json"))

    # RAG Pipeline Parameters
    CHUNK_SIZE: int = Field(default=500, description="Target chunk character length")
    CHUNK_OVERLAP: int = Field(default=100, description="Character overlap between consecutive chunks")
    TOP_K_DENSE: int = Field(default=10, description="Number of candidate chunks from dense semantic search")
    TOP_K_SPARSE: int = Field(default=10, description="Number of candidate chunks from BM25 sparse search")
    RRF_K: int = Field(default=60, description="Reciprocal Rank Fusion constant denominator smoothing parameter")
    TOP_K_RERANK: int = Field(default=3, description="Final number of top chunks delivered to LLM prompt")
    MIN_CONFIDENCE_THRESHOLD: float = Field(default=0.45, description="Minimum confidence score below which fallback is triggered")

    # Google Drive Data Sync
    GDRIVE_FOLDER_ID: str = Field(default="1CX9szQJMiVBEYPqMnPxixGA8b3nOrWpc", description="Target Google Drive folder ID")

    # Frontend Client
    BACKEND_API_URL: str = Field(default="http://localhost:8000", description="Backend URL for Streamlit client")

    @property
    def raw_data_path(self) -> Path:
        """Returns the absolute path to raw data directory."""
        if self.RAW_DATA_DIRECTORY.is_absolute():
            return self.RAW_DATA_DIRECTORY
        return self.BASE_DIR / self.RAW_DATA_DIRECTORY

    @property
    def persist_path(self) -> Path:
        """Returns the absolute path to processed data directory."""
        if self.PERSIST_DIRECTORY.is_absolute():
            return self.PERSIST_DIRECTORY
        return self.BASE_DIR / self.PERSIST_DIRECTORY

    @property
    def sample_dataset_abs_path(self) -> Path:
        """Returns the absolute path to sample dataset JSON."""
        if self.SAMPLE_DATASET_PATH.is_absolute():
            return self.SAMPLE_DATASET_PATH
        return self.BASE_DIR / self.SAMPLE_DATASET_PATH


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
