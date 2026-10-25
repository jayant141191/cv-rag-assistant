"""Central configuration, loaded from environment variables / a `.env` file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # --- OpenAI ---------------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: _env("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    judge_model: str = field(default_factory=lambda: _env("OPENAI_JUDGE_MODEL", "gpt-4o"))
    # text-embedding-3-large is OpenAI's strongest general-purpose embedding model.
    embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    )
    # Optional dimension truncation (text-embedding-3-* support it). Empty = native size.
    embedding_dimensions: int | None = field(
        default_factory=lambda: int(v) if (v := os.getenv("OPENAI_EMBEDDING_DIMENSIONS")) else None
    )

    # --- Vector store ---------------------------------------------------
    chroma_dir: Path = field(
        default_factory=lambda: Path(_env("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db")))
    )
    collection_name: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "cvs"))

    # --- Chunking -------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "120")))

    # --- Retrieval / graph ----------------------------------------------
    top_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_K", "6")))
    fetch_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_FETCH_K", "20")))
    max_query_rewrites: int = field(default_factory=lambda: int(_env("MAX_QUERY_REWRITES", "1")))
    max_generation_retries: int = field(
        default_factory=lambda: int(_env("MAX_GENERATION_RETRIES", "1"))
    )

    def require_api_key(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )


def get_settings() -> Settings:
    return Settings()
