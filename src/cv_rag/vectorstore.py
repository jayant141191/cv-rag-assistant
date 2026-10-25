"""Local, persistent Chroma vector store with OpenAI embeddings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .chunking import chunk_cv
from .config import Settings, get_settings
from .loaders import load_bytes


def get_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    settings = settings or get_settings()
    settings.require_api_key()
    kwargs = {"model": settings.embedding_model, "api_key": settings.openai_api_key}
    if settings.embedding_dimensions:
        kwargs["dimensions"] = settings.embedding_dimensions
    return OpenAIEmbeddings(**kwargs)


class CVVectorStore:
    """Thin wrapper over Chroma with CV-specific helpers."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        Path(self.settings.chroma_dir).mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=get_embeddings(self.settings),
            persist_directory=str(self.settings.chroma_dir),
            # Cosine distance suits OpenAI's normalised embeddings.
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------- ingest
    def add_cv(self, data: bytes, filename: str, candidate: str | None = None) -> dict:
        """Parse, chunk and index one CV. Re-uploading a file replaces its old chunks."""
        text = load_bytes(data, filename)
        chunks = chunk_cv(
            text,
            source=filename,
            candidate=candidate,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.delete_source(Path(filename).name)
        docs = [Document(page_content=c.text, metadata=c.metadata) for c in chunks]
        self.store.add_documents(docs, ids=[c.id for c in chunks])
        return {
            "source": Path(filename).name,
            "candidate": chunks[0].metadata["candidate"] if chunks else candidate,
            "chunks": len(chunks),
        }

    def add_path(self, path: str | Path, candidate: str | None = None) -> dict:
        path = Path(path)
        return self.add_cv(path.read_bytes(), path.name, candidate)

    # ----------------------------------------------------------- retrieve
    def search(self, query: str, candidates: list[str] | None = None, k: int | None = None,
               use_mmr: bool = True) -> list[Document]:
        k = k or self.settings.top_k
        where = None
        if candidates:
            where = (
                {"candidate": candidates[0]}
                if len(candidates) == 1
                else {"candidate": {"$in": list(candidates)}}
            )
        if use_mmr:
            # MMR keeps results diverse, so one long CV doesn't crowd out the others.
            return self.store.max_marginal_relevance_search(
                query, k=k, fetch_k=max(self.settings.fetch_k, k), filter=where
            )
        return self.store.similarity_search(query, k=k, filter=where)

    # -------------------------------------------------------------- admin
    def _all_metadatas(self) -> list[dict]:
        return self.store.get(include=["metadatas"]).get("metadatas") or []

    def list_candidates(self) -> list[str]:
        return sorted({m.get("candidate") for m in self._all_metadatas() if m.get("candidate")})

    def list_sources(self) -> list[dict]:
        summary: dict[str, dict] = {}
        for m in self._all_metadatas():
            src = m.get("source", "unknown")
            entry = summary.setdefault(src, {"source": src, "candidate": m.get("candidate"), "chunks": 0})
            entry["chunks"] += 1
        return sorted(summary.values(), key=lambda e: e["source"])

    def delete_source(self, source: str) -> None:
        existing = self.store.get(where={"source": source})
        if existing.get("ids"):
            self.store.delete(ids=existing["ids"])

    def reset(self) -> None:
        self.store.reset_collection()


@lru_cache(maxsize=1)
def get_vectorstore() -> CVVectorStore:
    return CVVectorStore()
