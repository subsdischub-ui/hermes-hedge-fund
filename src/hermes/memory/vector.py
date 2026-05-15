"""Vector store for semantic retrieval of lessons + reflections."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from hermes.settings import get_settings


class VectorStore:
    def __init__(self, path: Path, collection: str, embedding_model: str):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Use Chroma's default ONNX MiniLM unless caller swaps it.
        self.collection = self.client.get_or_create_collection(name=collection)
        self.embedding_model = embedding_model

    def add(self, ids: list[str], docs: list[str], metas: list[dict] | None = None) -> None:
        self.collection.add(ids=ids, documents=docs, metadatas=metas or [{}] * len(docs))

    def query(self, text: str, k: int = 6) -> list[dict]:
        r = self.collection.query(query_texts=[text], n_results=k)
        ids = r["ids"][0] if r["ids"] else []
        docs = r["documents"][0] if r["documents"] else []
        metas = (r["metadatas"] or [[]])[0]
        dists = (r["distances"] or [[]])[0]
        return [
            {"id": ids[i], "text": docs[i], "meta": metas[i] if i < len(metas) else {}, "distance": dists[i] if i < len(dists) else None}
            for i in range(len(ids))
        ]


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    s = get_settings()
    return VectorStore(
        path=s.vector_dir,
        collection=s.yaml("memory", "vector_collection", default="hermes_lessons"),
        embedding_model=s.yaml(
            "memory", "embedding_model", default="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
