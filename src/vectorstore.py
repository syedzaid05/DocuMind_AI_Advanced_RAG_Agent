"""Hybrid store: ChromaDB dense vectors + BM25 sparse, fused with RRF."""

from __future__ import annotations

import re
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from . import embeddings
from .config import settings

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HybridStore:
    def __init__(self, persist_path: str, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._reset_cache()

    def _reset_cache(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []

    # ------------------------------------------------------------------ write
    def add(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        ids = [c["id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        vectors = embeddings.embed_documents(docs)
        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=vectors,
        )
        self._reset_cache()
        return len(chunks)

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})
        self._reset_cache()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        self._reset_cache()

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return self.collection.count()

    def sources(self) -> list[str]:
        if self.count() == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        return sorted({m["source"] for m in data["metadatas"] if "source" in m})

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        data = self.collection.get(include=["documents", "metadatas"])
        self._ids = data["ids"]
        self._docs = [d or "" for d in data["documents"]]
        self._metas = data["metadatas"]
        corpus = [tokenize(d) for d in self._docs]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def dense_search(self, query: str, k: int) -> list[dict]:
        if self.count() == 0:
            return []
        qvec = embeddings.embed_query(query)
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            out.append(
                {
                    "id": cid,
                    "text": doc,
                    "metadata": meta,
                    "dense_score": 1.0 - float(dist),
                }
            )
        return out

    def sparse_search(self, query: str, k: int) -> list[dict]:
        self._ensure_bm25()
        if not self._bm25 or not self._ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "id": self._ids[i],
                "text": self._docs[i],
                "metadata": self._metas[i],
                "sparse_score": float(scores[i]),
            }
            for i in ranked
            if scores[i] > 0
        ]

    def hybrid_search(
        self,
        query: str,
        dense_k: int | None = None,
        sparse_k: int | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of dense and sparse results."""
        dense_k = dense_k or settings.dense_top_k
        sparse_k = sparse_k or settings.sparse_top_k
        top_k = top_k or (dense_k + sparse_k)

        merged: dict[str, dict] = {}
        rrf_k = settings.rrf_k

        for rank, hit in enumerate(self.dense_search(query, dense_k)):
            entry = merged.setdefault(hit["id"], {**hit, "rrf": 0.0})
            entry.update({k: v for k, v in hit.items() if k != "id"})
            entry["rrf"] += 1.0 / (rrf_k + rank + 1)

        for rank, hit in enumerate(self.sparse_search(query, sparse_k)):
            entry = merged.setdefault(hit["id"], {**hit, "rrf": 0.0})
            entry.update({k: v for k, v in hit.items() if k != "id"})
            entry["rrf"] += 1.0 / (rrf_k + rank + 1)

        results = sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)
        return results[:top_k]
