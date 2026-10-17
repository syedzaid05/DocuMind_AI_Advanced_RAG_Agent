"""Local embedding model (sentence-transformers) with caching.

BGE-family models expect an instruction prefix on the query side only, so we
keep separate encode paths for documents and queries. Text->vector results are
memoized to avoid recomputing repeated strings.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .cache import LruDict, make_key
from .config import settings

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_EMB_CACHE = LruDict(maxsize=100_000)


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def _encode(texts: list[str]) -> np.ndarray:
    keys = [make_key(settings.embedding_model, t) for t in texts]
    results: list[np.ndarray | None] = [None] * len(texts)
    missing: list[int] = []
    for i, key in enumerate(keys):
        cached = _EMB_CACHE.get(key)
        if cached is None:
            missing.append(i)
        else:
            results[i] = cached

    if missing:
        model = get_embedder()
        vectors = model.encode(
            [texts[i] for i in missing],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for j, i in enumerate(missing):
            vec = np.asarray(vectors[j], dtype="float32")
            results[i] = vec
            _EMB_CACHE.set(keys[i], vec)

    return np.vstack(results).astype("float32")


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _encode(texts).tolist()


def embed_query(query: str) -> list[float]:
    text = query.strip()
    if "bge" in settings.embedding_model.lower():
        text = _QUERY_PREFIX + text
    return _encode([text])[0].tolist()


def clear_cache() -> None:
    _EMB_CACHE.clear()
