"""Retrieval: multi-query hybrid search + RRF + rerank + small-to-big resolution."""

from __future__ import annotations

from functools import lru_cache

from .cache import LruDict, make_key
from .config import settings
from .vectorstore import HybridStore

_RERANK_CACHE = LruDict(maxsize=2048)


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def rerank(query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
    if not candidates:
        return []
    top_n = top_n or settings.rerank_top_n
    key = make_key(
        "rerank",
        settings.reranker_model,
        query,
        [(c["id"], c.get("text", "")) for c in candidates],
    )
    scores = _RERANK_CACHE.get(key)
    if scores is None:
        model = get_reranker()
        scores = [float(s) for s in model.predict([(query, c["text"]) for c in candidates])]
        _RERANK_CACHE.set(key, scores)

    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = score
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_n]


def _rrf_fuse(result_lists: list[list[dict]], k: int, top_k: int | None = None) -> list[dict]:
    """Reciprocal Rank Fusion across several ranked result lists."""
    agg: dict[str, dict] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            entry = agg.setdefault(hit["id"], {"hit": dict(hit), "score": 0.0})
            entry["hit"].update({kk: vv for kk, vv in hit.items() if kk != "id"})
            entry["score"] += 1.0 / (k + rank + 1)
    fused = [{**e["hit"], "rrf_fused": e["score"]} for e in agg.values()]
    fused.sort(key=lambda x: x["rrf_fused"], reverse=True)
    return fused[:top_k] if top_k else fused


def _resolve_parents(candidates: list[dict], docstore, top_n: int) -> list[dict]:
    """Collapse matched children to their best parent, ordered by child score."""
    best: dict[str, dict] = {}
    for child in candidates:
        pid = child["metadata"].get("parent_id")
        if not pid:
            continue
        if pid not in best or child["rerank_score"] > best[pid]["rerank_score"]:
            best[pid] = child

    ranked = sorted(best.values(), key=lambda c: c["rerank_score"], reverse=True)[:top_n]
    parents = {p["id"]: p for p in docstore.get([c["metadata"]["parent_id"] for c in ranked])}

    resolved: list[dict] = []
    for child in ranked:
        pid = child["metadata"]["parent_id"]
        parent = parents.get(pid)
        if not parent:
            continue
        resolved.append(
            {
                "id": pid,
                "text": parent["text"],
                "metadata": parent["metadata"],
                "rerank_score": child["rerank_score"],
                "matched_child": child["text"],
            }
        )
    return resolved


def retrieve(
    question: str,
    store: HybridStore,
    plan: dict | None = None,
    top_n: int | None = None,
    min_score: float | None = None,
    docstore=None,
) -> list[dict]:
    """Multi-query hybrid retrieve -> RRF fuse -> rerank -> (parent) resolve."""
    top_n = top_n or settings.rerank_top_n
    threshold = settings.min_rerank_score if min_score is None else min_score

    queries: list[str] = []
    if plan and plan.get("standalone"):
        queries.append(plan["standalone"])
    else:
        queries.append(question)
    if plan:
        queries.extend(plan.get("paraphrases") or [])
        if plan.get("hyde"):
            queries.append(plan["hyde"])

    seen: set[str] = set()
    unique_queries = [q for q in queries if q and not (q in seen or seen.add(q))]

    result_lists = [store.hybrid_search(q) for q in unique_queries]
    candidates = _rrf_fuse(result_lists, k=settings.rrf_k)
    rerank_query = (plan or {}).get("standalone") or question

    is_parent_mode = bool(
        docstore is not None
        and candidates
        and candidates[0].get("metadata", {}).get("parent_id")
    )
    if is_parent_mode:
        reranked = rerank(rerank_query, candidates, top_n=top_n * 3)
        resolved = _resolve_parents(reranked, docstore, top_n)
        return [c for c in resolved if c["rerank_score"] >= threshold]

    reranked = rerank(rerank_query, candidates, top_n=top_n)
    return [c for c in reranked if c.get("rerank_score", 0.0) >= threshold]


def clear_cache() -> None:
    _RERANK_CACHE.clear()
