"""Central configuration loaded from environment / .env."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
PARENTS_PATH = DATA_DIR / "parents.json"
COLLECTION_NAME = "documind"


def load_env(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE pairs into os.environ (does not override existing)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


load_env()
for _d in (DATA_DIR, CHROMA_DIR, UPLOAD_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # Providers
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")

    # Models (Groq primary, Gemini fallback)
    groq_model: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    gemini_model: str = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

    # Local models
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
    )
    reranker_model: str = os.environ.get(
        "RERANKER_MODEL", "BAAI/bge-reranker-base"
    )

    # Chunking
    chunk_size: int = _get_int("CHUNK_SIZE", 900)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 150)

    # Parent/child (small-to-big) retrieval
    use_parent_child: bool = _get_bool("USE_PARENT_CHILD", True)
    child_size: int = _get_int("CHILD_SIZE", 350)
    child_overlap: int = _get_int("CHILD_OVERLAP", 80)
    contextual_chunks: bool = _get_bool("CONTEXTUAL_CHUNKS", False)

    # Retrieval
    dense_top_k: int = _get_int("DENSE_TOP_K", 20)
    sparse_top_k: int = _get_int("SPARSE_TOP_K", 20)
    rerank_top_n: int = _get_int("RERANK_TOP_N", 5)
    min_rerank_score: float = _get_float("MIN_RERANK_SCORE", -4.0)
    rrf_k: int = _get_int("RRF_K", 60)

    # Query understanding (rewrite + multi-query + HyDE)
    enable_query_expansion: bool = _get_bool("ENABLE_QUERY_EXPANSION", True)
    enable_hyde: bool = _get_bool("ENABLE_HYDE", True)
    num_query_variants: int = _get_int("NUM_QUERY_VARIANTS", 3)
    history_turns: int = _get_int("HISTORY_TURNS", 4)

    # Answer quality
    enable_answer_verification: bool = _get_bool("ENABLE_ANSWER_VERIFICATION", True)

    # Caching
    enable_cache: bool = _get_bool("ENABLE_CACHE", True)

    # Generation
    temperature: float = _get_float("TEMPERATURE", 0.2)
    max_output_tokens: int = _get_int("MAX_OUTPUT_TOKENS", 2048)


settings = Settings()
