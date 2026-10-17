"""LLM layer: Groq (primary) + Gemini (fallback). REST only, no SDKs.

Features: single-shot generation, token streaming, query planning (rewrite +
multi-query + HyDE) and answer groundedness verification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterator

import requests

from .cache import JsonDiskCache, make_key
from .config import CACHE_DIR, settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

ANSWER_SYSTEM = (
    "You are DocuMind, a meticulous research assistant. Follow these rules strictly:\n"
    "1. Answer ONLY from the provided context blocks. Never use outside knowledge.\n"
    "2. Every factual claim must carry an inline citation to the block it came from, "
    "formatted as [1], [2], etc. Multiple sources: [1][3].\n"
    "3. If the context does not fully answer the question, say exactly what is missing "
    "and answer only the supported part. Never guess.\n"
    "4. Do not repeat the question. Do not add a source list at the end (it is shown "
    "separately). Never invent citations or facts.\n"
    "5. Always structure the answer in Markdown using these sections (omit a "
    "section ONLY if the context has nothing for it):\n"
    "   **Direct answer:** one or two sentences.\n"
    "   **Key details:** 4-6 bullet points with figures, names and dates.\n"
    "   **Additional context:** one or two short paragraphs of background from the "
    "context.\n"
    "   **Summary:** one concluding sentence.\n"
    "   Aim for at least 10 lines in total. Every line must come from the context - "
    "do NOT pad with filler, repetition, or generic commentary."
)

PLANNER_SYSTEM = (
    "You are a search-query planner for a retrieval system over a document corpus. "
    "You never answer the question. You only produce search aids. Respond with valid "
    "JSON and nothing else."
)

VERIFIER_SYSTEM = (
    "You are a strict groundedness auditor. Given context and an answer, judge whether "
    "every factual claim is supported by the context. The answer cites sources as "
    "[n]. Respond ONLY with valid JSON."
)


@dataclass
class LLMResult:
    text: str
    provider: str
    cached: bool = False


_CACHE = JsonDiskCache(CACHE_DIR / "llm.json", enabled=settings.enable_cache)


def _gen_key(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    return make_key(
        "gen",
        settings.groq_model,
        settings.gemini_model,
        system,
        prompt,
        round(temperature, 4),
        max_tokens,
    )


# --------------------------------------------------------------------- helpers
def _context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        blocks.append(
            f"[{i}] (source: {meta.get('source')}, page {meta.get('page')})\n"
            f"{c['text']}"
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: list[dict]) -> str:
    return f"Context:\n{_context(chunks)}\n\nQuestion: {query}\n\nAnswer with citations:"


def _format_history(history: list[dict] | None, turns: int) -> str:
    if not history:
        return "(none)"
    lines = []
    for msg in history[-turns * 2 :]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip().replace("\n", " ")
        lines.append(f"{role}: {content[:400]}")
    return "\n".join(lines) if lines else "(none)"


def _parse_json(text: str) -> dict:
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _groq_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.7.1",
    }


# ------------------------------------------------------------------ generation
def _groq_generate(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    resp = requests.post(
        GROQ_URL,
        headers=_groq_headers(),
        json={
            "model": settings.groq_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _gemini_generate(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    url = GEMINI_URL.format(model=settings.gemini_model)
    resp = requests.post(
        url,
        params={"key": settings.gemini_api_key},
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    candidates = resp.json().get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def generate(
    prompt: str,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    """Try Groq first, fall back to Gemini on any failure."""
    _CACHE.enabled = settings.enable_cache
    system = system or ANSWER_SYSTEM
    temperature = settings.temperature if temperature is None else temperature
    max_tokens = max_tokens or settings.max_output_tokens

    key = _gen_key(prompt, system, temperature, max_tokens)
    cached = _CACHE.get(key)
    if cached:
        return LLMResult(cached["text"], cached.get("provider", "?"), cached=True)

    errors: list[str] = []
    if settings.groq_api_key:
        try:
            result = LLMResult(_groq_generate(prompt, system, temperature, max_tokens), "groq")
            _CACHE.set(key, {"text": result.text, "provider": result.provider})
            return result
        except Exception as exc:  # noqa: BLE001
            errors.append(f"groq: {exc}")
    if settings.gemini_api_key:
        try:
            result = LLMResult(
                _gemini_generate(prompt, system, temperature, max_tokens), "gemini"
            )
            _CACHE.set(key, {"text": result.text, "provider": result.provider})
            return result
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gemini: {exc}")
    raise RuntimeError("All LLM providers failed -> " + "; ".join(errors))


def clear_cache() -> None:
    _CACHE.clear()


# ------------------------------------------------------------------- streaming
def _sse_lines(resp: requests.Response) -> Iterator[str]:
    """Yield SSE data payloads, forcing UTF-8 (requests often guesses latin-1)."""
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data:"):
            continue
        yield line[5:].strip()


def _iter_groq_sse(resp: requests.Response) -> Iterator[str]:
    for data in _sse_lines(resp):
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            yield delta


def _iter_gemini_sse(resp: requests.Response) -> Iterator[str]:
    for data in _sse_lines(resp):
        if not data:
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        for cand in obj.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if part.get("text"):
                    yield part["text"]


def _caching_stream(provider: str, gen: Iterator[str], key: str) -> Iterator[str]:
    buf: list[str] = []
    for delta in gen:
        buf.append(delta)
        yield delta
    _CACHE.set(key, {"text": "".join(buf), "provider": provider})


def stream_answer(
    prompt: str,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, Iterator[str]]:
    """Return (provider, token_iterator). Falls back before the first token."""
    _CACHE.enabled = settings.enable_cache
    system = system or ANSWER_SYSTEM
    temperature = settings.temperature if temperature is None else temperature
    max_tokens = max_tokens or settings.max_output_tokens
    errors: list[str] = []

    key = _gen_key(prompt, system, temperature, max_tokens)
    cached = _CACHE.get(key)
    if cached:
        return cached.get("provider", "?"), iter([cached["text"]])

    if settings.groq_api_key:
        try:
            resp = requests.post(
                GROQ_URL,
                headers=_groq_headers(),
                json={
                    "model": settings.groq_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                timeout=90,
                stream=True,
            )
            if resp.status_code == 200:
                return "groq", _caching_stream("groq", _iter_groq_sse(resp), key)
            errors.append(f"groq: HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"groq: {exc}")

    if settings.gemini_api_key:
        url = GEMINI_STREAM_URL.format(model=settings.gemini_model)
        try:
            resp = requests.post(
                url,
                params={"key": settings.gemini_api_key, "alt": "sse"},
                headers={"Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
                timeout=120,
                stream=True,
            )
            if resp.status_code == 200:
                return "gemini", _caching_stream("gemini", _iter_gemini_sse(resp), key)
            errors.append(f"gemini: HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gemini: {exc}")

    raise RuntimeError("All LLM providers failed -> " + "; ".join(errors))


# -------------------------------------------------------------- query planning
def plan_queries(
    question: str,
    history: list[dict] | None = None,
    n: int | None = None,
    hyde: bool = True,
) -> dict:
    """Produce {standalone, paraphrases[], hyde} using one cheap LLM call."""
    n = n if n is not None else settings.num_query_variants
    history_text = _format_history(history, settings.history_turns)
    schema = (
        '{"standalone": "self-contained rewrite of the latest question", '
        f'"paraphrases": ["{n} alternative search queries"], '
        '"hyde": "a short hypothetical passage that would answer the question"}'
    )
    user = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user question: {question}\n\n"
        "Rewrite the latest question into a self-contained 'standalone' query "
        "(resolve pronouns and references using the conversation). Then produce "
        f"{n} diverse paraphrases that improve keyword and semantic recall, and a "
        "concise 'hyde' hypothetical answer passage. Return JSON exactly matching:\n"
        f"{schema}"
    )
    try:
        raw = generate(user, system=PLANNER_SYSTEM, temperature=0.1, max_tokens=500).text
        data = _parse_json(raw)
    except Exception:  # noqa: BLE001
        data = {}

    standalone = (data.get("standalone") or "").strip() or question
    paraphrases = [
        str(p).strip()
        for p in (data.get("paraphrases") or [])
        if isinstance(p, (str, int)) and str(p).strip()
    ][:n]
    hyde_text = (data.get("hyde") or "").strip() if hyde else ""
    return {"standalone": standalone, "paraphrases": paraphrases, "hyde": hyde_text}


# ---------------------------------------------------------------- verification
def verify_answer(query: str, answer: str, chunks: list[dict]) -> dict:
    """Audit groundedness of an answer against the retrieved context."""
    user = (
        f"Context:\n{_context(chunks)}\n\n"
        f"Question: {query}\n\n"
        f"Answer to audit:\n{answer}\n\n"
        "Return JSON only: "
        '{"grounded": true|false, "score": 0.0-1.0, '
        '"unsupported_claims": ["..."], "reason": "short explanation"}'
    )
    try:
        raw = generate(user, system=VERIFIER_SYSTEM, temperature=0.0, max_tokens=400).text
        data = _parse_json(raw)
    except Exception:  # noqa: BLE001
        return {}
    if not data:
        return {}
    try:
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "grounded": bool(data.get("grounded", False)),
        "score": score,
        "unsupported_claims": [str(c) for c in (data.get("unsupported_claims") or [])][:5],
        "reason": str(data.get("reason", "")).strip(),
    }
