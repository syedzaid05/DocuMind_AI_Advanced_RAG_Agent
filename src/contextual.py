"""Optional contextual retrieval: prepend a short situating context per chunk.

Inspired by Anthropic's "contextual retrieval": each chunk gets 1-2 sentences
of document-level context before embedding, improving match quality. This costs
one LLM call per batch of chunks and is opt-in (settings.contextual_chunks).
"""

from __future__ import annotations

import json
import re

from . import llm

CTX_SYSTEM = (
    "You situate document chunks within their source document. You never answer "
    "questions. Respond ONLY with valid JSON."
)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _contextualize_batch(doc_text: str, chunks: list[dict], idxs: list[int]) -> dict[int, str]:
    listing = "\n".join(f"{i}. {chunks[i]['text']}" for i in idxs)
    user = (
        f"<document>\n{doc_text[:6000]}\n</document>\n\n"
        "For each numbered chunk below, write 1-2 short sentences that situate it "
        "within the document (what section/topic it belongs to and how it fits). "
        "Be specific and do not answer any question.\n\n"
        f"Chunks:\n{listing}\n\n"
        'Return JSON: {"contexts": {"<chunk number>": "<context>", ...}}'
    )
    raw = llm.generate(user, system=CTX_SYSTEM, temperature=0.0, max_tokens=900).text
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    contexts = data.get("contexts", {})
    out: dict[int, str] = {}
    for key, value in contexts.items():
        try:
            out[int(key)] = str(value).strip()
        except (TypeError, ValueError):
            continue
    return out


def contextualize_document(
    doc_text: str, chunks: list[dict], batch_size: int = 8
) -> list[dict]:
    """Return chunks with a contextual prefix prepended to their text."""
    enriched = [dict(c) for c in chunks]
    for start in range(0, len(chunks), batch_size):
        idxs = list(range(start, min(start + batch_size, len(chunks))))
        try:
            contexts = _contextualize_batch(doc_text, chunks, idxs)
        except Exception:  # noqa: BLE001
            contexts = {}
        for i in idxs:
            ctx = contexts.get(i)
            if ctx:
                enriched[i]["text"] = f"{ctx}\n{enriched[i]['text']}"
    return enriched
