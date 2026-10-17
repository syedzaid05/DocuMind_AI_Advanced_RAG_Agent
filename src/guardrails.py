"""Input, retrieval and output guardrails for the RAG pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INJECTION_PATTERNS = [
    r"ignore (all|previous|prior) (instructions|prompts|rules)",
    r"disregard (the )?(above|previous|system)",
    r"you are now (a|an|in)",
    r"reveal (your )?(system prompt|instructions)",
    r"developer mode",
    r"jailbreak",
    r"pretend to be",
    r"act as (if|though)",
    r"repeat the (system|initial) (prompt|message)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_CITATION_RE = re.compile(r"[\[\(【]\s*(\d+)\s*[\]\)】]")
_FULLWIDTH_CITATION_RE = re.compile(r"【\s*(\d+)\s*】")
MAX_QUERY_CHARS = 2000


def normalize_citations(text: str) -> str:
    """Convert full-width citation markers (e.g. 【1】) to standard [1]."""
    return _FULLWIDTH_CITATION_RE.sub(r"[\1]", text)


def normalize_answer(text: str) -> str:
    """Normalize citation markers and exotic whitespace from LLM output."""
    text = normalize_citations(text)
    text = text.replace("\u202f", " ").replace("\u00a0", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


@dataclass
class GuardrailResult:
    ok: bool
    stage: str
    message: str = ""
    flags: list[str] = field(default_factory=list)


def check_input(query: str) -> GuardrailResult:
    q = (query or "").strip()
    if not q:
        return GuardrailResult(False, "input", "Please enter a question.")
    if len(q) > MAX_QUERY_CHARS:
        return GuardrailResult(
            False, "input", f"Question too long (max {MAX_QUERY_CHARS} characters)."
        )
    if _INJECTION_RE.search(q):
        return GuardrailResult(
            False,
            "input",
            "I can't follow instructions that try to override my role. Ask about your documents instead.",
            flags=["prompt_injection"],
        )
    return GuardrailResult(True, "input", flags=["clean"])


def check_retrieval(query: str, chunks: list[dict]) -> GuardrailResult:
    if not chunks:
        return GuardrailResult(
            False,
            "retrieval",
            "I couldn't find anything relevant in the indexed documents.",
            flags=["no_context"],
        )
    return GuardrailResult(True, "retrieval", flags=["context_found"])


def check_output(answer: str, chunks: list[dict]) -> GuardrailResult:
    """Heuristic groundedness/citation check on the generated answer."""
    flags: list[str] = []
    if not answer or not answer.strip():
        return GuardrailResult(
            False, "output", "The model returned an empty answer.", flags=["empty"]
        )

    cited = {int(n) for n in _CITATION_RE.findall(answer)}
    valid = {i for i in range(1, len(chunks) + 1)}
    if cited:
        if not cited.issubset(valid):
            flags.append("invalid_citation")
    else:
        flags.append("no_citations")

    lower = answer.lower()
    if any(
        phrase in lower
        for phrase in ("as an ai", "i cannot access", "i don't have access to the")
    ):
        flags.append("refusal")

    return GuardrailResult(True, "output", flags=flags)
