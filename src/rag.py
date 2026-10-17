"""End-to-end RAG pipeline.

Flow: guardrails -> query planning (rewrite/multi-query/HyDE) -> multi-query
hybrid retrieval -> RRF -> rerank -> guardrails -> Groq/Gemini (stream) ->
groundedness verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import guardrails, llm, retrieval
from .config import CHROMA_DIR, COLLECTION_NAME, PARENTS_PATH, settings
from .contextual import contextualize_document
from .docstore import JsonDocStore
from .ingestion import chunk_document, chunk_document_parent_child, extract_pages
from .vectorstore import HybridStore

_NO_CITE_NOTE = "\n\n_(No inline citations were produced; review sources below.)_"


@dataclass
class IngestReport:
    files: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)


@dataclass
class Prepared:
    ok: bool
    message: str
    chunks: list[dict]
    plan: dict
    effective_query: str
    guardrails: dict


@dataclass
class QueryResult:
    answer: str
    provider: str
    sources: list[dict]
    guardrails: dict
    verification: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self) -> None:
        self.store = HybridStore(str(CHROMA_DIR), COLLECTION_NAME)
        self.docstore = JsonDocStore(PARENTS_PATH)

    # ---------------------------------------------------------------- ingest
    def ingest_pdf(self, pdf_path: str | Path) -> int:
        path = Path(pdf_path)
        pages = extract_pages(path)
        if not pages:
            raise ValueError(f"No extractable text in {path.name}")
        self.store.delete_source(path.name)
        self.docstore.delete_source(path.name)

        if not settings.use_parent_child:
            chunks = chunk_document(pages, source=path.name)
            return self.store.add(chunks)

        parents, children = chunk_document_parent_child(pages, source=path.name)
        if settings.contextual_chunks:
            doc_text = "\n\n".join(p["text"] for p in pages)
            children = contextualize_document(doc_text, children)
        self.docstore.upsert(parents)
        return self.store.add(children)

    def ingest_many(self, pdf_paths: list[str | Path]) -> IngestReport:
        report = IngestReport()
        for p in pdf_paths:
            try:
                report.chunks += self.ingest_pdf(p)
                report.files += 1
            except Exception:
                report.skipped.append(Path(p).name)
        return report

    # ------------------------------------------------------------- retrieval
    def prepare(self, question: str, history: list[dict] | None = None) -> Prepared:
        in_check = guardrails.check_input(question)
        if not in_check.ok:
            return Prepared(False, in_check.message, [], {}, question, {"input": in_check.flags})

        plan: dict = {}
        if settings.enable_query_expansion:
            try:
                plan = llm.plan_queries(question, history, hyde=settings.enable_hyde)
            except Exception:
                plan = {}

        effective = (plan.get("standalone") or question).strip()
        chunks = retrieval.retrieve(
            effective, self.store, plan=plan, docstore=self.docstore
        )

        ret_check = guardrails.check_retrieval(question, chunks)
        if not ret_check.ok:
            return Prepared(
                False,
                ret_check.message,
                [],
                plan,
                effective,
                {"input": in_check.flags, "retrieval": ret_check.flags},
            )
        return Prepared(
            True,
            "",
            chunks,
            plan,
            effective,
            {"input": in_check.flags, "retrieval": ret_check.flags},
        )

    def answer_stream(self, prepared: Prepared) -> tuple[str, Iterator[str]]:
        prompt = llm.build_prompt(prepared.effective_query, prepared.chunks)
        return llm.stream_answer(prompt)

    # ----------------------------------------------------------------- query
    def query(self, question: str, history: list[dict] | None = None) -> QueryResult:
        prep = self.prepare(question, history)
        if not prep.ok:
            return QueryResult(
                answer=prep.message,
                provider="guardrail",
                sources=[],
                guardrails=prep.guardrails,
            )

        prompt = llm.build_prompt(prep.effective_query, prep.chunks)
        result = llm.generate(prompt)
        answer = guardrails.normalize_answer(result.text)

        out_check = guardrails.check_output(answer, prep.chunks)
        if "no_citations" in out_check.flags:
            answer += _NO_CITE_NOTE

        verification = self.verify(prep.effective_query, answer, prep.chunks)

        return QueryResult(
            answer=answer,
            provider=result.provider,
            sources=prep.chunks,
            guardrails={**prep.guardrails, "output": out_check.flags},
            verification=verification,
            debug={
                "effective_query": prep.effective_query,
                "queries": [prep.effective_query, *(prep.plan.get("paraphrases") or [])],
                "prompt_chars": len(prompt),
            },
        )

    @staticmethod
    def verify(query: str, answer: str, chunks: list[dict]) -> dict:
        if not settings.enable_answer_verification:
            return {}
        try:
            return llm.verify_answer(query, answer, chunks)
        except Exception:
            return {}

    # ------------------------------------------------------------------ admin
    def stats(self) -> dict:
        return {
            "chunks": self.store.count(),
            "parents": self.docstore.count(),
            "sources": self.store.sources(),
        }

    def reset(self) -> None:
        self.store.reset()
        self.docstore.clear()
        llm.clear_cache()
        retrieval.clear_cache()

    @staticmethod
    def clear_cache() -> None:
        llm.clear_cache()
        retrieval.clear_cache()


__all__ = ["RAGPipeline", "IngestReport", "QueryResult", "Prepared"]
