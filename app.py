"""DocuMind AI - Advanced RAG Agent (Streamlit UI).

Upload PDF -> extract -> chunk -> local embeddings (bge-base) -> ChromaDB + BM25
-> query planning (rewrite/multi-query/HyDE) -> hybrid retrieve -> rerank ->
guardrails -> Groq/Gemini streamed answer + grounding verification + citations.
"""

from __future__ import annotations

import streamlit as st

from src import guardrails
from src.config import UPLOAD_DIR, settings
from src.rag import RAGPipeline

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def save_uploads(files) -> list[str]:
    paths: list[str] = []
    for f in files:
        dest = UPLOAD_DIR / f.name
        dest.write_bytes(f.getbuffer())
        paths.append(str(dest))
    return paths


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, c in enumerate(sources, start=1):
            meta = c["metadata"]
            score = c.get("rerank_score")
            score_txt = f" · rerank {score:.2f}" if score is not None else ""
            st.markdown(
                f"**[{i}] {meta.get('source')} — page {meta.get('page')}**{score_txt}"
            )
            snippet = c.get("matched_child") or c["text"]
            snippet = snippet[:600] + ("…" if len(snippet) > 600 else "")
            label = "Matched chunk: " if c.get("matched_child") else ""
            st.caption(label + snippet)


def render_verification(verification: dict) -> None:
    if not verification:
        return
    grounded = verification.get("grounded")
    score = verification.get("score", 0.0)
    if grounded:
        st.success(f"✓ Grounded — confidence {score:.0%}")
    else:
        st.warning(f"⚠ Weak grounding — confidence {score:.0%}")
    reason = verification.get("reason")
    if reason:
        st.caption(f"Verifier: {reason}")
    unsupported = verification.get("unsupported_claims") or []
    if unsupported:
        with st.expander("Unsupported claims flagged by verifier"):
            for claim in unsupported:
                st.markdown(f"- {claim}")


def sidebar(pipeline: RAGPipeline) -> None:
    with st.sidebar:
        st.header("Knowledge base")
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Index documents", type="primary", use_container_width=True):
            if not uploads:
                st.warning("Add at least one PDF first.")
            else:
                with st.spinner("Extracting, chunking and embedding…"):
                    report = pipeline.ingest_many(save_uploads(uploads))
                st.success(f"Indexed {report.files} file(s) → {report.chunks} chunks.")
                if report.skipped:
                    st.error("Skipped: " + ", ".join(report.skipped))

        stats = pipeline.stats()
        c1, c2 = st.columns(2)
        c1.metric("Child chunks", stats["chunks"])
        c2.metric("Parent pages", stats.get("parents", 0))
        if stats["sources"]:
            st.markdown("**Documents**")
            for name in stats["sources"]:
                st.markdown(f"- {name}")

        st.divider()
        st.subheader("Answer quality")
        settings.rerank_top_n = st.slider(
            "Reranked chunks per answer", 2, 10, settings.rerank_top_n
        )
        settings.temperature = st.slider(
            "Temperature", 0.0, 1.0, settings.temperature, 0.05
        )
        settings.enable_query_expansion = st.toggle(
            "Query rewrite + multi-query", value=settings.enable_query_expansion,
            help="Rewrite follow-ups and search with paraphrases for higher recall.",
        )
        settings.enable_hyde = st.toggle(
            "HyDE (hypothetical document)", value=settings.enable_hyde,
            help="Use a generated hypothetical passage to improve dense retrieval.",
        )
        settings.enable_answer_verification = st.toggle(
            "Verify grounding", value=settings.enable_answer_verification,
            help="Audit the answer against the retrieved context.",
        )

        st.subheader("Indexing")
        settings.use_parent_child = st.toggle(
            "Small-to-big retrieval", value=settings.use_parent_child,
            help="Match small chunks but send the full parent page to the LLM. Re-index to apply.",
        )
        settings.contextual_chunks = st.toggle(
            "Contextual chunks (slower)", value=settings.contextual_chunks,
            help="Prepend LLM-generated context to each chunk before embedding. Re-index to apply.",
        )
        settings.enable_cache = st.toggle(
            "Cache LLM responses", value=settings.enable_cache
        )
        if st.button("Clear caches", use_container_width=True):
            pipeline.clear_cache()
            st.success("Caches cleared.")

        st.divider()
        if st.button("Reset knowledge base", use_container_width=True):
            pipeline.reset()
            st.session_state.messages = []
            st.success("Knowledge base cleared.")


def chat(pipeline: RAGPipeline) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.info("Upload and index PDFs, then ask a question about your documents.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])
            if msg.get("verification"):
                render_verification(msg["verification"])

    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if pipeline.stats()["chunks"] == 0:
            answer = "Your knowledge base is empty. Upload PDFs first."
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": []}
            )
            return

        with st.spinner("Planning query and retrieving…"):
            prepared = pipeline.prepare(question, history)

        if not prepared.ok:
            st.markdown(prepared.message)
            st.session_state.messages.append(
                {"role": "assistant", "content": prepared.message, "sources": []}
            )
            return

        try:
            provider, tokens = pipeline.answer_stream(prepared)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Generation failed: {exc}")
            return

        placeholder = st.empty()
        raw = ""
        for delta in tokens:
            raw += delta
            placeholder.markdown(raw + "▌")

        answer = guardrails.normalize_answer(raw)
        if "no_citations" in guardrails.check_output(answer, prepared.chunks).flags:
            answer += "\n\n_(No inline citations were produced; review sources below.)_"
        placeholder.markdown(answer)

        st.caption(f"Answered by: {provider}")
        verification = pipeline.verify(prepared.effective_query, answer, prepared.chunks)
        render_verification(verification)
        render_sources(prepared.chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": prepared.chunks,
            "verification": verification,
        }
    )


def main() -> None:
    pipeline = get_pipeline()

    st.title("📄 DocuMind AI")
    st.caption(
        f"Hybrid RAG · local embeddings (`{settings.embedding_model}`) · "
        f"rerank (`{settings.reranker_model}`) · Groq → Gemini fallback"
    )

    sidebar(pipeline)
    chat(pipeline)


if __name__ == "__main__":
    main()
