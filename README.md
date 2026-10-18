# DocuMind AI — Advanced RAG Agent

A document-question-answering agent that ingests PDFs and answers questions with
**grounded, cited responses**. It combines local embeddings, hybrid retrieval
(dense + sparse), cross-encoder reranking, query planning, guardrails,
answer verification, and a Groq-primary / Gemini-fallback LLM chain — all wrapped
in a Streamlit chat UI.

> **Last updated:** 2026-10-18 · Now deployable to Streamlit Community Cloud
> (secrets-based config, no `.env` required there). See
> [Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud).

---

## Table of contents

- [Features](#features)
- [Pipeline](#pipeline)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud)
- [Configuration](#configuration)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Evaluation](#evaluation)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Security notes](#security-notes)

---

## Features

- **PDF ingestion** — text extraction with de-hyphenation, whitespace cleanup and per-page citations.
- **Parent/child retrieval (small-to-big)** — match small chunks, send the full parent page to the LLM.
- **Optional contextual chunks** — LLM-generated context prepended to each chunk before embedding.
- **Local embeddings** — `BAAI/bge-base-en-v1.5` (768-dim), no external embedding API.
- **Hybrid search** — ChromaDB (cosine) + BM25 fused with Reciprocal Rank Fusion.
- **Cross-encoder reranking** — `BAAI/bge-reranker-base`.
- **Query planning** — one LLM call produces a standalone rewrite, multiple paraphrases, and a HyDE passage.
- **Guardrails** — prompt-injection detection, empty-context handling, citation validation.
- **Answer verification** — a strict auditor LLM scores groundedness and flags unsupported claims.
- **Streaming** — token-by-token output from Groq (fallback to Gemini).
- **Conversation memory** — follow-up questions resolved against chat history.
- **Caching** — embeddings, rerank scores and LLM responses are memoized (memory + disk).
- **Evaluation harness** — hit-rate, MRR, keyword recall, citation rate and groundedness.
- **Citations** — every answer links claims to source file and page; full-width citation
  markers (e.g. `【1】`) are normalized to `[1]`.

---

## Pipeline

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for diagrams, sequence flows and
design tradeoffs.

```
Upload PDF
   │
   ▼
Extract text  ──►  Clean & de-hyphenate  ──►  Chunk
   (pypdf)           (per page)               ├── parents  (full page  -> docstore)
                                              └── children (small slices -> Chroma + BM25)
   │
   ▼
Local embeddings (bge-base-en-v1.5)
   │
   ▼
Query planning (Groq/Gemini): standalone rewrite + paraphrases + HyDE
   │
   ▼
Hybrid retrieval  ──►  RRF fusion  ──►  Cross-encoder rerank  ──►  resolve to parents
   │
   ▼
Guardrails  ──►  Groq generates (Gemini fallback, streamed)
   │
   ▼
Grounding verification  ──►  Answer + citations in Streamlit
```

---

## Project structure

```
DocuMind_AI_Advanced_RAG_Agent/
├── app.py                     # Streamlit UI (chat, upload, toggles, citations)
├── verify_keys.py             # Standalone API-key checker for Groq + Gemini
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml           # Streamlit UI theme / browser stats
├── docs/
│   └── ARCHITECTURE.md        # Deep dive: diagrams, sequence flows, tradeoffs
├── .env                       # Your secrets and settings (gitignored)
├── .env.example               # Template
├── src/
│   ├── config.py              # Settings loaded from .env + paths
│   ├── ingestion.py           # PDF extraction + (parent/child) chunking
│   ├── embeddings.py          # Local sentence-transformers embeddings + cache
│   ├── vectorstore.py         # ChromaDB + BM25 hybrid store (RRF)
│   ├── retrieval.py           # Multi-query retrieve, rerank, parent resolution
│   ├── contextual.py          # Optional contextual-chunk enrichment
│   ├── guardrails.py          # Input/retrieval/output guards, citation normalization
│   ├── docstore.py            # JSON store for parent pages
│   ├── cache.py               # LRU (memory) + JSON disk cache
│   ├── llm.py                 # Groq + Gemini: generate, stream, plan, verify
│   └── rag.py                 # RAGPipeline orchestration
├── eval/
│   ├── run_eval.py            # Metrics harness
│   ├── make_sample_corpus.py  # Generates a demo PDF
│   ├── dataset.jsonl          # Golden Q&A set
│   └── sample_docs/           # Generated sample corpus
└── data/                      # Runtime state (gitignored)
    ├── chroma/                # ChromaDB persistence
    ├── uploads/               # Uploaded PDFs
    ├── cache/                 # LLM response cache
    └── parents.json           # Parent page docstore
```

---

## Requirements

- **Python 3.10+** (developed on 3.13)
- ~2 GB disk for `torch` (pulled by `sentence-transformers`) and model weights
- A **Groq** API key (primary) and a **Gemini** API key (fallback)

Dependencies (`requirements.txt`): `streamlit`, `pypdf`, `chromadb`, `rank-bm25`,
`sentence-transformers`, `requests`, `python-dotenv`, `fpdf2`.

---

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (downloads torch; can take several minutes)
python -m pip install -r requirements.txt

# 3. Configure secrets
Copy-Item .env.example .env
# then edit .env and add GROQ_API_KEY and GEMINI_API_KEY

# 4. Verify the keys
python verify_keys.py
```

---

## Deploy to Streamlit Cloud

The app is fully compatible with [Streamlit Community Cloud](https://share.streamlit.io/).

**1. Push the repo to GitHub** (make sure `.venv/`, `data/` and `.env` are
gitignored — they already are).

**2. Create the app** at <https://share.streamlit.io>:

- Repository: your fork/repo
- Branch: `main`
- Main file path: `app.py`

**3. Add secrets** — in **Manage app → Settings → Secrets**, paste (toml):

```toml
GROQ_API_KEY = "gsk_..."
GEMINI_API_KEY = "AIza..."
```

`src/config.py` reads `st.secrets` automatically on Streamlit Cloud (no `.env`
file exists there), and falls back to `.env` locally. All other settings from
the [Configuration](#configuration) table can also be placed in secrets.

**4. Deploy.** First deployment takes several minutes: the container installs
`requirements.txt` and downloads bge-base + bge-reranker weights on first query.

### Cloud-specific notes

- **Ephemeral storage** — each app container is ephemeral. The ChromaDB index,
  `data/parents.json` and caches reset when the app sleeps or reboots, so
  re-index your PDFs after a restart (they are lost too — upload again).
- **Free tier memory (~1 GB)** — running bge-base + bge-reranker + torch inside
  1 GB is tight. If the app crashes with `MemoryError` on the free tier,
  disable **Verify grounding** (skips the LLM verifier), turn off **HyDE**,
  or reduce **Reranked chunks**. Paid Cloud (1 GB→2 GB+ resources) is fine.
- **Model weights are re-downloaded** after every reboot; keep expectations of
  a slow first answer (~2–5 min) after each wake-up.
- **No `packages.txt` needed** — everything is pure pip.
- The API keys are never logged; rotate immediately if exposed.

---

## Configuration

All settings are read from `.env` (local) or `st.secrets` (Streamlit Cloud), loaded by
`src/config.py`. Anything omitted uses a default.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq key (primary LLM) |
| `GEMINI_API_KEY` | — | Google Gemini key (fallback LLM) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model (thinking disabled for speed) |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Local embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Single-level chunking (when parent-child off) |
| `USE_PARENT_CHILD` | `true` | Enable small-to-big retrieval |
| `CHILD_SIZE` / `CHILD_OVERLAP` | `350` / `80` | Child chunk size/overlap |
| `CONTEXTUAL_CHUNKS` | `false` | Prepend LLM context to chunks (slower indexing) |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | `20` / `20` | Candidates per retrieval method |
| `RERANK_TOP_N` | `5` | Chunks fed to the LLM |
| `MIN_RERANK_SCORE` | `-4.0` | Drop results below this rerank score |
| `RRF_K` | `60` | RRF constant |
| `ENABLE_QUERY_EXPANSION` | `true` | Rewrite + multi-query |
| `ENABLE_HYDE` | `true` | Hypothetical-document expansion |
| `NUM_QUERY_VARIANTS` | `3` | Paraphrases per question |
| `HISTORY_TURNS` | `4` | Turns of chat history used for rewriting |
| `ENABLE_ANSWER_VERIFICATION` | `true` | Groundedness audit |
| `ENABLE_CACHE` | `true` | Cache LLM responses |
| `TEMPERATURE` | `0.2` | Sampling temperature |
| `MAX_OUTPUT_TOKENS` | `2048` | Max answer length |

---

## Usage

### Run the chat app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then in the browser: **upload PDFs → Index documents → ask questions**.
The sidebar exposes toggles for query expansion, HyDE, verification, small-to-big,
contextual chunks and caching, plus sliders for top-N and temperature.

### Use the pipeline programmatically

```python
from src.rag import RAGPipeline

pipe = RAGPipeline()
pipe.ingest_pdf("my_report.pdf")          # or pipe.ingest_many([...])

result = pipe.query("What were the Q3 revenues?")
print(result.provider, "->", result.answer)
for i, s in enumerate(result.sources, 1):
    print(i, s["metadata"]["source"], "p", s["metadata"]["page"])

# Streaming
prepared = pipe.prepare("Summarize the risks section.")
provider, tokens = pipe.answer_stream(prepared)
print(provider, "".join(tokens))
```

---

## How it works

| Stage | Module | Notes |
|---|---|---|
| Extract & chunk | `ingestion.py` | Per-page cleaning; recursive splitting; `chunk_document_parent_child` returns parents + children |
| Embed | `embeddings.py` | bge-base, query gets the BGE instruction prefix; results memoized |
| Store | `vectorstore.py` | Children in Chroma (dense) + BM25 (sparse); RRF fusion |
| Parents | `docstore.py` | Full pages persisted as JSON, fetched by `parent_id` |
| Plan | `llm.plan_queries` | One call → `{standalone, paraphrases, hyde}` |
| Retrieve | `retrieval.py` | Runs hybrid search per query variant, RRF-fuses, reranks, resolves to parents |
| Guard | `guardrails.py` | Injection patterns, empty-context, citation validation, normalization |
| Generate | `llm.generate` / `stream_answer` | Groq first; Gemini fallback; SSE streamed with forced UTF-8 |
| Verify | `llm.verify_answer` | Returns `{grounded, score, unsupported_claims, reason}` |
| Orchestrate | `rag.py` | `prepare()` → `answer_stream()` / `query()` |

---

## Evaluation

```powershell
# 1. Generate a demo corpus (2 pages)
.\.venv\Scripts\python.exe eval\make_sample_corpus.py

# 2. Index it and run the metrics
.\.venv\Scripts\python.exe eval\run_eval.py `
    --dataset eval\dataset.jsonl `
    --pdf eval\sample_docs\sample_corpus.pdf `
    --reset
```

Output columns: retrieval **Hit**, **MRR**, keyword recall (**KW**), **Cite**,
**Ground**. Aggregate metrics are printed at the bottom.

Dataset format (`eval/dataset.jsonl`), one JSON object per line:

```json
{"question": "How tall is the Eiffel Tower?", "expected_source": "sample_corpus.pdf", "expected_page": 1, "expected_keywords": ["330"]}
```

Options: `--top-n`, `--out results.json`, `--reset`, `--pdf <paths...>`.

Reference result on the sample corpus:

```
Hit-rate 100.0%  MRR 1.000  Keyword-recall 100.0%  Citation 100.0%  Grounded 100.0%
```

---

## Performance

Caching makes repeat queries near-instant:

| Scenario | Time |
|---|---|
| First query (cold models + plan + generate + verify) | ~45 s |
| Same query again (cached) | ~0.03 s |

First run also downloads model weights (bge-base ~440 MB) into the Hugging Face cache.
Clear all caches from the sidebar or `RAGPipeline.clear_cache()`.

---

## Troubleshooting

- **`HTTP 403 error code: 1010` from Groq** — Cloudflare blocks generic user-agents;
  the client already sends a browser-like `User-Agent`.
- **Gemini `404 ... no longer available`** — Google retires models for new keys;
  the default is `gemini-3.8-flash`. If a call hangs, thinking is disabled via
  `thinkingConfig.thinkingBudget = 0`.
- **Garbled `【1】` / mojibake in streamed answers** — fixed by forcing UTF-8 SSE
  decoding; `normalize_answer` also converts full-width markers to `[1]`.
- **Hugging Face symlink warning on Windows** — harmless; enable Developer Mode to silence.
- **Empty answers / "knowledge base is empty"** — index documents first.
- **Port already in use** — run with `--server.port 8600`.

---

## Security notes

- `.env` is excluded from version control via `.gitignore`. Never commit real keys.
- API keys are read from the environment only; they are never logged.
- If a key is ever exposed (e.g. pasted into a chat or issue), rotate it.
