"""Evaluation harness for DocuMind.

Measures retrieval and answer quality against a golden dataset.

Dataset: JSONL, one object per line:
    {
      "question": "...",
      "expected_source": "sample_corpus.pdf",   # optional
      "expected_page": 1,                        # optional
      "expected_keywords": ["330", "meters"],    # optional
      "expect_no_answer": false                  # optional
    }

Usage:
    python eval/run_eval.py --dataset eval/dataset.jsonl \
        --pdf eval/sample_docs/sample_corpus.pdf --reset
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402

_CITE_RE = re.compile(r"[\[\(【]\s*(\d+)\s*[\]\)】]")
_REFUSAL_HINTS = ("don't know", "do not know", "not contain", "no information", "isn't")


def load_dataset(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    low = answer.lower()
    return sum(1 for k in keywords if k.lower() in low) / len(keywords)


def evaluate_item(pipeline: RAGPipeline, item: dict) -> dict:
    result = pipeline.query(item["question"])
    sources = result.sources
    pairs = [
        (s["metadata"].get("source"), s["metadata"].get("page")) for s in sources
    ]

    exp_source = item.get("expected_source")
    exp_page = item.get("expected_page")

    if exp_source:
        hit = any(
            src == exp_source and (exp_page is None or pg == exp_page)
            for src, pg in pairs
        )
        rr = 0.0
        for rank, (src, pg) in enumerate(pairs, start=1):
            if src == exp_source and (exp_page is None or pg == exp_page):
                rr = 1.0 / rank
                break
    else:
        hit = len(sources) > 0
        rr = 0.0

    answer = result.answer
    low = answer.lower()
    refusal = any(h in low for h in _REFUSAL_HINTS)

    return {
        "question": item["question"],
        "retrieval_hit": bool(hit),
        "reciprocal_rank": rr,
        "keyword_recall": keyword_recall(answer, item.get("expected_keywords", [])),
        "has_citation": bool(_CITE_RE.search(answer)),
        "grounded": bool(result.verification.get("grounded", False)),
        "grounded_score": float(result.verification.get("score", 0.0)),
        "provider": result.provider,
        "refused": refusal,
        "expect_no_answer": bool(item.get("expect_no_answer", False)),
        "answer": answer,
    }


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuMind evaluation harness")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--pdf", nargs="*", default=[], help="PDFs to ingest first")
    parser.add_argument("--reset", action="store_true", help="Reset the index first")
    parser.add_argument("--top-n", type=int, default=None, help="Reranked chunks")
    parser.add_argument("--out", default=None, help="Write JSON results here")
    args = parser.parse_args()

    if args.top_n:
        settings.rerank_top_n = args.top_n

    pipeline = RAGPipeline()
    if args.reset:
        pipeline.reset()
    if args.pdf:
        report = pipeline.ingest_many(args.pdf)
        print(f"Indexed {report.files} file(s), {report.chunks} child chunks.\n")

    items = load_dataset(Path(args.dataset))
    if not items:
        print("Empty dataset.")
        return 1

    rows = [evaluate_item(pipeline, item) for item in items]

    print(f"{'Question':<44} {'Hit':>4} {'RR':>5} {'KW':>6} {'Cite':>5} {'Ground':>7}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['question'][:43]:<44} "
            f"{'Y' if row['retrieval_hit'] else 'n':>4} "
            f"{row['reciprocal_rank']:>5.2f} "
            f"{_pct(row['keyword_recall']):>6} "
            f"{'Y' if row['has_citation'] else 'n':>5} "
            f"{_pct(row['grounded_score']):>7}"
        )

    n = len(rows)
    summary = {
        "n": n,
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in rows) / n,
        "mrr": sum(r["reciprocal_rank"] for r in rows) / n,
        "keyword_recall": statistics.fmean(r["keyword_recall"] for r in rows),
        "citation_rate": sum(r["has_citation"] for r in rows) / n,
        "grounded_rate": sum(r["grounded"] for r in rows) / n,
        "avg_grounded_score": statistics.fmean(r["grounded_score"] for r in rows),
    }

    print("-" * 80)
    print(
        f"Hit-rate {_pct(summary['retrieval_hit_rate'])}  "
        f"MRR {summary['mrr']:.3f}  "
        f"Keyword-recall {_pct(summary['keyword_recall'])}  "
        f"Citation {_pct(summary['citation_rate'])}  "
        f"Grounded {_pct(summary['grounded_rate'])} "
        f"(avg {_pct(summary['avg_grounded_score'])})"
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
