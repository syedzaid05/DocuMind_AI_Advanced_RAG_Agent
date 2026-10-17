"""PDF text extraction and recursive chunking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .config import settings

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
_WS = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=\w)", "", text)  # de-hyphenate line breaks
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Return [{page: int, text: str}] for a PDF (1-indexed pages)."""
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        text = clean_text(raw)
        if text:
            pages.append({"page": i, "text": text})
    return pages


def _split_recursive(text: str, size: int, overlap: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []

    sep = seps[0] if seps else ""
    rest = seps[1:] if len(seps) > 1 else []

    if sep == "":
        # Hard character split with overlap.
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + size])
            start += size - overlap
        return [c for c in chunks if c.strip()]

    pieces = text.split(sep)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + sep + piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(piece) > size:
            chunks.extend(_split_recursive(piece, size, overlap, rest))
            current = ""
        else:
            current = piece
    if current.strip():
        chunks.append(current)

    # Apply overlap by merging tails of previous chunk.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap:] if len(prev) > overlap else prev
            overlapped.append(tail + " " + cur)
        chunks = overlapped
    return [c.strip() for c in chunks if c.strip()]


def chunk_document(
    pages: Iterable[dict],
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Split page text into overlapping chunks carrying citation metadata."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    overlap = min(overlap, max(size - 1, 0))

    chunks: list[dict] = []
    idx = 0
    for page in pages:
        for piece in _split_recursive(page["text"], size, overlap, _SEPARATORS):
            chunks.append(
                {
                    "id": f"{source}::p{page['page']}::c{idx}",
                    "text": piece,
                    "metadata": {
                        "source": source,
                        "page": page["page"],
                        "chunk_index": idx,
                    },
                }
            )
            idx += 1
    return chunks


def chunk_document_parent_child(
    pages: Iterable[dict],
    source: str,
    child_size: int | None = None,
    child_overlap: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (parents, children).

    A parent is a whole page (full context); children are small overlapping
    slices of that page used for precise matching. Each child carries a
    ``parent_id`` so matches can be resolved back to the parent context.
    """
    csize = child_size or settings.child_size
    coverlap = child_overlap if child_overlap is not None else settings.child_overlap
    coverlap = min(coverlap, max(csize - 1, 0))

    parents: list[dict] = []
    children: list[dict] = []
    for page in pages:
        pid = f"{source}::p{page['page']}"
        parents.append(
            {
                "id": pid,
                "text": page["text"],
                "metadata": {
                    "source": source,
                    "page": page["page"],
                    "kind": "parent",
                },
            }
        )
        for i, piece in enumerate(
            _split_recursive(page["text"], csize, coverlap, _SEPARATORS)
        ):
            children.append(
                {
                    "id": f"{pid}::c{i}",
                    "text": piece,
                    "metadata": {
                        "source": source,
                        "page": page["page"],
                        "parent_id": pid,
                        "kind": "child",
                        "chunk_index": i,
                    },
                }
            )
    return parents, children
