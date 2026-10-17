"""Persistent JSON store for parent documents (small-to-big retrieval)."""

from __future__ import annotations

import json
import threading
from pathlib import Path


class JsonDocStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, dict] | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._data is not None:
            return
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def upsert(self, records: list[dict]) -> None:
        with self._lock:
            self._load()
            for rec in records:
                self._data[rec["id"]] = rec
            self._save()

    def get(self, ids: list[str]) -> list[dict]:
        with self._lock:
            self._load()
            return [self._data[i] for i in ids if i in self._data]

    def delete_source(self, source: str) -> None:
        with self._lock:
            self._load()
            self._data = {
                k: v
                for k, v in self._data.items()
                if v.get("metadata", {}).get("source") != source
            }
            self._save()

    def count(self) -> int:
        with self._lock:
            self._load()
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._save()
