"""Lightweight caching: in-memory LRU + JSON disk cache.

Used to avoid recomputing embeddings, reranking and LLM responses.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


def make_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


class LruDict:
    """Thread-safe bounded dict with least-recently-used eviction."""

    def __init__(self, maxsize: int = 2048) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return default

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class JsonDiskCache:
    """Simple persistent cache backed by a single JSON file."""

    def __init__(self, path: Path, enabled: bool = True, max_entries: int = 5000) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.max_entries = max_entries
        self._data: dict[str, Any] | None = None
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

    def get(self, key: str) -> Any:
        if not self.enabled:
            return None
        with self._lock:
            self._load()
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._load()
            self._data[key] = value
            if len(self._data) > self.max_entries:
                drop = max(1, self.max_entries // 10)
                for k in list(self._data.keys())[:drop]:
                    self._data.pop(k, None)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._data), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            try:
                self.path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
