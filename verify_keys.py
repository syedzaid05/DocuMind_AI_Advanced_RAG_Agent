"""Verify GROQ_API_KEY and GEMINI_API_KEY from .env.

Usage:
    python verify_keys.py

Uses only the Python standard library (no pip installs needed).
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env(path: Path) -> None:
    if not path.exists():
        sys.exit(f"Missing {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def request(url: str, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return 0, str(e.reason)


def check_groq(key: str) -> bool:
    print("Checking Groq...")
    if not key:
        print("  [FAIL] GROQ_API_KEY is empty")
        return False
    url = "https://api.groq.com/openai/v1/models"
    status, body = request(
        url,
        {
            "Authorization": f"Bearer {key}",
            "User-Agent": "curl/8.7.1",
            "Accept": "application/json",
        },
    )
    if status == 200:
        try:
            models = [m["id"] for m in json.loads(body).get("data", [])]
        except json.JSONDecodeError:
            models = []
        print(f"  [OK] {len(models)} models available, e.g. {models[:3]}")
        return True
    print(f"  [FAIL] HTTP {status}: {body[:300]}")
    return False


def check_gemini(key: str) -> bool:
    print("Checking Gemini...")
    if not key:
        print("  [FAIL] GEMINI_API_KEY is empty")
        return False
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    status, body = request(url, {})
    if status == 200:
        try:
            models = [m["name"] for m in json.loads(body).get("models", [])]
        except json.JSONDecodeError:
            models = []
        print(f"  [OK] {len(models)} models available, e.g. {models[:3]}")
        return True
    print(f"  [FAIL] HTTP {status}: {body[:300]}")
    return False


def main() -> int:
    load_env(ENV_PATH)
    groq_ok = check_groq(os.environ.get("GROQ_API_KEY", ""))
    gemini_ok = check_gemini(os.environ.get("GEMINI_API_KEY", ""))

    print()
    if groq_ok and gemini_ok:
        print("All keys valid.")
        return 0
    print("Some keys failed. See details above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
