"""Shared utility helpers for the NLP Policy Chatbot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_directory(path: Path) -> None:
    """Create a directory and its parents when needed."""
    path.mkdir(parents=True, exist_ok=True)


def project_relative(path: Path) -> str:
    """Return a readable path string for logs and CLI output."""
    try:
        from src.config import PROJECT_ROOT

        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries as UTF-8 JSON Lines."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON Lines file into dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dictionary as pretty UTF-8 JSON."""
    ensure_directory(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
