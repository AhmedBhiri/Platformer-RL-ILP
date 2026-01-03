from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional


class JsonlLogger:
    """
    Writes one JSON record per line (jsonl).
    Append-friendly, easy to parse, robust for long runs.
    """

    def __init__(self, path: str, flush_every: int = 1) -> None:
        self.path = path
        self.flush_every = max(1, int(flush_every))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")
        self._n = 0

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", time.time())
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n % self.flush_every == 0:
            self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
        finally:
            self._f.close()


def obs_to_dict(obs: Any) -> Dict[str, Any]:
    """Convert Obs dataclass (or similar) into plain dict."""
    if is_dataclass(obs):
        return asdict(obs)
    if hasattr(obs, "__dict__"):
        return dict(obs.__dict__)
    return {"obs": str(obs)}
