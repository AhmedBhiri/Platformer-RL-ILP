from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional


class JsonlLogger:
    """
    Writes one JSON record per line.
    Safe for long runs, easy to parse, append-friendly.
    """

    def __init__(self, path: str, flush_every: int = 1) -> None:
        self.path = path
        self.flush_every = max(1, flush_every)
        self._n = 0

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")
        self._start_time = time.time()

    def log(self, record: Dict[str, Any]) -> None:
        # Add a timestamp and elapsed time for debugging / demo.
        record = dict(record)
        record["ts"] = time.time()
        record["elapsed_s"] = record["ts"] - self._start_time

        self._f.write(json.dumps(record) + "\n")
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
