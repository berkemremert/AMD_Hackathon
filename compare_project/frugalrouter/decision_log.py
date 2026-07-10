from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from frugalrouter.types import RouteResult, Task


class DecisionLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("FRUGAL_DECISION_LOG", "logs/decisions.jsonl"))

    def write(self, task: Task, result: RouteResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": asdict(task),
            "result": asdict(result),
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")

