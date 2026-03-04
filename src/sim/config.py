from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class SimConfig:
    seed: int = 0
    dt: float = 1.0
    steps: int = 200
    map_width: float = 1000.0
    map_height: float = 1000.0
    runs_dir: str = "runs"
    run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_run_dir(config: SimConfig) -> Path:
    base = Path(config.runs_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_id = config.run_name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    run_dir = base / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir
