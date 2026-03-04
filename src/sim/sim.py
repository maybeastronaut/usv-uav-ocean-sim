from __future__ import annotations

import csv
import random
from pathlib import Path

from sim.config import SimConfig


class Sim:
    def __init__(self, config: SimConfig, metrics_path: Path) -> None:
        self.config = config
        self.t = 0.0
        self.step_count = 0
        self.state: dict[str, object] = {}
        self._rng = random.Random(config.seed)
        self.metrics_path = metrics_path
        self._init_metrics_file()

    def _init_metrics_file(self) -> None:
        with self.metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "time", "dummy_metric"])
            writer.writeheader()

    def step(self) -> None:
        self.t += self.config.dt
        self.step_count += 1

        row = {
            "step": self.step_count,
            "time": self.t,
            "dummy_metric": self._rng.random(),
        }
        with self.metrics_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "time", "dummy_metric"])
            writer.writerow(row)

    def run(self) -> None:
        for _ in range(self.config.steps):
            self.step()
