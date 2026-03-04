from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.config import SimConfig
from sim.coverage.decay import apply_decay


@dataclass(frozen=True)
class GridShape:
    nx: int
    ny: int


class CoverageGrid:
    def __init__(self, config: SimConfig) -> None:
        if config.cell_size <= 0.0:
            raise ValueError("cell_size must be > 0")
        if config.decay_tau <= 0.0:
            raise ValueError("decay_tau must be > 0")

        self.config = config
        self.map_width = config.map_width
        self.map_height = config.map_height
        self.cell_size = config.cell_size
        self.decay_mode = config.decay_mode
        self.decay_tau = config.decay_tau

        self.nx = int(math.ceil(self.map_width / self.cell_size))
        self.ny = int(math.ceil(self.map_height / self.cell_size))
        self.shape = GridShape(nx=self.nx, ny=self.ny)

        self.x_centers = (np.arange(self.nx, dtype=float) + 0.5) * self.cell_size
        self.y_centers = (np.arange(self.ny, dtype=float) + 0.5) * self.cell_size
        self.last_observed_time = np.full((self.ny, self.nx), -np.inf, dtype=float)
        # True only after a cell has ever been observed at least once.
        self.visited_mask = np.zeros((self.ny, self.nx), dtype=bool)

    @property
    def visited_count(self) -> int:
        return int(np.count_nonzero(self.visited_mask))

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        ix = int(math.floor(x / self.cell_size))
        iy = int(math.floor(y / self.cell_size))
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return (ix, iy)

    def cell_center(self, ix: int, iy: int) -> tuple[float, float]:
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            raise IndexError("cell index out of bounds")
        return (self.x_centers[ix], self.y_centers[iy])

    def observe(self, position: tuple[float, float], radius: float, t: float) -> int:
        if radius <= 0.0:
            return 0
        px, py = position
        ix_min = max(0, int(math.floor((px - radius) / self.cell_size)))
        ix_max = min(self.nx - 1, int(math.floor((px + radius) / self.cell_size)))
        iy_min = max(0, int(math.floor((py - radius) / self.cell_size)))
        iy_max = min(self.ny - 1, int(math.floor((py + radius) / self.cell_size)))
        if ix_min > ix_max or iy_min > iy_max:
            return 0

        updated = 0
        r2 = radius * radius
        for iy in range(iy_min, iy_max + 1):
            cy = self.y_centers[iy]
            if cy < 0.0:
                continue
            for ix in range(ix_min, ix_max + 1):
                cx = self.x_centers[ix]
                dx = cx - px
                dy = cy - py
                if dx * dx + dy * dy <= r2:
                    self.last_observed_time[iy, ix] = t
                    self.visited_mask[iy, ix] = True
                    updated += 1
        return updated

    def info_map(self, t: float) -> np.ndarray:
        delta_t = np.maximum(0.0, t - self.last_observed_time)
        return apply_decay(delta_t=delta_t, mode=self.decay_mode, tau=self.decay_tau)

    def mean_info(self, t: float, mode: str = "all") -> float:
        # mode="all": include both visited and never-visited cells.
        # mode="visited": only include cells with at least one observation.
        values = self._select_info_values(self.info_map(t), mode=mode)
        return float(np.mean(values))

    def min_info(self, t: float, mode: str = "visited") -> float:
        # default "visited" avoids permanent zeros from never-observed cells.
        values = self._select_info_values(self.info_map(t), mode=mode)
        return float(np.min(values))

    def percentile_info(self, t: float, p: float, mode: str = "visited") -> float:
        # default "visited" focuses on quality inside observed area.
        values = self._select_info_values(self.info_map(t), mode=mode)
        return float(np.percentile(values, p))

    def metric_snapshot(self, t: float, percentile: float = 5.0) -> dict[str, float]:
        return {
            "time": float(t),
            "mean_all": self.mean_info(t, mode="all"),
            "min_visited": self.min_info(t, mode="visited"),
            f"p{int(percentile)}_visited": self.percentile_info(t, percentile, mode="visited"),
        }

    def _select_info_values(self, info: np.ndarray, mode: str) -> np.ndarray:
        if mode == "all":
            return info
        if mode == "visited":
            visited_values = info[self.visited_mask]
            # If nothing is observed yet, fall back to full map for stable behavior.
            if visited_values.size == 0:
                return info
            return visited_values
        raise ValueError(f"unknown mode: {mode}")


def check_resolution(cell_size: float, sensor_radius: float) -> dict[str, float | str]:
    if cell_size <= 0.0:
        raise ValueError("cell_size must be > 0")
    if sensor_radius <= 0.0:
        raise ValueError("sensor_radius must be > 0")

    ratio = sensor_radius / cell_size
    if ratio < 2.0:
        level = "WARNING"
        advice = (
            "R/CELL < 2. Coverage can look blocky and evaluation may be unstable."
        )
    elif ratio <= 10.0:
        level = "OK"
        advice = "R/CELL is in the recommended range [2, 10]."
    else:
        level = "NOTICE"
        advice = (
            "R/CELL > 10. Updates may touch too many cells and over-smooth coverage."
        )
    return {"ratio": float(ratio), "level": level, "message": advice}
