from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from sim.config import SimConfig
from sim.coverage.coverage_grid import CoverageGrid


@dataclass(frozen=True)
class RegionShape:
    nrx: int
    nry: int


class RegionMap:
    def __init__(self, grid: CoverageGrid, config: SimConfig) -> None:
        if config.region_cell_size <= 0.0:
            raise ValueError("region_cell_size must be > 0")
        self.grid = grid
        self.config = config
        self.region_cell_size = config.region_cell_size
        self.nrx = int(math.ceil(config.map_width / self.region_cell_size))
        self.nry = int(math.ceil(config.map_height / self.region_cell_size))
        self.shape = RegionShape(nrx=self.nrx, nry=self.nry)

    def region_center(self, rx: int, ry: int) -> tuple[float, float]:
        self._check_region_bounds(rx, ry)
        x0, x1, y0, y1 = self.region_bounds(rx, ry)
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)

    def region_bounds(self, rx: int, ry: int) -> tuple[float, float, float, float]:
        self._check_region_bounds(rx, ry)
        x0 = rx * self.region_cell_size
        y0 = ry * self.region_cell_size
        x1 = min((rx + 1) * self.region_cell_size, self.config.map_width)
        y1 = min((ry + 1) * self.region_cell_size, self.config.map_height)
        return (x0, x1, y0, y1)

    def region_target_with_offset(
        self,
        rx: int,
        ry: int,
        rng: random.Random,
        offset_ratio: float = 0.25,
    ) -> tuple[float, float]:
        x0, x1, y0, y1 = self.region_bounds(rx, ry)
        cx, cy = self.region_center(rx, ry)
        span_x = x1 - x0
        span_y = y1 - y0
        dx = rng.uniform(-offset_ratio * span_x, offset_ratio * span_x)
        dy = rng.uniform(-offset_ratio * span_y, offset_ratio * span_y)
        tx = cx + dx
        ty = cy + dy
        tx = min(max(tx, x0), x1)
        ty = min(max(ty, max(0.0, y0)), min(self.config.map_height, y1))
        return (tx, ty)

    def grid_cell_ranges(self, rx: int, ry: int) -> tuple[slice, slice]:
        self._check_region_bounds(rx, ry)
        x0, x1, y0, y1 = self.region_bounds(rx, ry)

        ix0 = int(np.searchsorted(self.grid.x_centers, x0, side="left"))
        ix1 = int(np.searchsorted(self.grid.x_centers, x1, side="left"))
        iy0 = int(np.searchsorted(self.grid.y_centers, y0, side="left"))
        iy1 = int(np.searchsorted(self.grid.y_centers, y1, side="left"))
        return (slice(iy0, iy1), slice(ix0, ix1))

    def grid_cells_in_region(self, rx: int, ry: int) -> list[tuple[int, int]]:
        y_slice, x_slice = self.grid_cell_ranges(rx, ry)
        cells: list[tuple[int, int]] = []
        for iy in range(y_slice.start, y_slice.stop):
            for ix in range(x_slice.start, x_slice.stop):
                cells.append((ix, iy))
        return cells

    def region_info_map(self, t: float, agg_mode: str | None = None) -> np.ndarray:
        mode = agg_mode or self.config.region_agg_mode
        info = self.grid.info_map(t)
        out = np.zeros((self.nry, self.nrx), dtype=float)
        for ry in range(self.nry):
            for rx in range(self.nrx):
                y_slice, x_slice = self.grid_cell_ranges(rx, ry)
                region_vals = info[y_slice, x_slice]
                if region_vals.size == 0:
                    out[ry, rx] = 0.0
                    continue
                if mode == "mean":
                    out[ry, rx] = float(np.mean(region_vals))
                elif mode == "min":
                    out[ry, rx] = float(np.min(region_vals))
                elif mode == "p5":
                    out[ry, rx] = float(np.percentile(region_vals, 5))
                else:
                    raise ValueError(f"unknown region agg mode: {mode}")
        return np.clip(out, 0.0, 1.0)

    def _check_region_bounds(self, rx: int, ry: int) -> None:
        if not (0 <= rx < self.nrx and 0 <= ry < self.nry):
            raise IndexError("region index out of bounds")
