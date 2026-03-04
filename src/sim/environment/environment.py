from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from sim.config import SimConfig
from sim.environment.obstacle import CircleObstacle

RegionName = Literal["nearshore", "risk_zone", "offshore"]


@dataclass(frozen=True)
class YBand:
    y_min: float
    y_max: float


class Environment2D:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.map_width = config.map_width
        self.map_height = config.map_height
        self.base_x = config.base_x if config.base_x is not None else config.map_width / 2.0
        self.base_y = config.base_y if config.base_y is not None else 0.0
        self.comm_radius = config.comm_radius
        self._rng = random.Random(config.seed)

        if not (0.0 <= self.base_x <= self.map_width and 0.0 <= self.base_y <= self.map_height):
            raise ValueError("base_position must be inside map bounds")
        if not math.isclose(self.base_y, 0.0, abs_tol=1e-9):
            raise ValueError("base_position must be on coastline y=0")
        if not (
            0.0 <= config.nearshore_y_max <= config.risk_zone_y_min
            and config.risk_zone_y_min <= config.risk_zone_y_max <= config.offshore_y_min
            and config.offshore_y_min <= config.offshore_y_max <= config.map_height
        ):
            raise ValueError("invalid y-band boundaries for nearshore/risk_zone/offshore")
        if config.risk_obstacle_radius_min <= 0.0:
            raise ValueError("risk_obstacle_radius_min must be > 0")
        if config.risk_obstacle_radius_min > config.risk_obstacle_radius_max:
            raise ValueError("risk_obstacle_radius_min must be <= risk_obstacle_radius_max")

        self.bands: dict[RegionName, YBand] = {
            "nearshore": YBand(0.0, config.nearshore_y_max),
            "risk_zone": YBand(config.risk_zone_y_min, config.risk_zone_y_max),
            "offshore": YBand(config.offshore_y_min, config.offshore_y_max),
        }
        self.obstacles: list[CircleObstacle] = self._generate_risk_obstacles(config.risk_obstacle_count)

    @property
    def base_position(self) -> tuple[float, float]:
        return (self.base_x, self.base_y)

    def distance_to_base(self, x: float, y: float) -> float:
        return math.hypot(x - self.base_x, y - self.base_y)

    def is_inside_map(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.map_width and 0.0 <= y <= self.map_height

    def is_in_region(self, x: float, y: float, region: RegionName) -> bool:
        if not self.is_inside_map(x, y):
            return False
        band = self.bands[region]
        return band.y_min <= y <= band.y_max

    def is_in_obstacle(self, x: float, y: float) -> bool:
        return any(ob.contains(x, y) for ob in self.obstacles)

    def current_at(self, x: float, y: float, t: float) -> tuple[float, float]:
        region = self._region_of_y(y)
        if region == "nearshore":
            base_vx, base_vy = self.config.current_near
        elif region == "risk_zone":
            base_vx, base_vy = self.config.current_risk
        else:
            base_vx, base_vy = self.config.current_offshore

        noise_amp = self.config.current_noise_amplitude
        noise_x = noise_amp * math.sin(0.0011 * x + 0.0017 * y + 0.137 * self.config.seed)
        noise_y = noise_amp * math.cos(0.0013 * x + 0.0015 * y + 0.173 * self.config.seed)

        time_amp = self.config.current_time_variation_amplitude
        time_term = self._sin_time_term(t, self.config.current_time_variation_period)
        tvx = time_amp * 0.70 * time_term
        tvy = time_amp * 0.30 * time_term

        return (base_vx + noise_x + tvx, base_vy + noise_y + tvy)

    def wind_at(self, x: float, y: float, t: float) -> tuple[float, float]:
        base_wx, base_wy = self.config.wind_base
        y_norm = 0.0 if self.map_height <= 0 else max(0.0, min(1.0, y / self.map_height))
        speed_gain = 1.0 + self.config.wind_y_gradient * y_norm
        wx = base_wx * speed_gain
        wy = base_wy * speed_gain

        amp = self.config.wind_time_variation_amplitude
        time_term = self._sin_time_term(t, self.config.wind_time_variation_period)
        base_speed = math.hypot(base_wx, base_wy)
        if base_speed <= 1e-9:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = base_wx / base_speed, base_wy / base_speed
        wx += amp * ux * time_term
        wy += amp * uy * time_term
        return (wx, wy)

    def in_comm_range(
        self, p1: tuple[float, float], p2: tuple[float, float]
    ) -> bool:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy) <= self.comm_radius

    def comm_quality(
        self, p1: tuple[float, float], p2: tuple[float, float]
    ) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist = math.hypot(dx, dy)
        if self.comm_radius <= 0:
            return 0.0
        quality = 1.0 - dist / self.comm_radius
        return max(0.0, min(1.0, quality))

    def field_grid_points(self, step: float) -> list[tuple[float, float]]:
        if step <= 0:
            raise ValueError("step must be > 0")
        xs: list[float] = []
        ys: list[float] = []

        x = 0.0
        while x <= self.map_width + 1e-9:
            xs.append(min(x, self.map_width))
            x += step
        if abs(xs[-1] - self.map_width) > 1e-9:
            xs.append(self.map_width)

        y = 0.0
        while y <= self.map_height + 1e-9:
            ys.append(min(y, self.map_height))
            y += step
        if abs(ys[-1] - self.map_height) > 1e-9:
            ys.append(self.map_height)

        points: list[tuple[float, float]] = []
        for gy in ys:
            for gx in xs:
                points.append((gx, gy))
        return points

    def sample_point(self, region: RegionName, max_attempts: int = 10000) -> tuple[float, float]:
        band = self.bands[region]
        for _ in range(max_attempts):
            x = self._rng.uniform(0.0, self.map_width)
            y = self._rng.uniform(band.y_min, band.y_max)
            if not self.is_inside_map(x, y):
                continue
            if self.is_in_obstacle(x, y):
                continue
            return (x, y)
        raise RuntimeError(f"failed to sample a point in region={region}")

    def _generate_risk_obstacles(self, count: int) -> list[CircleObstacle]:
        obstacles: list[CircleObstacle] = []
        risk = self.bands["risk_zone"]

        for _ in range(count):
            radius = self._rng.uniform(
                self.config.risk_obstacle_radius_min,
                self.config.risk_obstacle_radius_max,
            )
            y_min = risk.y_min + radius
            y_max = risk.y_max - radius
            x_min = radius
            x_max = self.map_width - radius
            if y_min > y_max or x_min > x_max:
                raise ValueError(
                    "invalid obstacle/risk-zone setup: obstacle does not fit in map/risk band"
                )

            placed = False
            for _attempt in range(2000):
                cx = self._rng.uniform(x_min, x_max)
                cy = self._rng.uniform(y_min, y_max)

                if cx - radius < 0.0 or cx + radius > self.map_width:
                    continue
                if cy - radius < 0.0 or cy + radius > self.map_height:
                    continue
                if not (risk.y_min <= cy <= risk.y_max):
                    continue

                obstacles.append(CircleObstacle(center_x=cx, center_y=cy, radius=radius))
                placed = True
                break

            if not placed:
                raise RuntimeError("failed to place an obstacle in risk_zone")

        return obstacles

    def _region_of_y(self, y: float) -> RegionName:
        if y <= self.bands["nearshore"].y_max:
            return "nearshore"
        if y <= self.bands["risk_zone"].y_max:
            return "risk_zone"
        return "offshore"

    @staticmethod
    def _sin_time_term(t: float, period: float) -> float:
        if period <= 0.0:
            return 0.0
        return math.sin(2.0 * math.pi * (t / period))
