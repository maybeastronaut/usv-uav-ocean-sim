from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CircleObstacle:
    center_x: float
    center_y: float
    radius: float

    def contains(self, x: float, y: float) -> bool:
        dx = x - self.center_x
        dy = y - self.center_y
        return dx * dx + dy * dy <= self.radius * self.radius
