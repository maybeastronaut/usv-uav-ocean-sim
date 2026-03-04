from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


AgentType = Literal["UAV", "USV"]
AgentTaskStatus = Literal["idle", "moving", "working", "waiting_recharge", "charging", "emergency"]


@dataclass
class BaseAgent:
    agent_id: str
    agent_type: AgentType
    pos: tuple[float, float]
    max_speed: float
    sensor_radius: float
    comm_radius: float
    vel: tuple[float, float] = (0.0, 0.0)
    current_task_id: int | None = None
    task_status: AgentTaskStatus = "idle"
    alive: bool = True
    last_seen_time: float = 0.0
    energy: float = 0.0
    low_energy_threshold: float = 0.0
    goal_pos: tuple[float, float] | None = None
    current_waypoints: list[tuple[float, float]] = field(default_factory=list)
    current_wp_idx: int = 0
    cruise_goal: tuple[float, float] | None = None
    stats: dict[str, float] = field(
        default_factory=lambda: {
            "distance": 0.0,
            "work_ticks": 0.0,
            "avoid_failures": 0.0,
        }
    )

    def distance_to(self, target: tuple[float, float]) -> float:
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        return math.hypot(dx, dy)

    def set_task(self, task_id: int | None) -> None:
        self.current_task_id = task_id
        self.task_status = "idle" if task_id is None else "moving"
        if task_id is None:
            self.goal_pos = None
            self.current_waypoints = []
            self.current_wp_idx = 0

    def clear_task(self) -> None:
        self.current_task_id = None
        self.task_status = "idle"
        self.goal_pos = None
        self.current_waypoints = []
        self.current_wp_idx = 0

    def clamp_to_map(self, map_width: float, map_height: float) -> None:
        x = min(max(self.pos[0], 0.0), map_width)
        y = min(max(self.pos[1], 0.0), map_height)
        self.pos = (x, y)

    def _update_distance_stat(self, old_pos: tuple[float, float], new_pos: tuple[float, float]) -> None:
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        self.stats["distance"] = self.stats.get("distance", 0.0) + math.hypot(dx, dy)
