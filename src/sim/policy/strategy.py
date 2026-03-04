from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BaseStrategy:
    seed: int

    def select_task(self, agent, tasks, region_info_map, t: float) -> int | None:
        _ = (agent, tasks, region_info_map, t)
        return None

    def select_goal(self, agent, task, env, t: float) -> tuple[float, float] | None:
        _ = (agent, task, env, t)
        return None


class RandomCruiseStrategy(BaseStrategy):
    def __init__(self, seed: int) -> None:
        super().__init__(seed=seed)
        self._rng = random.Random(seed + 5001)

    def select_task(self, agent, tasks, region_info_map, t: float) -> int | None:
        _ = (agent, region_info_map, t)
        if not tasks:
            return None
        ordered = sorted(tasks, key=lambda task: task.task_id)
        idx = self._rng.randrange(len(ordered))
        return ordered[idx].task_id

    def select_goal(self, agent, task, env, t: float) -> tuple[float, float] | None:
        _ = (agent, t)
        if task is not None:
            return task.target_pos
        if agent.cruise_goal is not None:
            if agent.distance_to(agent.cruise_goal) > 80.0:
                return agent.cruise_goal
        for _ in range(300):
            x = self._rng.uniform(0.0, env.map_width)
            y = self._rng.uniform(0.0, env.map_height)
            if env.is_in_obstacle(x, y):
                continue
            agent.cruise_goal = (x, y)
            return agent.cruise_goal
        return env.base_position


class NearestTaskStrategy(BaseStrategy):
    def select_task(self, agent, tasks, region_info_map, t: float) -> int | None:
        _ = (region_info_map, t)
        if not tasks:
            return None
        best = min(
            tasks,
            key=lambda task: (agent.distance_to(task.target_pos), task.task_id, agent.agent_id),
        )
        return best.task_id

    def select_goal(self, agent, task, env, t: float) -> tuple[float, float] | None:
        _ = (agent, env, t)
        if task is None:
            return None
        return task.target_pos


class PriorityTaskStrategy(BaseStrategy):
    PRIORITY_WEIGHT: float = 0.65
    BASE_ALPHA: float = 1.35
    BASE_BETA: float = 1.30
    LOW_BATTERY_TRIGGER: float = 0.45
    SWITCH_COST: float = 0.02

    def select_task(self, agent, tasks, region_info_map, t: float) -> int | None:
        _ = (region_info_map, t)
        if not tasks:
            return None
        candidates = list(tasks)

        dist_map = {task.task_id: agent.distance_to(task.target_pos) for task in candidates}
        max_dist = max(dist_map.values()) if dist_map else 1.0
        max_dist = max(max_dist, 1.0)

        alpha = self.BASE_ALPHA
        beta = self.BASE_BETA
        battery_frac = float(getattr(agent, "battery_frac", 1.0))
        if agent.agent_type == "UAV" and battery_frac < self.LOW_BATTERY_TRIGGER:
            alpha = 1.6
            beta = 2.0

        recharge_pressure = float(getattr(agent, "stats", {}).get("recharge_pressure", 0.0))

        def score(task) -> tuple[float, float, float, int]:
            dist = dist_map[task.task_id]
            norm_dist = min(1.0, dist / max_dist)
            energy_risk = self._energy_risk(agent=agent, dist=dist, norm_dist=norm_dist)
            if agent.agent_type == "USV":
                # If recharge pressure rises, suppress far monitor choices for USV.
                energy_risk += recharge_pressure * (0.6 + 0.8 * norm_dist)

            switch_cost = self._switch_cost(agent=agent, task=task)
            task_score = self.PRIORITY_WEIGHT * float(task.priority) - alpha * norm_dist - beta * energy_risk - switch_cost
            # max() with tuple: higher score first, then shorter distance, then smaller task_id.
            return (task_score, -dist, -task.task_id, task.task_id)

        best = max(candidates, key=score)
        return best.task_id

    def select_goal(self, agent, task, env, t: float) -> tuple[float, float] | None:
        _ = (agent, env, t)
        if task is None:
            return None
        return task.target_pos

    def _energy_risk(self, agent, dist: float, norm_dist: float) -> float:
        if agent.agent_type != "UAV":
            return 0.25 * norm_dist

        battery = float(getattr(agent, "battery", 0.0))
        discharge = max(1e-6, float(getattr(agent, "discharge_rate", 1.0)))
        speed = max(1e-6, float(getattr(agent, "max_speed", 1.0)))
        battery_frac = float(getattr(agent, "battery_frac", 1.0))
        expected_range = (battery / discharge) * speed
        expected_range = max(1.0, expected_range)
        reach_ratio = dist / expected_range

        risk = max(0.0, reach_ratio - battery_frac)
        if battery_frac < self.LOW_BATTERY_TRIGGER and reach_ratio > 0.4:
            # Extra penalty for low battery UAV picking far targets.
            risk += 0.8 * (reach_ratio - 0.4)
        return risk

    def _switch_cost(self, agent, task) -> float:
        if task.region_id is None:
            return 0.0
        stats = getattr(agent, "stats", {})
        last_rx = stats.get("last_monitor_rx")
        last_ry = stats.get("last_monitor_ry")
        if last_rx is None or last_ry is None:
            return 0.0
        if (int(last_rx), int(last_ry)) == task.region_id:
            return 0.0
        return self.SWITCH_COST


def create_strategy(name: str, seed: int):
    key = (name or "").strip().lower()
    if key == "random":
        return RandomCruiseStrategy(seed=seed)
    if key == "nearest":
        return NearestTaskStrategy(seed=seed)
    if key == "priority":
        return PriorityTaskStrategy(seed=seed)
    raise ValueError(f"unknown strategy: {name}")
