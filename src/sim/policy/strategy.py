from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BaseStrategy:
    seed: int

    def pair_score(self, agent, task, region_info_map, t: float) -> float:
        _ = (agent, task, region_info_map, t)
        return -1e9

    def select_task(self, agent, tasks, region_info_map, t: float) -> int | None:
        if not tasks:
            return None
        best = max(
            tasks,
            key=lambda task: (
                self.pair_score(agent, task, region_info_map, t),
                -agent.distance_to(task.target_pos),
                -task.task_id,
            ),
        )
        return best.task_id

    def select_goal(self, agent, task, env, t: float) -> tuple[float, float] | None:
        _ = (agent, task, env, t)
        return None


class RandomCruiseStrategy(BaseStrategy):
    def __init__(self, seed: int) -> None:
        super().__init__(seed=seed)
        self._rng = random.Random(seed + 5001)

    def pair_score(self, agent, task, region_info_map, t: float) -> float:
        _ = region_info_map
        # Deterministic pseudo-random score by seed+agent+task+time bucket.
        bucket = int(round(t))
        key = self.seed * 1000003 + _agent_num(agent.agent_id) * 917 + int(task.task_id) * 53 + bucket * 7
        r = random.Random(key)
        return r.random()

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
    def pair_score(self, agent, task, region_info_map, t: float) -> float:
        _ = (region_info_map, t)
        return -agent.distance_to(task.target_pos)

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

    def pair_score(self, agent, task, region_info_map, t: float) -> float:
        _ = (region_info_map, t)
        dist = agent.distance_to(task.target_pos)
        max_dist = max(1.0, 15000.0)
        norm_dist = min(1.0, dist / max_dist)

        alpha = self.BASE_ALPHA
        beta = self.BASE_BETA
        battery_frac = float(getattr(agent, "battery_frac", 1.0))
        if agent.agent_type == "UAV" and battery_frac < self.LOW_BATTERY_TRIGGER:
            alpha = 1.6
            beta = 2.0

        recharge_pressure = float(getattr(agent, "stats", {}).get("recharge_pressure", 0.0))
        energy_risk = self._energy_risk(agent=agent, dist=dist, norm_dist=norm_dist)
        if agent.agent_type == "USV":
            energy_risk += recharge_pressure * (0.6 + 0.8 * norm_dist)

        switch_cost = self._switch_cost(agent=agent, task=task)
        return self.PRIORITY_WEIGHT * float(task.priority) - alpha * norm_dist - beta * energy_risk - switch_cost

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
        expected_range = max(1.0, (battery / discharge) * speed)
        reach_ratio = dist / expected_range

        risk = max(0.0, reach_ratio - battery_frac)
        if battery_frac < self.LOW_BATTERY_TRIGGER and reach_ratio > 0.4:
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


def create_strategy(name: str, seed: int, config=None):
    key = (name or "").strip().lower()
    if key == "random":
        return RandomCruiseStrategy(seed=seed)
    if key == "nearest":
        return NearestTaskStrategy(seed=seed)
    if key == "priority":
        return PriorityTaskStrategy(seed=seed)
    if key == "multimetric":
        from sim.policy.multimetric import MultiMetricStrategy

        return MultiMetricStrategy(seed=seed, config=config)
    raise ValueError(f"unknown strategy: {name}")


def _agent_num(agent_id: str) -> int:
    digits = "".join(ch for ch in str(agent_id) if ch.isdigit())
    if digits:
        return int(digits)
    return sum(ord(ch) for ch in str(agent_id))
