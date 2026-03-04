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
    PRIORITY_SOFT_TIE: float = 0.20

    def select_task(self, agent, tasks, region_info_map, t: float) -> int | None:
        _ = (region_info_map, t)
        if not tasks:
            return None
        max_priority = max(task.priority for task in tasks)
        candidates = [task for task in tasks if task.priority >= max_priority - self.PRIORITY_SOFT_TIE]
        # Within a soft-tied priority band, prefer nearer tasks for better throughput.
        best = min(
            candidates,
            key=lambda task: (agent.distance_to(task.target_pos), -task.priority, task.task_id),
        )
        return best.task_id

    def select_goal(self, agent, task, env, t: float) -> tuple[float, float] | None:
        _ = (agent, env, t)
        if task is None:
            return None
        return task.target_pos


def create_strategy(name: str, seed: int):
    key = (name or "").strip().lower()
    if key == "random":
        return RandomCruiseStrategy(seed=seed)
    if key == "nearest":
        return NearestTaskStrategy(seed=seed)
    if key == "priority":
        return PriorityTaskStrategy(seed=seed)
    raise ValueError(f"unknown strategy: {name}")
