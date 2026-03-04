from __future__ import annotations

import random
from dataclasses import dataclass, field

from sim.config import SimConfig
from sim.tasks.region_map import RegionMap


@dataclass
class Task:
    task_id: int
    task_type: str
    region_id: tuple[int, int]
    target_pos: tuple[float, float]
    priority: float
    created_time: float
    status: str = "pending"
    metadata: dict[str, object] = field(default_factory=dict)


class TaskGenerator:
    def __init__(self, region_map: RegionMap, config: SimConfig) -> None:
        self.region_map = region_map
        self.config = config
        self._rng = random.Random(config.seed + 2027)
        self._next_task_id = 1
        self._pending: list[Task] = []
        self._last_generated_time_by_region: dict[tuple[int, int], float] = {}

    def generate_tasks(self, t: float) -> list[Task]:
        created: list[Task] = []
        info_map = self.region_map.region_info_map(t)
        self._refresh_pending_priorities(info_map, t)

        pending = self.get_pending_tasks()
        pending_count = len(pending)
        if pending_count >= self.config.max_pending_tasks:
            return created

        pending_regions = {task.region_id for task in pending}
        new_count = 0
        cooldown = getattr(self.config, "region_task_cooldown", self.config.task_cooldown)
        for ry in range(self.region_map.nry):
            for rx in range(self.region_map.nrx):
                if pending_count >= self.config.max_pending_tasks:
                    return created
                if new_count >= self.config.max_new_tasks_per_tick:
                    return created
                region_id = (rx, ry)
                region_info = float(info_map[ry, rx])
                if region_info >= self.config.task_info_threshold:
                    continue
                if region_id in pending_regions:
                    continue

                last_t = self._last_generated_time_by_region.get(region_id, -float("inf"))
                if t - last_t < cooldown:
                    continue

                target_pos = self.region_map.region_target_with_offset(
                    rx=rx,
                    ry=ry,
                    rng=self._rng,
                    offset_ratio=0.25,
                )
                priority = self._priority(region_id=region_id, region_info=region_info)
                task = Task(
                    task_id=self._next_task_id,
                    task_type="monitor",
                    region_id=region_id,
                    target_pos=target_pos,
                    priority=priority,
                    created_time=t,
                    status="pending",
                    metadata={"region_info": region_info, "region_weight": self._region_weight(region_id)},
                )
                self._next_task_id += 1
                self._pending.append(task)
                self._last_generated_time_by_region[region_id] = t
                pending_regions.add(region_id)
                created.append(task)
                pending_count += 1
                new_count += 1
        return created

    def get_pending_tasks(self) -> list[Task]:
        return [t for t in self._pending if t.status == "pending"]

    def pop_highest_priority_task(self) -> Task | None:
        pending = self.get_pending_tasks()
        if not pending:
            return None
        best = max(pending, key=lambda t: (t.priority, -t.created_time))
        best.status = "assigned"
        return best

    def mark_task_done(self, task_id: int) -> bool:
        for task in self._pending:
            if task.task_id == task_id:
                task.status = "done"
                return True
        return False

    def _refresh_pending_priorities(self, info_map, t: float) -> None:
        for task in self._pending:
            if task.status != "pending":
                continue
            rx, ry = task.region_id
            region_info = float(info_map[ry, rx])
            task.priority = self._priority(region_id=task.region_id, region_info=region_info)
            task.metadata["region_info"] = region_info
            task.metadata["region_weight"] = self._region_weight(task.region_id)
            task.metadata["updated_time"] = t

    def _region_weight(self, region_id: tuple[int, int]) -> float:
        rx, ry = region_id
        _, y = self.region_map.region_center(rx, ry)
        if y <= self.config.nearshore_y_max:
            return self.config.nearshore_task_weight
        if y <= self.config.risk_zone_y_max:
            return self.config.risk_task_weight
        return self.config.offshore_task_weight

    def _priority(self, region_id: tuple[int, int], region_info: float) -> float:
        weight = self._region_weight(region_id)
        raw = weight * (1.0 - region_info)
        return max(0.0, min(1.5, raw))
