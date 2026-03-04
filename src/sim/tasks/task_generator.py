from __future__ import annotations

import random
from dataclasses import dataclass, field

from sim.config import SimConfig
from sim.tasks.region_map import RegionMap


@dataclass
class Task:
    task_id: int
    task_type: str
    region_id: tuple[int, int] | None
    target_pos: tuple[float, float]
    priority: float
    created_time: float
    status: str = "pending"
    assigned_to: str | None = None
    assigned_time: float | None = None
    last_update_time: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
    uav_id: str | None = None
    usv_id: str | None = None
    rendezvous_pos: tuple[float, float] | None = None
    required_energy: float | None = None


class TaskGenerator:
    def __init__(self, region_map: RegionMap, config: SimConfig) -> None:
        self.region_map = region_map
        self.config = config
        self._rng = random.Random(config.seed + 2027)
        self._next_task_id = 1
        self._pending: list[Task] = []
        self._last_generated_time_by_region: dict[tuple[int, int], float] = {}

    def generate_tasks(self, t: float) -> list[Task]:
        """Generate monitor tasks from low-info regions."""
        created: list[Task] = []
        info_map = self.region_map.region_info_map(t)
        self._refresh_pending_priorities(info_map, t)

        pending_monitor = self.get_pending_tasks(task_type="monitor")
        pending_count = len(pending_monitor)
        if pending_count >= self.config.max_pending_tasks:
            return created

        pending_regions = {task.region_id for task in pending_monitor if task.region_id is not None}
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
                    assigned_to=None,
                    assigned_time=None,
                    last_update_time=t,
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

    def create_recharge_task(
        self,
        *,
        uav_id: str,
        usv_id: str | None,
        rendezvous_pos: tuple[float, float],
        t: float,
        required_energy: float,
        priority: float | None = None,
    ) -> tuple[Task, bool]:
        existing = self.find_recharge_task(uav_id, statuses=("pending", "assigned", "in_progress"))
        if existing is not None:
            existing.usv_id = usv_id
            existing.rendezvous_pos = rendezvous_pos
            existing.target_pos = rendezvous_pos
            existing.required_energy = required_energy
            existing.priority = float(priority if priority is not None else existing.priority)
            existing.last_update_time = t
            return existing, False

        task = Task(
            task_id=self._next_task_id,
            task_type="recharge",
            region_id=None,
            target_pos=rendezvous_pos,
            priority=float(priority if priority is not None else self.config.recharge_task_priority),
            created_time=t,
            status="pending",
            assigned_to=None,
            assigned_time=None,
            last_update_time=t,
            metadata={},
            uav_id=uav_id,
            usv_id=usv_id,
            rendezvous_pos=rendezvous_pos,
            required_energy=required_energy,
        )
        self._next_task_id += 1
        self._pending.append(task)
        return task, True

    def find_recharge_task(
        self,
        uav_id: str,
        statuses: tuple[str, ...] = ("pending", "assigned", "in_progress"),
    ) -> Task | None:
        for task in self._pending:
            if task.task_type != "recharge":
                continue
            if task.uav_id != uav_id:
                continue
            if task.status in statuses:
                return task
        return None

    def get_pending_tasks(self, task_type: str | None = None) -> list[Task]:
        return [task for task in self._pending if task.status == "pending" and (task_type is None or task.task_type == task_type)]

    def get_assigned_tasks(self, task_type: str | None = None) -> list[Task]:
        return [task for task in self._pending if task.status == "assigned" and (task_type is None or task.task_type == task_type)]

    def get_in_progress_tasks(self, task_type: str | None = None) -> list[Task]:
        return [
            task
            for task in self._pending
            if task.status == "in_progress" and (task_type is None or task.task_type == task_type)
        ]

    def all_tasks(self, task_type: str | None = None) -> list[Task]:
        if task_type is None:
            return list(self._pending)
        return [task for task in self._pending if task.task_type == task_type]

    def get_task(self, task_id: int) -> Task | None:
        for task in self._pending:
            if task.task_id == task_id:
                return task
        return None

    def pop_highest_priority_task(self) -> Task | None:
        pending = self.get_pending_tasks(task_type="monitor")
        if not pending:
            return None
        best = max(pending, key=lambda t: (t.priority, -t.created_time))
        best.status = "assigned"
        best.last_update_time = best.created_time
        return best

    def assign_task(self, task_id: int, agent_id: str, t: float) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return False
        if task.status != "pending":
            return False
        task.status = "assigned"
        task.assigned_to = agent_id
        task.assigned_time = t
        task.last_update_time = t
        return True

    def set_task_status(self, task_id: int, status: str, t: float, assigned_to: str | None = None) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return False
        task.status = status
        task.assigned_to = assigned_to
        task.last_update_time = t
        if status in ("assigned", "in_progress") and task.assigned_time is None:
            task.assigned_time = t
        return True

    def release_task(self, task_id: int, t: float) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return False
        if task.status not in ("assigned", "in_progress"):
            return False
        task.status = "pending"
        task.assigned_to = None
        task.assigned_time = None
        task.last_update_time = t
        return True

    def cancel_task(self, task_id: int, t: float) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return False
        task.status = "cancelled"
        task.assigned_to = None
        task.assigned_time = None
        task.last_update_time = t
        return True

    def mark_task_done(self, task_id: int, t: float | None = None) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return False
        task.status = "done"
        task.assigned_to = None
        task.assigned_time = None
        if t is not None:
            task.last_update_time = t
        return True

    def update_timeouts(self, t: float, timeout: float) -> list[int]:
        rolled_back: list[int] = []
        for task in self._pending:
            if task.status != "assigned":
                continue
            if task.assigned_time is None:
                continue
            if t - task.assigned_time > timeout:
                task.status = "pending"
                task.assigned_to = None
                task.assigned_time = None
                task.last_update_time = t
                rolled_back.append(task.task_id)
        return rolled_back

    def complete_tasks_for_region(self, region_id: tuple[int, int], t: float) -> list[int]:
        done_ids: list[int] = []
        for task in self._pending:
            if task.task_type != "monitor":
                continue
            if task.region_id != region_id:
                continue
            if task.status in ("pending", "assigned", "in_progress"):
                self.mark_task_done(task.task_id, t=t)
                done_ids.append(task.task_id)
        return done_ids

    def _refresh_pending_priorities(self, info_map, t: float) -> None:
        for task in self._pending:
            if task.status != "pending":
                continue
            if task.task_type != "monitor" or task.region_id is None:
                continue
            rx, ry = task.region_id
            region_info = float(info_map[ry, rx])
            task.priority = self._priority(region_id=task.region_id, region_info=region_info)
            task.metadata["region_info"] = region_info
            task.metadata["region_weight"] = self._region_weight(task.region_id)
            task.metadata["updated_time"] = t
            task.last_update_time = t

    def refresh_pending_priorities(self, info_map, t: float) -> None:
        self._refresh_pending_priorities(info_map, t)

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
