from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MultiMetricStrategy:
    seed: int
    config: object

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
        _ = (agent, env, t)
        if task is None:
            return None
        return task.target_pos

    def pair_score(self, agent, task, region_info_map, t: float) -> float:
        _ = t
        if task.region_id is None:
            return -1e9

        rx, ry = task.region_id
        need = 0.0
        if 0 <= ry < region_info_map.shape[0] and 0 <= rx < region_info_map.shape[1]:
            need = 1.0 - float(region_info_map[ry, rx])

        prio = float(task.priority)
        dist = agent.distance_to(task.target_pos)
        diag = max(1.0, math.hypot(self.config.map_width, self.config.map_height))
        norm_denom = max(1.0, 0.6 * diag)
        norm_dist = max(0.0, min(1.0, dist / norm_denom))

        risk_cost = self._risk_cost(task)
        if self.config.ablate_risk_term:
            risk_cost = 0.0

        energy_risk = self._energy_risk(agent=agent, dist=dist, norm_dist=norm_dist)
        if self.config.ablate_energy_term:
            energy_risk = 0.0

        soft_bonus = self._soft_partition_bonus(agent=agent, task=task)
        if self.config.ablate_softpart:
            soft_bonus = 0.0

        score = (
            self.config.w_need * need
            + self.config.w_prio * prio
            - self.config.w_dist * norm_dist
            - self.config.w_risk * risk_cost
            - self.config.w_energy * energy_risk
            + self.config.w_softpart * soft_bonus
        )
        return float(score)

    def _risk_cost(self, task) -> float:
        y = float(task.target_pos[1])
        if self.config.nearshore_y_max < y <= self.config.risk_zone_y_max:
            return float(self.config.risk_zone_penalty)
        return 0.0

    def _energy_risk(self, agent, dist: float, norm_dist: float) -> float:
        if agent.agent_type == "UAV":
            battery_frac = float(getattr(agent, "battery_frac", 1.0))
            battery = float(getattr(agent, "battery", 1.0))
            discharge = max(1e-6, float(getattr(agent, "discharge_rate", 1.0)))
            speed = max(1e-6, float(getattr(agent, "max_speed", 1.0)))
            expected_range = max(1.0, (battery / discharge) * speed)
            reach_ratio = dist / expected_range
            risk = max(0.0, reach_ratio - battery_frac)
            low_trigger = float(self.config.uav_low_battery_frac) + 0.1
            if battery_frac < low_trigger:
                risk += max(0.0, norm_dist - 0.12) * 1.5
            return risk

        recharge_pressure = float(getattr(agent, "stats", {}).get("recharge_pressure", 0.0))
        if recharge_pressure <= 0.0:
            return 0.0
        return recharge_pressure * (0.5 + norm_dist)

    def _soft_partition_bonus(self, agent, task) -> float:
        if agent.agent_type != "USV":
            return 0.0

        stats = getattr(agent, "stats", {})
        pref_x = stats.get("preferred_center_x")
        pref_y = stats.get("preferred_center_y")
        if pref_x is None or pref_y is None:
            return 0.0

        dx = float(task.target_pos[0]) - float(pref_x)
        dy = float(task.target_pos[1]) - float(pref_y)
        pref_dist = math.hypot(dx, dy)
        sigma = max(1.0, float(self.config.softpart_sigma))
        bonus = math.exp(-pref_dist / sigma)
        scale = float(stats.get("softpart_scale", 1.0))
        return float(bonus * scale)
