from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FeedbackMonitor:
    recent_metrics: list[dict[str, float]] = field(default_factory=list)
    max_recent: int = 8

    def update(self, sim_state: Any, t: float) -> dict[str, float]:
        info_map = sim_state.coverage.info_map(t)
        pending_count = len(sim_state.tasks.get_pending_tasks(task_type="monitor"))
        assigned_count = len(sim_state.tasks.get_assigned_tasks(task_type="monitor"))
        done_count = len([task for task in sim_state.tasks.all_tasks(task_type="monitor") if task.status == "done"])
        uavs = sim_state._uavs()
        battery_fracs = [float(uav.battery_frac) for uav in uavs] if uavs else [0.0]

        metrics = {
            "t": float(t),
            "mean_info_all": float(np.mean(info_map)) if info_map.size else 0.0,
            "p5_info_all": float(np.percentile(info_map, 5)) if info_map.size else 0.0,
            "min_info_all": float(np.min(info_map)) if info_map.size else 0.0,
            "pending_count": float(pending_count),
            "assigned_count": float(assigned_count),
            "done_count_cum": float(done_count),
            "recharge_count_cum": float(sim_state.transition_counts.get("recharge_done", 0)),
            "uav_battery_min": float(min(battery_fracs)),
            "uav_battery_mean": float(sum(battery_fracs) / len(battery_fracs)) if battery_fracs else 0.0,
            "usv_preference_hit_rate": float(sim_state.usv_preference_hit_rate()),
            "usv_cross_band_ratio": float(sim_state.usv_cross_band_ratio()),
        }
        self.recent_metrics.append(metrics)
        if len(self.recent_metrics) > self.max_recent:
            self.recent_metrics = self.recent_metrics[-self.max_recent :]
        return metrics


class FeedbackController:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.cooldown_until: dict[str, float] = {}
        self.recent_metrics: list[dict[str, float]] = []

    def step(self, metrics: dict[str, float], sim_state: Any) -> list[dict[str, Any]]:
        t = float(metrics["t"])
        self.recent_metrics.append(metrics)
        if len(self.recent_metrics) > 8:
            self.recent_metrics = self.recent_metrics[-8:]

        pending_high = metrics["pending_count"] > float(self.config.fb_pending_high_frac * self.config.max_pending_tasks)
        mean_low = metrics["mean_info_all"] < float(self.config.fb_meaninfo_low)
        p5_low = metrics["p5_info_all"] < float(self.config.fb_p5info_low)
        pending_rising = self._pending_rising(k=3)
        energy_pressure = (
            metrics["uav_battery_mean"] < float(self.config.fb_energy_pressure_batt)
            or metrics["uav_battery_min"] < 0.25
        )

        actions: list[dict[str, Any]] = []

        if pending_high or (mean_low and pending_rising):
            if self._ready("GLOBAL_REASSIGN", t):
                actions.append(
                    {
                        "type": "GLOBAL_REASSIGN",
                        "reason": "pending_high" if pending_high else "coverage_drop_pending_rising",
                        "mode": str(self.config.fb_reassign_mode),
                    }
                )
                self._touch("GLOBAL_REASSIGN", t)

        if pending_high or mean_low or p5_low:
            if self._ready("RELAX_SOFTPART", t):
                actions.append(
                    {
                        "type": "RELAX_SOFTPART",
                        "reason": "pending_or_coverage_low",
                        "scale": 0.2,
                        "duration": float(self.config.fb_relax_duration_sec),
                    }
                )
                self._touch("RELAX_SOFTPART", t)

        if energy_pressure:
            if self._ready("BOOST_RECHARGE_PRIORITY", t):
                actions.append(
                    {
                        "type": "BOOST_RECHARGE_PRIORITY",
                        "reason": "energy_pressure",
                        "mult": 1.5,
                        "duration": float(self.config.fb_recharge_boost_duration_sec),
                    }
                )
                self._touch("BOOST_RECHARGE_PRIORITY", t)

        return actions

    def _pending_rising(self, k: int) -> bool:
        if len(self.recent_metrics) < (k + 1):
            return False
        old = self.recent_metrics[-(k + 1)]["pending_count"]
        new = self.recent_metrics[-1]["pending_count"]
        return new > old + 1e-6

    def _ready(self, action_type: str, t: float) -> bool:
        return t >= self.cooldown_until.get(action_type, -math.inf)

    def _touch(self, action_type: str, t: float) -> None:
        self.cooldown_until[action_type] = t + float(self.config.fb_cooldown_sec)
