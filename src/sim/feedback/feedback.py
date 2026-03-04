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
        usvs = sim_state._usvs()
        battery_fracs = [float(uav.battery_frac) for uav in uavs] if uavs else [0.0]
        num_disabled = sum(1 for usv in usvs if usv.health_state == "DISABLED")
        num_damaged = sum(1 for usv in usvs if usv.health_state == "DAMAGED")
        num_can_charge = sum(1 for usv in usvs if usv.can_charge())

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
            "num_usv_disabled": float(num_disabled),
            "num_usv_damaged": float(num_damaged),
            "num_usv_total": float(len(usvs)),
            "num_usv_can_charge": float(num_can_charge),
        }
        self.recent_metrics.append(metrics)
        if len(self.recent_metrics) > self.max_recent:
            self.recent_metrics = self.recent_metrics[-self.max_recent :]
        return metrics


class FeedbackController:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.cooldown_until: dict[str, float] = {}
        self.reason_cooldown_until: dict[tuple[str, str], float] = {}
        self.recent_metrics: list[dict[str, float]] = []
        self.prev_num_usv_disabled = 0.0

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
        disabled_now = float(metrics.get("num_usv_disabled", 0.0))
        disabled_increase = disabled_now > self.prev_num_usv_disabled + 1e-9

        actions: list[dict[str, Any]] = []

        if bool(getattr(self.config, "enable_robust_response", True)) and disabled_increase:
            if self._ready("GLOBAL_REASSIGN", t) and self._reason_ready("GLOBAL_REASSIGN", "usv_failure", t):
                actions.append(
                    {
                        "type": "GLOBAL_REASSIGN",
                        "reason": "usv_failure",
                        "mode": str(self.config.fb_reassign_mode),
                    }
                )
                self._touch("GLOBAL_REASSIGN", t)
                self._touch_reason("GLOBAL_REASSIGN", "usv_failure", t)

            if (
                self._ready("RELAX_SOFTPART", t)
                and self._reason_ready("RELAX_SOFTPART", "failure_rebalance", t)
                and not self._relax_active(sim_state, t)
            ):
                actions.append(
                    {
                        "type": "RELAX_SOFTPART",
                        "reason": "failure_rebalance",
                        "scale": 0.2,
                        "duration": float(self.config.fb_relax_duration_sec) * 2.0,
                    }
                )
                self._touch("RELAX_SOFTPART", t)
                self._touch_reason("RELAX_SOFTPART", "failure_rebalance", t)

            charge_capacity_drop = float(metrics.get("num_usv_can_charge", 0.0)) < float(metrics.get("num_usv_total", 0.0))
            if (
                charge_capacity_drop
                and energy_pressure
                and self._ready("BOOST_RECHARGE_PRIORITY", t)
                and self._reason_ready("BOOST_RECHARGE_PRIORITY", "charging_capacity_drop", t)
                and not self._recharge_boost_active(sim_state, t)
            ):
                actions.append(
                    {
                        "type": "BOOST_RECHARGE_PRIORITY",
                        "reason": "charging_capacity_drop",
                        "mult": 1.5,
                        "duration": float(self.config.fb_recharge_boost_duration_sec),
                    }
                )
                self._touch("BOOST_RECHARGE_PRIORITY", t)
                self._touch_reason("BOOST_RECHARGE_PRIORITY", "charging_capacity_drop", t)

        if pending_high or (mean_low and pending_rising):
            reason = "pending_high" if pending_high else "coverage_drop_pending_rising"
            if self._ready("GLOBAL_REASSIGN", t) and self._reason_ready("GLOBAL_REASSIGN", reason, t):
                actions.append(
                    {
                        "type": "GLOBAL_REASSIGN",
                        "reason": reason,
                        "mode": str(self.config.fb_reassign_mode),
                    }
                )
                self._touch("GLOBAL_REASSIGN", t)
                self._touch_reason("GLOBAL_REASSIGN", reason, t)

        if pending_high or mean_low or p5_low:
            if (
                self._ready("RELAX_SOFTPART", t)
                and self._reason_ready("RELAX_SOFTPART", "pending_or_coverage_low", t)
                and not self._relax_active(sim_state, t)
            ):
                actions.append(
                    {
                        "type": "RELAX_SOFTPART",
                        "reason": "pending_or_coverage_low",
                        "scale": 0.2,
                        "duration": float(self.config.fb_relax_duration_sec),
                    }
                )
                self._touch("RELAX_SOFTPART", t)
                self._touch_reason("RELAX_SOFTPART", "pending_or_coverage_low", t)

        if energy_pressure:
            if (
                self._ready("BOOST_RECHARGE_PRIORITY", t)
                and self._reason_ready("BOOST_RECHARGE_PRIORITY", "energy_pressure", t)
                and not self._recharge_boost_active(sim_state, t)
            ):
                actions.append(
                    {
                        "type": "BOOST_RECHARGE_PRIORITY",
                        "reason": "energy_pressure",
                        "mult": 1.5,
                        "duration": float(self.config.fb_recharge_boost_duration_sec),
                    }
                )
                self._touch("BOOST_RECHARGE_PRIORITY", t)
                self._touch_reason("BOOST_RECHARGE_PRIORITY", "energy_pressure", t)

        self.prev_num_usv_disabled = disabled_now
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
        self.cooldown_until[action_type] = t + self._cooldown_for(action_type)

    def _reason_ready(self, action_type: str, reason: str, t: float) -> bool:
        return t >= self.reason_cooldown_until.get((action_type, reason), -math.inf)

    def _touch_reason(self, action_type: str, reason: str, t: float) -> None:
        self.reason_cooldown_until[(action_type, reason)] = t + self._cooldown_for(action_type)

    def _cooldown_for(self, action_type: str) -> float:
        if action_type == "RELAX_SOFTPART":
            return float(getattr(self.config, "fb_cooldown_relax", self.config.fb_cooldown_sec))
        if action_type == "GLOBAL_REASSIGN":
            return float(getattr(self.config, "fb_cooldown_reassign", self.config.fb_cooldown_sec))
        if action_type == "BOOST_RECHARGE_PRIORITY":
            return float(getattr(self.config, "fb_cooldown_recharge_boost", self.config.fb_cooldown_sec))
        return float(self.config.fb_cooldown_sec)

    def _relax_active(self, sim_state: Any, t: float) -> bool:
        if sim_state is None:
            return False
        checker = getattr(sim_state, "_feedback_relax_active", None)
        if callable(checker):
            return bool(checker(t))
        return False

    def _recharge_boost_active(self, sim_state: Any, t: float) -> bool:
        if sim_state is None:
            return False
        checker = getattr(sim_state, "_feedback_recharge_boost_active", None)
        if callable(checker):
            return bool(checker(t))
        return False
