from __future__ import annotations

import numpy as np

from sim.agents.uav import UAVAgent
from sim.agents.usv import USVAgent
from sim.config import SimConfig
from sim.policy.multimetric import MultiMetricStrategy
from sim.tasks.task_generator import Task


def _make_task(task_id: int, pos: tuple[float, float], priority: float = 1.0) -> Task:
    return Task(
        task_id=task_id,
        task_type="monitor",
        region_id=(0, 0),
        target_pos=pos,
        priority=priority,
        created_time=0.0,
    )


def test_soft_partition_bonus() -> None:
    cfg = SimConfig(task_policy="multimetric", strategy="multimetric")
    policy = MultiMetricStrategy(seed=0, config=cfg)

    usv_left = USVAgent(
        agent_id="USV-1",
        pos=(1000.0, 4000.0),
        max_speed=cfg.usv_speed,
        sensor_radius=cfg.usv_sensor_radius,
        comm_radius=cfg.comm_radius,
        turn_rate_deg=cfg.usv_turn_rate_deg,
        charge_rate=cfg.usv_charge_rate,
    )
    usv_right = USVAgent(
        agent_id="USV-2",
        pos=(1000.0, 4000.0),
        max_speed=cfg.usv_speed,
        sensor_radius=cfg.usv_sensor_radius,
        comm_radius=cfg.comm_radius,
        turn_rate_deg=cfg.usv_turn_rate_deg,
        charge_rate=cfg.usv_charge_rate,
    )

    usv_left.stats["preferred_center_x"] = 1200.0
    usv_left.stats["preferred_center_y"] = 5000.0
    usv_left.stats["softpart_scale"] = 1.0
    usv_right.stats["preferred_center_x"] = 9000.0
    usv_right.stats["preferred_center_y"] = 5000.0
    usv_right.stats["softpart_scale"] = 1.0

    task = _make_task(task_id=1, pos=(1500.0, 5000.0), priority=1.0)
    info_map = np.array([[0.3]], dtype=float)

    score_left = policy.pair_score(usv_left, task, info_map, t=0.0)
    score_right = policy.pair_score(usv_right, task, info_map, t=0.0)
    assert score_left > score_right


def test_multimetric_energy_penalty() -> None:
    cfg = SimConfig(task_policy="multimetric", strategy="multimetric")
    policy = MultiMetricStrategy(seed=0, config=cfg)

    uav = UAVAgent(
        agent_id="UAV-1",
        pos=(2000.0, 2000.0),
        max_speed=cfg.uav_speed,
        sensor_radius=cfg.uav_sensor_radius,
        comm_radius=cfg.comm_radius,
        energy=cfg.uav_battery_max,
        low_energy_threshold=cfg.uav_low_battery_frac,
        battery_max=cfg.uav_battery_max,
        discharge_rate=cfg.uav_discharge_rate,
        critical_battery_threshold=cfg.uav_critical_battery_frac,
    )

    near_task = _make_task(task_id=1, pos=(2200.0, 2200.0), priority=1.0)
    far_task = _make_task(task_id=2, pos=(9000.0, 9000.0), priority=1.0)
    info_map = np.array([[0.2]], dtype=float)

    uav.battery = 0.3 * uav.battery_max
    low_near = policy.pair_score(uav, near_task, info_map, t=0.0)
    low_far = policy.pair_score(uav, far_task, info_map, t=0.0)

    uav.battery = 0.9 * uav.battery_max
    high_far = policy.pair_score(uav, far_task, info_map, t=0.0)

    assert low_near > low_far
    assert high_far > low_far
