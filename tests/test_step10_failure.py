from __future__ import annotations

from sim.config import SimConfig
from sim.simulator import Simulator


def test_disable_usv_removes_from_assignment() -> None:
    cfg = SimConfig(
        seed=0,
        strategy="multimetric",
        task_policy="multimetric",
        enable_feedback=False,
        enable_failures=False,
    )
    sim = Simulator(cfg)

    sim._apply_failure_event({"usv_id": "USV-1", "kind": "DISABLED"}, t=0.0)
    disabled = sim.agent_by_id["USV-1"]
    assert disabled.health_state == "DISABLED"

    sim.tasks.generate_tasks(0.0)
    sim._assign_idle_agents(0.0)

    for task in sim.tasks.get_assigned_tasks(task_type="monitor"):
        assert task.assigned_to != "USV-1"
    assert disabled.current_task_id is None

    uav = sim._uavs()[0]
    uav.battery = 0.1 * uav.battery_max
    sim._update_recharge_needs(0.0)
    sim._assign_recharge_tasks(0.0)
    recharge_tasks = sim.recharge_tasks()
    assert recharge_tasks
    for task in recharge_tasks:
        assert task.usv_id != "USV-1"


def test_recharge_rebind_on_usv_disable() -> None:
    cfg = SimConfig(
        seed=0,
        strategy="multimetric",
        task_policy="multimetric",
        enable_feedback=False,
        enable_failures=False,
        num_usv=3,
    )
    sim = Simulator(cfg)

    uav = sim._uavs()[0]
    uav.battery = 0.1 * uav.battery_max
    sim._update_recharge_needs(0.0)
    sim._assign_recharge_tasks(0.0)

    first = sim.tasks.find_recharge_task(uav.agent_id, statuses=("pending", "assigned", "in_progress"))
    assert first is not None
    assert first.usv_id is not None
    old_usv_id = first.usv_id

    sim._apply_failure_event({"usv_id": old_usv_id, "kind": "DISABLED"}, t=5.0)
    sim._update_recharge_needs(5.0)
    sim._assign_recharge_tasks(5.0)

    rebound = sim.tasks.find_recharge_task(uav.agent_id, statuses=("pending", "assigned", "in_progress"))
    assert rebound is not None
    assert rebound.usv_id is not None
    assert rebound.usv_id != old_usv_id
