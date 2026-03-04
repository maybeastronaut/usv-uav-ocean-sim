from __future__ import annotations

from sim.config import SimConfig
from sim.simulator import Simulator


def test_battery_discharge() -> None:
    cfg = SimConfig(seed=10, num_uav=1, num_usv=0)
    sim = Simulator(cfg)
    uav = next(agent for agent in sim.agents if agent.agent_type == "UAV")
    start = uav.battery
    uav.discharge(dt=20.0, moving=True)
    assert uav.battery < start
    assert uav.battery >= 0.0


def test_recharge_completion() -> None:
    cfg = SimConfig(seed=11, num_uav=1, num_usv=1, sim_dt=5.0, t_end=60.0)
    sim = Simulator(cfg)
    uav = next(agent for agent in sim.agents if agent.agent_type == "UAV")
    usv = next(agent for agent in sim.agents if agent.agent_type == "USV")

    uav.pos = (2000.0, 2000.0)
    usv.pos = (2020.0, 2010.0)
    uav.battery = 0.15 * uav.battery_max

    required = 0.2 * uav.battery_max
    task, _ = sim.tasks.create_recharge_task(
        uav_id=uav.agent_id,
        usv_id=usv.agent_id,
        rendezvous_pos=usv.pos,
        t=0.0,
        required_energy=required,
        priority=cfg.recharge_task_priority,
    )
    task.status = "assigned"
    task.assigned_to = f"{uav.agent_id}|{usv.agent_id}"
    uav.set_task(task.task_id)
    usv.set_task(task.task_id)

    before = uav.battery
    sim._step_agents(t=0.0, dt=cfg.sim_dt)  # noqa: SLF001 - intentional for unit test
    assert uav.battery >= before

    for tick in range(1, 30):
        sim._step_agents(t=tick * cfg.sim_dt, dt=cfg.sim_dt)  # noqa: SLF001 - intentional for unit test
        task_state = sim.tasks.get_task(task.task_id)
        if task_state is not None and task_state.status == "done":
            break

    done_task = sim.tasks.get_task(task.task_id)
    assert done_task is not None
    assert done_task.status == "done"
    assert uav.battery >= required
