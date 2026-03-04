from __future__ import annotations

from sim.agents.usv import USVAgent
from sim.config import SimConfig
from sim.simulator import Simulator


def test_assignment_single_occupancy() -> None:
    cfg = SimConfig(seed=1, t_end=60.0, sim_dt=5.0)
    sim = Simulator(cfg)
    sim.tasks.generate_tasks(t=0.0)
    sim._assign_idle_agents(t=0.0)  # noqa: SLF001 - intentional for unit test

    assigned = sim.tasks.get_assigned_tasks()
    assigned_task_ids = [task.task_id for task in assigned]
    assigned_agents = [task.assigned_to for task in assigned]
    assert len(assigned_task_ids) == len(set(assigned_task_ids))
    assert len(assigned_agents) == len(set(assigned_agents))


def test_timeout_rollback() -> None:
    cfg = SimConfig(seed=2, t_end=30.0, sim_dt=5.0, task_timeout=1.0)
    sim = Simulator(cfg)
    sim.tasks.generate_tasks(t=0.0)
    sim._assign_idle_agents(t=0.0)  # noqa: SLF001 - intentional for unit test
    assigned_before = sim.tasks.get_assigned_tasks()
    assert len(assigned_before) > 0

    rolled = sim.tasks.update_timeouts(t=2.0, timeout=cfg.task_timeout)
    rolled_set = set(rolled)
    for agent in sim.agents:
        if agent.current_task_id in rolled_set:
            agent.clear_task()

    assert len(rolled) > 0
    for task_id in rolled:
        task = sim.tasks.get_task(task_id)
        assert task is not None
        assert task.status == "pending"
        assert task.assigned_to is None


def test_completion_by_observe() -> None:
    cfg = SimConfig(seed=3, t_end=60.0, sim_dt=5.0)
    sim = Simulator(cfg)
    created = sim.tasks.generate_tasks(t=0.0)
    assert len(created) > 0
    target_task = created[0]
    ok = sim.tasks.assign_task(task_id=target_task.task_id, agent_id=sim.agents[0].agent_id, t=0.0)
    assert ok
    sim.agents[0].set_task(target_task.task_id)
    center = sim.region_map.region_center(*target_task.region_id)
    sim.coverage.observe(position=center, radius=cfg.region_cell_size * 1.2, t=0.0)

    done_before = len([t for t in sim.tasks.all_tasks() if t.status == "done"])
    sim._complete_tasks_by_region_info(t=0.0)  # noqa: SLF001 - intentional for unit test
    done_after = len([t for t in sim.tasks.all_tasks() if t.status == "done"])
    assert done_after > done_before


def test_usv_avoid_obstacle_step() -> None:
    cfg = SimConfig(seed=4)
    sim = Simulator(cfg)
    obstacle = sim.env.obstacles[0]
    x = max(0.0, obstacle.center_x - obstacle.radius - 20.0)
    y = obstacle.center_y
    usv = USVAgent(
        agent_id="USV-test",
        pos=(x, y),
        max_speed=cfg.usv_speed,
        sensor_radius=cfg.usv_sensor_radius,
        comm_radius=cfg.comm_radius,
        turn_rate_deg=cfg.usv_turn_rate_deg,
        heading=0.0,
    )
    target = (min(cfg.map_width, obstacle.center_x + obstacle.radius + 200.0), obstacle.center_y)
    usv.step_toward(
        target_pos=target,
        env=sim.env,
        t=0.0,
        dt=cfg.sim_dt,
        current_effect=cfg.current_effect_usv,
        avoidance_angles_deg=cfg.obstacle_avoidance_angles,
    )
    assert not sim.env.is_in_obstacle(usv.pos[0], usv.pos[1])
