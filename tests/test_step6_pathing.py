from __future__ import annotations

from sim.config import SimConfig
from sim.pathing.path_planner import USVPlanner, segment_circle_intersect, segment_hits_any_obstacle
from sim.simulator import Simulator


def test_segment_circle_intersect() -> None:
    c = (0.0, 0.0)
    r = 1.0
    assert segment_circle_intersect((-2.0, 0.0), (2.0, 0.0), c, r)
    assert not segment_circle_intersect((2.0, 2.0), (3.0, 3.0), c, r)


def test_usv_plan_avoids_obstacle() -> None:
    cfg = SimConfig(seed=11, safe_margin=120.0)
    sim = Simulator(cfg)
    usv = next(a for a in sim.agents if a.agent_type == "USV")
    obstacle = sim.env.obstacles[0]
    start = (max(0.0, obstacle.center_x - obstacle.radius - 250.0), obstacle.center_y)
    goal = (min(cfg.map_width, obstacle.center_x + obstacle.radius + 250.0), obstacle.center_y)
    usv.pos = start
    planner = USVPlanner(safe_margin=cfg.safe_margin)
    waypoints = planner.plan(usv, goal, sim.env, t=0.0)
    path = [start] + waypoints
    for i in range(len(path) - 1):
        assert not segment_hits_any_obstacle(path[i], path[i + 1], sim.env.obstacles, margin=cfg.safe_margin)
