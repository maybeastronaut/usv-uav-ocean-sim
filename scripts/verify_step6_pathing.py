from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.agents.usv import USVAgent
from sim.config import SimConfig
from sim.pathing.path_planner import USVPlanner, segment_hits_any_obstacle
from sim.simulator import Simulator


def fail(msg: str) -> int:
    print(f"STEP6 FAIL: {msg}")
    return 1


def run_metrics(strategy: str) -> tuple[float, float, int, list[float]]:
    cfg = SimConfig(seed=0, strategy=strategy, t_end=600.0, sim_dt=5.0)
    sim = Simulator(cfg)
    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt)
    if not result.history:
        raise RuntimeError(f"empty sim history for strategy={strategy}")
    mean_start = result.history[0]["mean_info_all"]
    mean_end = result.history[-1]["mean_info_all"]
    done_count = int(result.history[-1]["done_count"])
    pending_series = [row["pending_count"] for row in result.history]
    return mean_start, mean_end, done_count, pending_series


def main() -> int:
    cfg = SimConfig(seed=0, strategy="priority", t_end=600.0, sim_dt=5.0)
    sim = Simulator(cfg)

    # 1) USV path planner should avoid obstacle when direct segment intersects
    planner = USVPlanner(safe_margin=cfg.safe_margin)
    usv = next(a for a in sim.agents if a.agent_type == "USV")
    obstacle = sim.env.obstacles[0]
    start = (max(0.0, obstacle.center_x - obstacle.radius - 200.0), obstacle.center_y)
    goal = (min(cfg.map_width, obstacle.center_x + obstacle.radius + 300.0), obstacle.center_y)
    usv.pos = start
    waypoints = planner.plan(usv, goal, sim.env, t=0.0)
    full_path = [start] + waypoints
    for i in range(len(full_path) - 1):
        if segment_hits_any_obstacle(full_path[i], full_path[i + 1], sim.env.obstacles, margin=cfg.safe_margin):
            return fail("USV waypoint plan still intersects obstacle")

    # 2~4) strategy effectiveness checks
    nearest_start, nearest_end, nearest_done, _ = run_metrics("nearest")
    priority_start, priority_end, priority_done, pending_series = run_metrics("priority")

    if not (priority_end > priority_start and priority_end >= 0.09):
        return fail(
            "priority mean_info_all too low: "
            f"start={priority_start:.4f}, end={priority_end:.4f}, expected_end>=0.09"
        )

    min_done_expected = int(math.ceil(0.5 * nearest_done))
    if priority_done < min_done_expected:
        return fail(
            "priority done_count too low: "
            f"priority={priority_done}, nearest={nearest_done}, expected>={min_done_expected}"
        )

    if max(pending_series) >= cfg.max_pending_tasks and pending_series[-1] >= max(pending_series):
        return fail("pending_count saturated at max and did not recover")

    print(
        f"VERIFY METRICS nearest(mean_end={nearest_end:.4f}, done={nearest_done}) "
        f"priority(mean_end={priority_end:.4f}, done={priority_done})"
    )
    print("STEP6 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
