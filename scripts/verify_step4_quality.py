from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.coverage.coverage_grid import CoverageGrid
from sim.environment.environment import Environment2D
from sim.tasks.region_map import RegionMap
from sim.tasks.task_generator import TaskGenerator


def fail(msg: str) -> int:
    print(f"STEP4 QUALITY CHECK FAILED: {msg}")
    return 1


def main() -> int:
    cfg = SimConfig(seed=0)
    env = Environment2D(cfg)
    grid = CoverageGrid(cfg)
    region_map = RegionMap(grid, cfg)
    task_gen = TaskGenerator(region_map, cfg)

    # Run for enough ticks to stress generation behavior.
    total_ticks = 80
    dt = 30.0
    for step in range(total_ticks):
        t = step * dt

        for uav_idx in range(cfg.num_uav):
            region = "offshore" if (step + uav_idx) % 2 == 0 else "risk_zone"
            p = env.sample_point(region)
            grid.observe(p, cfg.uav_sensor_radius, t)
        for usv_idx in range(cfg.num_usv):
            region = "nearshore" if usv_idx < 2 else "risk_zone"
            p = env.sample_point(region)
            grid.observe(p, cfg.usv_sensor_radius, t)

        task_gen.generate_tasks(t)
        pending = task_gen.get_pending_tasks()

        # 1) pending limit check
        if len(pending) > cfg.max_pending_tasks:
            return fail(f"pending overflow: {len(pending)} > {cfg.max_pending_tasks}")

        # 2) no duplicate region in pending
        region_ids = [task.region_id for task in pending]
        if len(region_ids) != len(set(region_ids)):
            return fail("duplicate pending tasks found for same region")

        # 3) priority range check
        for task in pending:
            if not (0.0 <= task.priority <= 1.5):
                return fail(f"priority out of range [0,1.5], got {task.priority}")

        # 4) target_pos in sea bounds check
        for task in pending:
            tx, ty = task.target_pos
            if not (0.0 <= tx <= cfg.map_width and 0.0 <= ty <= cfg.map_height):
                return fail(f"target_pos out of map: {task.target_pos}")

    # Run preview to ensure integration still works.
    cmd = [sys.executable, str(ROOT / "scripts" / "preview_tasks.py"), "--seed", "0"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        return fail(f"preview_tasks.py failed: {err}")

    print("STEP4 QUALITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
