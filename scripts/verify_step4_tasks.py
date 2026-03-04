from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.coverage.coverage_grid import CoverageGrid
from sim.tasks.region_map import RegionMap
from sim.tasks.task_generator import TaskGenerator


def fail(msg: str) -> int:
    print(f"STEP4 FAIL: {msg}")
    return 1


def main() -> int:
    cfg = SimConfig(seed=0, region_cell_size=1000.0)
    grid = CoverageGrid(cfg)
    region_map = RegionMap(grid, cfg)
    task_gen = TaskGenerator(region_map, cfg)

    # 1) region dimension check
    exp_nrx = int(math.ceil(cfg.map_width / cfg.region_cell_size))
    exp_nry = int(math.ceil(cfg.map_height / cfg.region_cell_size))
    if region_map.nrx != exp_nrx or region_map.nry != exp_nry:
        return fail(f"region dims mismatch: got=({region_map.nrx},{region_map.nry}), expected=({exp_nrx},{exp_nry})")

    # 2) region info range check
    info = region_map.region_info_map(t=0.0)
    if info.min() < -1e-9 or info.max() > 1.0 + 1e-9:
        return fail(f"region_info_map values out of [0,1]: min={info.min()}, max={info.max()}")

    # 3) poor coverage should generate tasks
    created0 = task_gen.generate_tasks(t=0.0)
    if len(created0) <= 0:
        return fail("expected tasks > 0 under poor initial coverage")

    # 4) observe one region and verify priority down or no new generation for this region
    region_id = created0[0].region_id
    old_priority = created0[0].priority
    rx, ry = region_id
    center = region_map.region_center(rx, ry)
    grid.observe(position=center, radius=cfg.region_cell_size * 0.9, t=1.0)
    task_gen.generate_tasks(t=cfg.task_cooldown + 2.0)

    same_region_pending = [t for t in task_gen.get_pending_tasks() if t.region_id == region_id]
    if len(same_region_pending) > 1:
        return fail("dedup failed: duplicate pending tasks for same region")
    if same_region_pending:
        if same_region_pending[0].priority > old_priority + 1e-9:
            return fail(
                "priority did not decrease after observe: "
                f"old={old_priority}, new={same_region_pending[0].priority}"
            )

    # 5) cooldown + dedup check
    cfg2 = SimConfig(seed=0, region_cell_size=1000.0, task_cooldown=600.0)
    grid2 = CoverageGrid(cfg2)
    region_map2 = RegionMap(grid2, cfg2)
    task_gen2 = TaskGenerator(region_map2, cfg2)
    first_created = task_gen2.generate_tasks(t=0.0)
    second_created = task_gen2.generate_tasks(t=10.0)
    if len(second_created) > cfg2.max_new_tasks_per_tick:
        return fail(
            f"per-tick generation overflow: {len(second_created)} > {cfg2.max_new_tasks_per_tick}"
        )
    first_regions = {t.region_id for t in first_created}
    second_regions = {t.region_id for t in second_created}
    if not first_regions.isdisjoint(second_regions):
        return fail("cooldown/dedup failed: same region regenerated too soon")

    # 6) preview output check
    heatmap = ROOT / "runs" / "region_info_heatmap.png"
    pending_png = ROOT / "runs" / "pending_tasks.png"
    if heatmap.exists():
        heatmap.unlink()
    if pending_png.exists():
        pending_png.unlink()

    cmd = [sys.executable, str(ROOT / "scripts" / "preview_tasks.py"), "--seed", "0"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        return fail(f"preview_tasks.py failed: {err}")

    if not heatmap.exists() or heatmap.stat().st_size <= 0:
        return fail("region_info_heatmap.png missing or empty")
    if not pending_png.exists() or pending_png.stat().st_size <= 0:
        return fail("pending_tasks.png missing or empty")

    print("STEP4 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
