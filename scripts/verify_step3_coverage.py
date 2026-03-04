from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.coverage.coverage_grid import CoverageGrid, check_resolution


def fail(reason: str) -> int:
    print(f"STEP3 FAIL: {reason}")
    return 1


def main() -> int:
    cfg = SimConfig(seed=0, cell_size=100.0, decay_mode="exponential", decay_tau=1800.0)
    try:
        grid = CoverageGrid(cfg)
    except Exception as exc:  # noqa: BLE001
        return fail(f"CoverageGrid init failed: {exc}")

    # 1) dimension check
    exp_nx = int(math.ceil(cfg.map_width / cfg.cell_size))
    exp_ny = int(math.ceil(cfg.map_height / cfg.cell_size))
    if grid.nx != exp_nx or grid.ny != exp_ny:
        return fail(f"grid shape mismatch: got=({grid.nx},{grid.ny}) expected=({exp_nx},{exp_ny})")
    if grid.visited_count != 0:
        return fail(f"visited_count should be 0 at init, got {grid.visited_count}")

    # 2) observe increases info
    t_obs = 100.0
    mean_before = grid.mean_info(t_obs)
    updated = grid.observe(position=(cfg.map_width * 0.5, cfg.map_height * 0.5), radius=600.0, t=t_obs)
    if updated <= 0:
        return fail("observe updated zero cells")
    if grid.visited_count <= 0:
        return fail("visited_count should be > 0 after observe")
    mean_after = grid.mean_info(t_obs)
    if mean_after <= mean_before:
        return fail(f"mean_info did not increase after observe: before={mean_before}, after={mean_after}")

    min_all = grid.min_info(t_obs, mode="all")
    min_visited = grid.min_info(t_obs, mode="visited")
    if min_all > 1e-6:
        return fail(f"min_all expected near 0 after sparse observe, got {min_all}")
    if min_visited <= 0.8:
        return fail(f"min_visited expected > 0.8 right after observe, got {min_visited}")

    # 3) info decay with time
    mean_late = grid.mean_info(t_obs + 3600.0)
    if mean_late >= mean_after:
        return fail(f"decay not effective: at_obs={mean_after}, at_late={mean_late}")

    # 3.5) resolution diagnostics
    diag = check_resolution(cfg.cell_size, cfg.uav_sensor_radius)
    if "ratio" not in diag or not isinstance(diag["ratio"], float):
        return fail(f"resolution check missing ratio: {diag}")
    if diag["ratio"] < 2.0 and diag["level"] != "WARNING":
        return fail(f"expected WARNING for ratio<2, got {diag}")

    # 4) preview script output
    heatmap = ROOT / "runs" / "coverage_heatmap.png"
    curve = ROOT / "runs" / "coverage_curve.png"
    if heatmap.exists():
        heatmap.unlink()
    if curve.exists():
        curve.unlink()

    cmd = [sys.executable, str(ROOT / "scripts" / "preview_coverage.py"), "--seed", "0"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        return fail(f"preview_coverage.py failed: {err}")

    if not heatmap.exists() or heatmap.stat().st_size <= 0:
        return fail("coverage_heatmap.png missing or empty")
    if not curve.exists() or curve.stat().st_size <= 0:
        return fail("coverage_curve.png missing or empty")

    print("STEP3 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
