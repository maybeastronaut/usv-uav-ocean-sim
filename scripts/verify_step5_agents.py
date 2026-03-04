from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.simulator import Simulator


def fail(msg: str) -> int:
    print(f"STEP5 FAIL: {msg}")
    return 1


def main() -> int:
    cfg = SimConfig(seed=0, sim_dt=5.0, t_end=300.0)
    sim = Simulator(cfg)
    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt)

    # 1) run should complete and history non-empty
    if not result.history:
        return fail("simulator history is empty")

    # 2) pending -> assigned -> done transitions
    assigned_cnt = result.transition_counts.get("assigned", 0)
    done_cnt = result.transition_counts.get("done", 0)
    print(f"assigned_count={assigned_cnt}, done_count={done_cnt}")
    if assigned_cnt <= 0 or done_cnt <= 0:
        return fail("expected assigned and done transitions > 0")

    # 3) coverage mean info should increase
    mean_vals = [row["mean_info_all"] for row in result.history]
    if max(mean_vals) <= 0.0:
        return fail("mean_info_all never increased above 0")

    # 4) position bounds and USV obstacle checks
    for agent in sim.agents:
        x, y = agent.pos
        if not sim.env.is_inside_map(x, y):
            return fail(f"agent out of bounds: {agent.agent_id} pos={agent.pos}")
        if agent.agent_type == "USV" and sim.env.is_in_obstacle(x, y):
            return fail(f"USV inside obstacle: {agent.agent_id} pos={agent.pos}")
    if sim.out_of_bounds_count > 0:
        return fail(f"out_of_bounds_count={sim.out_of_bounds_count}")
    if sim.usv_collision_count > 0:
        return fail(f"usv_collision_count={sim.usv_collision_count}")

    # 5) preview outputs exist
    snapshot = ROOT / "runs" / "step5_snapshot.png"
    curves = ROOT / "runs" / "step5_curves.png"
    if snapshot.exists():
        snapshot.unlink()
    if curves.exists():
        curves.unlink()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "preview_sim_step5.py"),
        "--seed",
        "0",
        "--t-end",
        "300",
        "--dt",
        "5",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        return fail(f"preview_sim_step5.py failed: {err}")

    if not snapshot.exists() or snapshot.stat().st_size <= 0:
        return fail("step5_snapshot.png missing or empty")
    if not curves.exists() or curves.stat().st_size <= 0:
        return fail("step5_curves.png missing or empty")

    print("STEP5 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
