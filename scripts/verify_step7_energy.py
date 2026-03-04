from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.simulator import Simulator


def fail(msg: str) -> int:
    print(f"STEP7 FAIL: {msg}")
    return 1


def main() -> int:
    cfg = SimConfig(seed=0, strategy="nearest", t_end=1200.0, sim_dt=5.0)
    sim = Simulator(cfg)
    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt)
    if not result.history:
        return fail("empty simulation history")

    min_battery = float(min(row["uav_battery_min"] for row in result.history))
    final = result.history[-1]
    recharge_count = int(final["recharge_count_cum"])
    done_count = int(final["done_count"])
    pending_end = int(final["pending_count"])
    dead_count = int(final["uav_dead_count"])
    mean_battery = float(final["uav_battery_mean"])
    mean_info = float(final["mean_info_all"])

    if recharge_count < 4:
        return fail(f"recharge_count too low: {recharge_count} (<4)")
    if not (0.15 <= min_battery <= 0.40):
        return fail(f"uav_battery_min out of target range [0.15,0.40]: {min_battery:.4f}")
    if not (0.35 <= mean_battery <= 0.70):
        return fail(f"uav_battery_mean out of target range [0.35,0.70]: {mean_battery:.4f}")
    if dead_count != 0:
        return fail(f"uav_dead_count must be 0, got {dead_count}")
    if done_count < 20:
        return fail(f"monitor done_count too low: {done_count}")
    if mean_info <= 0.05:
        return fail(f"mean_info_all_end too low: {mean_info:.4f}")

    pending_series = [int(row["pending_count"]) for row in result.history]
    if pending_end >= cfg.max_pending_tasks:
        return fail(f"pending_end saturated at MAX_PENDING_TASKS: {pending_end}")
    if max(pending_series) >= cfg.max_pending_tasks and pending_end >= cfg.max_pending_tasks - 2:
        return fail("pending tasks near saturation without recovery")

    print(
        "VERIFY METRICS "
        f"recharge_count={recharge_count} "
        f"uav_battery_min_hist={min_battery:.4f} "
        f"uav_battery_min_final={final['uav_battery_min']:.4f} "
        f"uav_battery_mean_final={final['uav_battery_mean']:.4f} "
        f"done_count={done_count} "
        f"pending_end={pending_end} "
        f"mean_info_all_end={mean_info:.4f}"
    )
    print("STEP7 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
