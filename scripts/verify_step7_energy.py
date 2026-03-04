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

    initial_battery = 1.0
    min_battery = min(row["uav_battery_min"] for row in result.history)
    final = result.history[-1]
    recharge_count = int(final["recharge_count_cum"])
    done_count = int(final["done_count"])
    pending_end = int(final["pending_count"])
    dead_count = int(final["uav_dead_count"])

    if not (min_battery < initial_battery - 1e-6):
        return fail(f"uav battery did not decrease: min={min_battery:.4f}")
    if recharge_count < 1:
        return fail("no recharge completed")
    if dead_count > 1:
        return fail(f"too many UAV emergency events: {dead_count}")
    if done_count < 20:
        return fail(f"monitor done_count too low: {done_count}")

    pending_series = [int(row["pending_count"]) for row in result.history]
    if max(pending_series) >= cfg.max_pending_tasks and pending_end >= cfg.max_pending_tasks:
        return fail("pending tasks saturated at MAX_PENDING_TASKS")

    print(
        "VERIFY METRICS "
        f"recharge_count={recharge_count} "
        f"uav_battery_min={final['uav_battery_min']:.4f} "
        f"uav_battery_mean={final['uav_battery_mean']:.4f} "
        f"done_count={done_count} "
        f"pending_end={pending_end}"
    )
    print("STEP7 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
