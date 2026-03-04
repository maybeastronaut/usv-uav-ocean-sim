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


def run_strategy(strategy: str):
    cfg = SimConfig(seed=0, strategy=strategy, t_end=1200.0, sim_dt=5.0)
    sim = Simulator(cfg)
    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt)
    if not result.history:
        raise RuntimeError(f"empty simulation history for strategy={strategy}")
    final = result.history[-1]
    min_battery = float(min(row["uav_battery_min"] for row in result.history))
    return {
        "strategy": strategy,
        "history": result.history,
        "final": final,
        "min_battery_hist": min_battery,
        "recharge_count": int(final["recharge_count_cum"]),
        "done_count": int(final["done_count"]),
        "pending_end": int(final["pending_count"]),
        "dead_count": int(final["uav_dead_count"]),
        "mean_battery_final": float(final["uav_battery_mean"]),
        "mean_info_end": float(final["mean_info_all"]),
    }


def main() -> int:
    nearest = run_strategy("nearest")
    priority = run_strategy("priority")

    if nearest["recharge_count"] < 4:
        return fail(f"nearest recharge_count too low: {nearest['recharge_count']} (<4)")
    if not (0.15 <= nearest["min_battery_hist"] <= 0.40):
        return fail(
            "nearest uav_battery_min_hist out of target range [0.15,0.40]: "
            f"{nearest['min_battery_hist']:.4f}"
        )
    if not (0.35 <= nearest["mean_battery_final"] <= 0.70):
        return fail(
            "nearest uav_battery_mean_final out of target range [0.35,0.70]: "
            f"{nearest['mean_battery_final']:.4f}"
        )
    if nearest["dead_count"] != 0:
        return fail(f"nearest uav_dead_count must be 0, got {nearest['dead_count']}")
    if nearest["done_count"] < 20:
        return fail(f"nearest monitor done_count too low: {nearest['done_count']}")
    if nearest["mean_info_end"] <= 0.05:
        return fail(f"nearest mean_info_all_end too low: {nearest['mean_info_end']:.4f}")

    pending_series = [int(row["pending_count"]) for row in nearest["history"]]
    cfg = SimConfig(seed=0, strategy="nearest", t_end=1200.0, sim_dt=5.0)
    if nearest["pending_end"] >= cfg.max_pending_tasks:
        return fail(f"nearest pending_end saturated at MAX_PENDING_TASKS: {nearest['pending_end']}")
    if max(pending_series) >= cfg.max_pending_tasks and nearest["pending_end"] >= cfg.max_pending_tasks - 2:
        return fail("pending tasks near saturation without recovery")

    if priority["dead_count"] != 0:
        return fail(f"priority uav_dead_count must be 0, got {priority['dead_count']}")
    if priority["recharge_count"] < 3:
        return fail(f"priority recharge_count too low: {priority['recharge_count']} (<3)")
    if priority["done_count"] < int(0.8 * nearest["done_count"]):
        return fail(
            "priority done_count too low vs nearest: "
            f"priority={priority['done_count']}, nearest={nearest['done_count']}, "
            f"expected>={int(0.8 * nearest['done_count'])}"
        )
    if priority["mean_info_end"] < 0.8 * nearest["mean_info_end"]:
        return fail(
            "priority mean_info_all_end too low vs nearest: "
            f"priority={priority['mean_info_end']:.4f}, nearest={nearest['mean_info_end']:.4f}, "
            f"expected>={0.8 * nearest['mean_info_end']:.4f}"
        )

    print(
        "VERIFY METRICS nearest "
        f"recharge_count={nearest['recharge_count']} "
        f"uav_battery_min_hist={nearest['min_battery_hist']:.4f} "
        f"uav_battery_min_final={nearest['final']['uav_battery_min']:.4f} "
        f"uav_battery_mean_final={nearest['mean_battery_final']:.4f} "
        f"done_count={nearest['done_count']} "
        f"pending_end={nearest['pending_end']} "
        f"mean_info_all_end={nearest['mean_info_end']:.4f}"
    )
    print(
        "VERIFY METRICS priority "
        f"recharge_count={priority['recharge_count']} "
        f"uav_dead_count={priority['dead_count']} "
        f"done_count={priority['done_count']} "
        f"pending_end={priority['pending_end']} "
        f"mean_info_all_end={priority['mean_info_end']:.4f}"
    )
    print("STEP7 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
