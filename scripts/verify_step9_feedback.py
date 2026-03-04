from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.simulator import Simulator


def fail(msg: str) -> int:
    print(f"STEP9 FAIL: {msg}")
    return 1


def run_case(enable_feedback: bool) -> dict[str, float | int | bool]:
    cfg = SimConfig(
        seed=0,
        t_end=1200.0,
        sim_dt=5.0,
        strategy="multimetric",
        task_policy="multimetric",
        enable_feedback=enable_feedback,
    )
    sim = Simulator(cfg)
    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt)
    if not result.history:
        raise RuntimeError("empty simulation history")

    final = result.history[-1]
    return {
        "enable_feedback": enable_feedback,
        "mean_end": float(final["mean_info_all"]),
        "done": int(final["done_count"]),
        "pending_end": int(final["pending_count"]),
        "recharge_count": int(final["recharge_count_cum"]),
        "uav_dead_count": int(final["uav_dead_count"]),
        "fb_trigger_count": int(final.get("fb_trigger_count_cum", 0.0)),
        "max_pending": int(cfg.max_pending_tasks),
    }


def main() -> int:
    without_fb = run_case(enable_feedback=False)
    with_fb = run_case(enable_feedback=True)

    if with_fb["fb_trigger_count"] < 1:
        return fail(f"feedback never triggered: fb_trigger_count={with_fb['fb_trigger_count']}")

    if with_fb["mean_end"] < 0.95 * without_fb["mean_end"]:
        return fail(
            "WITH feedback mean_info_all_end regressed too much: "
            f"with={with_fb['mean_end']:.4f}, without={without_fb['mean_end']:.4f}"
        )

    if with_fb["pending_end"] > with_fb["max_pending"]:
        return fail(
            f"pending_end overflow: {with_fb['pending_end']} > MAX_PENDING_TASKS={with_fb['max_pending']}"
        )

    if with_fb["uav_dead_count"] != 0:
        return fail(f"uav_dead_count must be 0, got {with_fb['uav_dead_count']}")

    if with_fb["recharge_count"] < 1:
        return fail(f"recharge_count too low: {with_fb['recharge_count']} (<1)")

    improved = with_fb["mean_end"] >= without_fb["mean_end"] or with_fb["pending_end"] <= without_fb["pending_end"]
    if not improved:
        return fail(
            "WITH feedback did not improve any target metric: "
            f"mean(with={with_fb['mean_end']:.4f}, without={without_fb['mean_end']:.4f}), "
            f"pending(with={with_fb['pending_end']}, without={without_fb['pending_end']})"
        )

    print("mode,mean_info_all_end,done_count,pending_end,recharge_count,uav_dead_count,fb_trigger_count")
    for row in (without_fb, with_fb):
        mode = "WITH_FEEDBACK" if row["enable_feedback"] else "WITHOUT_FEEDBACK"
        print(
            f"{mode},{row['mean_end']:.4f},{row['done']},{row['pending_end']},"
            f"{row['recharge_count']},{row['uav_dead_count']},{row['fb_trigger_count']}"
        )

    print("STEP9 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
