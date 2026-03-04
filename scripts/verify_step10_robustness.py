from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.simulator import Simulator


FAIL_T = 600.0
WINDOW = 200.0


def fail(msg: str) -> int:
    print(f"STEP10 FAIL: {msg}")
    return 1


def _calc_recovery_time(history: list[dict[str, float]], fail_t: float) -> float:
    pre = [row["mean_info_all"] for row in history if row["time"] < fail_t]
    if not pre:
        return math.inf
    pre_mean = sum(pre[-5:]) / min(5, len(pre))
    target = 0.95 * pre_mean
    for row in history:
        if row["time"] < fail_t:
            continue
        if row["mean_info_all"] >= target:
            return float(row["time"] - fail_t)
    return math.inf


def _mean_min_after_failure(history: list[dict[str, float]], fail_t: float, window: float) -> float:
    vals = [row["mean_info_all"] for row in history if fail_t <= row["time"] <= fail_t + window]
    if not vals:
        return 0.0
    return float(min(vals))


def run_case(enable_robust_response: bool) -> dict[str, float | int | bool]:
    cfg = SimConfig(
        seed=0,
        t_end=1200.0,
        sim_dt=5.0,
        strategy="multimetric",
        task_policy="multimetric",
        enable_feedback=True,
        enable_failures=True,
        failure_mode="scheduled",
        failure_t_sec=FAIL_T,
        failure_usv_id=1,
        failure_kind="DISABLED",
        enable_robust_response=enable_robust_response,
        forced_relax_on_failure=enable_robust_response,
    )
    sim = Simulator(cfg)
    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt)
    if not result.history:
        raise RuntimeError("empty history")
    final = result.history[-1]

    return {
        "robust": enable_robust_response,
        "mean_end": float(final["mean_info_all"]),
        "mean_min_after_failure": _mean_min_after_failure(result.history, FAIL_T, WINDOW),
        "recovery_time_sec": _calc_recovery_time(result.history, FAIL_T),
        "pending_end": int(final["pending_count"]),
        "done_count": int(final["done_count"]),
        "uav_dead_count": int(final["uav_dead_count"]),
        "fb_trigger": int(final.get("fb_trigger_count_cum", 0.0)),
    }


def main() -> int:
    baseline = run_case(enable_robust_response=False)
    robust = run_case(enable_robust_response=True)

    base_rec = float(baseline["recovery_time_sec"])
    rob_rec = float(robust["recovery_time_sec"])

    if math.isinf(base_rec):
        if math.isinf(rob_rec):
            return fail("neither baseline nor robust recovered")
    elif rob_rec > base_rec + 1e-9:
        return fail(f"robust recovery slower: robust={rob_rec:.1f}s baseline={base_rec:.1f}s")

    if float(robust["mean_min_after_failure"]) < float(baseline["mean_min_after_failure"]) - 0.01:
        return fail(
            "robust mean_min_after_failure too low: "
            f"robust={robust['mean_min_after_failure']:.4f}, baseline={baseline['mean_min_after_failure']:.4f}"
        )

    if int(robust["uav_dead_count"]) != 0:
        return fail(f"uav_dead_count must be 0, got {robust['uav_dead_count']}")

    print("mode,recovery_time_sec,mean_min_after_failure,mean_end,pending_end,done_count,fb_trigger")
    for row in (baseline, robust):
        name = "ROBUST" if row["robust"] else "BASELINE"
        rec = row["recovery_time_sec"]
        rec_str = "inf" if math.isinf(float(rec)) else f"{float(rec):.1f}"
        print(
            f"{name},{rec_str},{row['mean_min_after_failure']:.4f},{row['mean_end']:.4f},"
            f"{row['pending_end']},{row['done_count']},{row['fb_trigger']}"
        )

    print("STEP10 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
