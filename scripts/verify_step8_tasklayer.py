from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.simulator import Simulator


def fail(msg: str) -> int:
    print(f"STEP8 FAIL: {msg}")
    return 1


def run_policy(policy: str, *, ablate_softpart: bool = False, ablate_energy: bool = False, ablate_risk: bool = False):
    cfg = SimConfig(
        seed=0,
        t_end=1200.0,
        sim_dt=5.0,
        strategy=policy,
        task_policy=policy,
        ablate_softpart=ablate_softpart,
        ablate_energy_term=ablate_energy,
        ablate_risk_term=ablate_risk,
    )
    sim = Simulator(cfg)
    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt)
    if not result.history:
        raise RuntimeError(f"empty history for {policy}")
    final = result.history[-1]
    return {
        "policy": policy,
        "mean_info_end": float(final["mean_info_all"]),
        "done": int(final["done_count"]),
        "pending": int(final["pending_count"]),
        "recharge": int(final["recharge_count_cum"]),
        "dead": int(final["uav_dead_count"]),
        "hit_rate": float(sim.usv_preference_hit_rate()),
        "cfg": cfg,
    }


def main() -> int:
    nearest = run_policy("nearest")
    priority = run_policy("priority")
    multimetric = run_policy("multimetric")
    ablate_soft = run_policy("multimetric", ablate_softpart=True)

    min_done_ref = min(nearest["done"], priority["done"])
    max_mean_ref = max(nearest["mean_info_end"], priority["mean_info_end"])

    if multimetric["done"] < int(0.95 * min_done_ref):
        return fail(
            f"multimetric done too low: {multimetric['done']} < 95% * min(nearest,priority)={min_done_ref}"
        )
    if multimetric["mean_info_end"] < 0.95 * max_mean_ref:
        return fail(
            "multimetric mean_info too low: "
            f"{multimetric['mean_info_end']:.4f} < 95% * max(nearest,priority)={max_mean_ref:.4f}"
        )
    if multimetric["dead"] != 0:
        return fail(f"multimetric uav_dead_count != 0: {multimetric['dead']}")
    if multimetric["pending"] > int(0.95 * multimetric["cfg"].max_pending_tasks):
        return fail(
            f"multimetric pending too high: {multimetric['pending']} > 95%*MAX={int(0.95 * multimetric['cfg'].max_pending_tasks)}"
        )

    if ablate_soft["hit_rate"] > multimetric["hit_rate"] - 0.03:
        return fail(
            "soft-partition ablation sanity failed: "
            f"baseline_hit={multimetric['hit_rate']:.4f}, ablated_hit={ablate_soft['hit_rate']:.4f}"
        )

    print("policy,mean_info_all_end,done_count,pending_end,recharge_count,uav_dead_count,usv_preference_hit_rate")
    for row in (nearest, priority, multimetric, ablate_soft):
        name = row["policy"] if row is not ablate_soft else "multimetric_ablate_softpart"
        print(
            f"{name},{row['mean_info_end']:.4f},{row['done']},{row['pending']},{row['recharge']},{row['dead']},{row['hit_rate']:.4f}"
        )
    print("STEP8 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
