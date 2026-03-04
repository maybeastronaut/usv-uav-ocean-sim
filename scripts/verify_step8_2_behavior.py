from __future__ import annotations

import io
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.simulator import Simulator


def run_case(*, ablate_softpart: bool) -> dict[str, float]:
    cfg = SimConfig(
        seed=0,
        t_end=1200.0,
        sim_dt=5.0,
        strategy="multimetric",
        task_policy="multimetric",
        ablate_softpart=ablate_softpart,
    )
    sim = Simulator(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt)

    final = result.history[-1]
    return {
        "mean_end": float(final["mean_info_all"]),
        "done": float(final["done_count"]),
        "pending_end": float(final["pending_count"]),
        "uav_dead_count": float(final["uav_dead_count"]),
        "usv_preference_hit_rate": float(sim.usv_preference_hit_rate()),
        "usv_cross_band_ratio": float(sim.usv_cross_band_ratio()),
    }


def main() -> int:
    m = run_case(ablate_softpart=False)
    a = run_case(ablate_softpart=True)

    mean_threshold = 0.0742 * 0.95
    done_threshold = 22.0 * 0.95
    pending_threshold = 89.0 * 1.05

    hit_diff = m["usv_preference_hit_rate"] - a["usv_preference_hit_rate"]
    cross_diff = a["usv_cross_band_ratio"] - m["usv_cross_band_ratio"]

    print(
        "multimetric",
        f"mean_end={m['mean_end']:.4f}",
        f"done={int(m['done'])}",
        f"pending_end={int(m['pending_end'])}",
        f"uav_dead_count={int(m['uav_dead_count'])}",
        f"usv_preference_hit_rate={m['usv_preference_hit_rate']:.4f}",
        f"usv_cross_band_ratio={m['usv_cross_band_ratio']:.4f}",
    )
    print(
        "multimetric_ablate_softpart",
        f"mean_end={a['mean_end']:.4f}",
        f"done={int(a['done'])}",
        f"pending_end={int(a['pending_end'])}",
        f"uav_dead_count={int(a['uav_dead_count'])}",
        f"usv_preference_hit_rate={a['usv_preference_hit_rate']:.4f}",
        f"usv_cross_band_ratio={a['usv_cross_band_ratio']:.4f}",
    )
    print(f"hit_rate_diff={hit_diff:.4f}")
    print(f"cross_band_ratio_diff={cross_diff:.4f}")

    ok = True
    reasons: list[str] = []

    if m["mean_end"] < mean_threshold:
        ok = False
        reasons.append(f"mean_end {m['mean_end']:.4f} < {mean_threshold:.4f}")
    if m["done"] < done_threshold:
        ok = False
        reasons.append(f"done {m['done']:.1f} < {done_threshold:.1f}")
    if m["pending_end"] > pending_threshold:
        ok = False
        reasons.append(f"pending_end {m['pending_end']:.1f} > {pending_threshold:.1f}")
    if int(m["uav_dead_count"]) != 0:
        ok = False
        reasons.append(f"uav_dead_count {int(m['uav_dead_count'])} != 0")
    if (hit_diff < 0.20) and (cross_diff < 0.20):
        ok = False
        reasons.append(
            f"behavior diffs too small: hit_rate_diff={hit_diff:.4f}, cross_band_ratio_diff={cross_diff:.4f}"
        )

    if ok:
        print("STEP8.2 BEHAVIOR PASS")
        return 0

    print("STEP8.2 BEHAVIOR FAIL")
    for r in reasons:
        print("-", r)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
