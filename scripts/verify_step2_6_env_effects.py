from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.environment.environment import Environment2D


def fail(reason: str) -> int:
    print(f"STEP2.6 FAIL: {reason}")
    return 1


def magnitude(v: tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def main() -> int:
    try:
        env = Environment2D(SimConfig(seed=0))
    except Exception as exc:  # noqa: BLE001
        return fail(f"Environment2D init failed: {exc}")

    # 1) current_at / wind_at basic callable checks
    c = env.current_at(1000.0, 1000.0, 0.0)
    w = env.wind_at(1000.0, 1000.0, 0.0)
    if not (
        isinstance(c, tuple)
        and len(c) == 2
        and all(isinstance(v, (int, float)) for v in c)
    ):
        return fail(f"current_at invalid return: {c}")
    if not (
        isinstance(w, tuple)
        and len(w) == 2
        and all(isinstance(v, (int, float)) for v in w)
    ):
        return fail(f"wind_at invalid return: {w}")

    # 2) current trend by zone (average trend with tolerance)
    samples_per_zone = 40
    near_vals: list[float] = []
    risk_vals: list[float] = []
    off_vals: list[float] = []

    for _ in range(samples_per_zone):
        x, y = env.sample_point("nearshore")
        near_vals.append(magnitude(env.current_at(x, y, 0.0)))
        x, y = env.sample_point("risk_zone")
        risk_vals.append(magnitude(env.current_at(x, y, 0.0)))
        x, y = env.sample_point("offshore")
        off_vals.append(magnitude(env.current_at(x, y, 0.0)))

    near_avg = sum(near_vals) / len(near_vals)
    risk_avg = sum(risk_vals) / len(risk_vals)
    off_avg = sum(off_vals) / len(off_vals)

    tol = 0.03
    if not (risk_avg + tol >= near_avg and off_avg + tol >= risk_avg):
        return fail(
            "current trend broken: "
            f"near={near_avg:.4f}, risk={risk_avg:.4f}, offshore={off_avg:.4f}"
        )

    # 3) comm range checks
    base = env.base_position
    near_p = (base[0] + 500.0, base[1] + 100.0)
    far_p = (env.map_width, env.map_height)
    if not env.in_comm_range(base, near_p):
        return fail("in_comm_range should be True for near point")
    if env.in_comm_range(base, far_p):
        return fail("in_comm_range should be False for far point")
    q_near = env.comm_quality(base, near_p)
    q_far = env.comm_quality(base, far_p)
    if not (0.0 <= q_far <= q_near <= 1.0):
        return fail(f"comm_quality out of expected range/order: near={q_near}, far={q_far}")

    # explicit peer-to-peer examples requested
    p1 = (0.0, 0.0)
    p2 = (1000.0, 0.0)
    p3 = (5000.0, 0.0)
    if not env.in_comm_range(p1, p2):
        return fail("in_comm_range(p1,p2) should be True for distance=1000")
    if env.in_comm_range(p1, p3):
        return fail("in_comm_range(p1,p3) should be False for distance=5000")

    # 4) preview existence checks
    preview = ROOT / "runs" / "environment_preview.png"
    if preview.exists():
        preview.unlink()
    cmd = [sys.executable, str(ROOT / "scripts" / "preview_environment.py"), "--seed", "0"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        return fail(f"preview_environment failed: {err}")
    if not preview.exists():
        return fail("environment_preview.png not generated")
    if preview.stat().st_size <= 0:
        return fail("environment_preview.png is empty")

    print("STEP2.6 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
