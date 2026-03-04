from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.environment.environment import Environment2D


def fail(msg: str) -> int:
    print(f"STEP2 FAIL: {msg}")
    return 1


def main() -> int:
    try:
        cfg = SimConfig(seed=0, map_width=10000.0, map_height=10000.0)
        env = Environment2D(cfg)
    except Exception as exc:  # noqa: BLE001
        return fail(f"cannot import/init Environment2D: {exc}")

    if abs(env.base_position[1] - 0.0) > 1e-9:
        return fail(f"base_position y is not on coastline: {env.base_position[1]}")

    for region in ("nearshore", "risk_zone", "offshore"):
        try:
            x, y = env.sample_point(region)
        except Exception as exc:  # noqa: BLE001
            return fail(f"sample_point failed for {region}: {exc}")

        if not env.is_inside_map(x, y):
            return fail(f"sample point out of map for {region}: ({x}, {y})")
        band = env.bands[region]
        if not (band.y_min <= y <= band.y_max):
            return fail(f"sample point y out of {region} band: y={y}, expected [{band.y_min}, {band.y_max}]")

    risk = env.bands["risk_zone"]
    for i, ob in enumerate(env.obstacles):
        if not (risk.y_min <= ob.center_y <= risk.y_max):
            return fail(f"obstacle {i} center_y={ob.center_y} outside risk_zone [{risk.y_min}, {risk.y_max}]")

    preview = ROOT / "runs" / "environment_preview.png"
    if preview.exists():
        preview.unlink()

    cmd = [sys.executable, str(ROOT / "scripts" / "preview_environment.py"), "--seed", "0"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip()
        return fail(f"preview_environment failed: {stderr}")

    if not preview.exists():
        return fail(f"preview file missing: {preview}")
    if preview.stat().st_size <= 0:
        return fail(f"preview file empty: {preview}")

    print("STEP2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
