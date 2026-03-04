from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = {
    "total_distance_uav",
    "total_distance_usv",
    "total_distance_all",
    "total_energy_uav",
    "total_energy_usv",
    "total_energy_all",
    "baseline_pre",
    "threshold_recover",
    "recovered",
    "recovery_time_sec",
    "info_per_distance",
    "done_per_distance",
}

REQUIRED_FIGS = [
    "exp2_recovery_curve_v2.png",
    "exp1_distance_all.png",
    "exp1_energy_all.png",
    "exp1_info_per_distance.png",
    "exp1_done_per_distance.png",
]


def fail(msg: str) -> int:
    print(f"STEP12 FAIL: {msg}")
    return 1


def main() -> int:
    outdir = ROOT / "runs" / "step12"
    if outdir.exists():
        shutil.rmtree(outdir)

    env = dict(os.environ)
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/pycache")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

    run_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "experiment_runner_step11.py"),
        "--exp",
        "all",
        "--outdir",
        str(outdir),
        "--seeds",
        "0,1,2",
        "--duration",
        "1200",
        "--low-load",
        "--no-plots",
    ]
    subprocess.run(run_cmd, check=True, env=env)

    plot_cmd = [sys.executable, str(ROOT / "scripts" / "plot_step12.py"), "--outdir", str(outdir)]
    subprocess.run(plot_cmd, check=True, env=env)

    results_path = outdir / "results.csv"
    summary_path = outdir / "summary.csv"
    figs_dir = outdir / "figs"

    if not results_path.exists() or results_path.stat().st_size <= 0:
        return fail("results.csv missing or empty")
    if not summary_path.exists() or summary_path.stat().st_size <= 0:
        return fail("summary.csv missing or empty")

    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    if not rows:
        return fail("results.csv has no rows")
    missing = REQUIRED_FIELDS - fields
    if missing:
        return fail(f"results.csv missing required Step12 fields: {sorted(missing)}")

    has_positive_recovery = False
    for row in rows:
        raw = row.get("recovery_time_sec", "")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if val > 0.0:
            has_positive_recovery = True
            break
    if not has_positive_recovery:
        return fail("no run has recovery_time_sec > 0")

    missing_figs = [name for name in REQUIRED_FIGS if not (figs_dir / name).exists()]
    if missing_figs:
        return fail(f"missing required figs: {missing_figs}")

    print(f"RESULTS={results_path}")
    print(f"SUMMARY={summary_path}")
    for name in REQUIRED_FIGS:
        print(f"FIG={figs_dir / name}")
    print("STEP12 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
