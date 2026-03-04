from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = {
    "exp_name",
    "policy",
    "seed",
    "duration",
    "enable_feedback",
    "enable_failures",
    "enable_robust_response",
    "failure_kind",
    "failure_t",
    "failure_usv",
    "mean_info_all_end",
    "mean_info_all_min",
    "mean_info_all_p5",
    "done_count",
    "pending_end",
    "total_distance_usv",
    "total_distance_uav",
    "recharge_count",
    "uav_dead_count",
    "fb_trigger_count",
    "recovery_time_sec",
    "mean_min_after_failure",
}


def fail(msg: str) -> int:
    print(f"STEP11 FAIL: {msg}")
    return 1


def main() -> int:
    outdir = ROOT / "runs" / "step11_smoke"
    if outdir.exists():
        shutil.rmtree(outdir)

    env = dict(**__import__("os").environ)
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/pycache")

    runner_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "experiment_runner_step11.py"),
        "--exp",
        "exp1",
        "--outdir",
        str(outdir),
        "--seeds",
        "0,1",
        "--duration",
        "600",
    ]
    subprocess.run(runner_cmd, check=True, env=env)

    plot_cmd = [sys.executable, str(ROOT / "scripts" / "plot_step11.py"), "--outdir", str(outdir)]
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
        return fail("results.csv has no data rows")
    missing = REQUIRED_FIELDS - fields
    if missing:
        return fail(f"results.csv missing fields: {sorted(missing)}")

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        s_reader = csv.DictReader(f)
        s_rows = list(s_reader)
    if len(s_rows) < 4:
        return fail(f"summary rows too few for exp1 policies: {len(s_rows)}")

    figs = list(figs_dir.glob("*.png")) if figs_dir.exists() else []
    if len(figs) < 3:
        return fail(f"figures too few: {len(figs)} (<3)")

    print(f"SMOKE_RESULTS={results_path}")
    print(f"SMOKE_SUMMARY={summary_path}")
    print(f"SMOKE_FIG_COUNT={len(figs)}")
    print("STEP11 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
