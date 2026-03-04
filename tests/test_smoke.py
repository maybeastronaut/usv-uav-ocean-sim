from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def test_smoke_run_creates_outputs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"

    env = os.environ.copy()
    src_path = project_root / "src"
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    cmd = [
        sys.executable,
        "-m",
        "sim.run",
        "--seed",
        "0",
        "--steps",
        "5",
        "--dt",
        "0.5",
        "--runs-dir",
        str(runs_root),
    ]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    run_dir_line = next((line for line in result.stdout.splitlines() if line.startswith("RUN_DIR=")), None)
    assert run_dir_line is not None, f"missing RUN_DIR in stdout: {result.stdout}"

    run_dir = Path(run_dir_line.split("=", 1)[1].strip())
    assert run_dir.exists()
    assert run_dir.parent == runs_root

    config_path = run_dir / "config.json"
    log_path = run_dir / "run.log"
    metrics_path = run_dir / "metrics.csv"

    assert config_path.exists()
    assert log_path.exists()
    assert metrics_path.exists()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["seed"] == 0
    assert config["steps"] == 5
    assert config["dt"] == 0.5

    log_text = log_path.read_text(encoding="utf-8")
    assert "seed=0" in log_text
    assert "steps=5" in log_text
    assert "dt=0.5" in log_text

    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 5
    assert rows[0]["step"] == "1"
    assert rows[-1]["step"] == "5"
