from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sim.config import SimConfig, make_run_dir
from sim.sim import Sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal ocean sim skeleton.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps.")
    parser.add_argument("--dt", type=float, default=1.0, help="Time delta per step.")
    parser.add_argument("--map-width", type=float, default=1000.0, help="Map width placeholder.")
    parser.add_argument("--map-height", type=float, default=1000.0, help="Map height placeholder.")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Root directory for run outputs.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit run directory name.")
    return parser.parse_args()


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("sim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def save_config(config: SimConfig, run_dir: Path) -> Path:
    config_path = run_dir / "config.json"
    payload = config.to_dict()
    payload["run_dir"] = str(run_dir)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return config_path


def main() -> int:
    args = parse_args()
    config = SimConfig(
        seed=args.seed,
        dt=args.dt,
        steps=args.steps,
        map_width=args.map_width,
        map_height=args.map_height,
        runs_dir=args.runs_dir,
        run_name=args.run_name,
    )
    run_dir = make_run_dir(config)
    log_path = run_dir / "run.log"
    logger = configure_logging(log_path)

    config_path = save_config(config, run_dir)
    metrics_path = run_dir / "metrics.csv"

    logger.info(
        "Simulation start | seed=%s steps=%s dt=%s map=(%s,%s) run_dir=%s",
        config.seed,
        config.steps,
        config.dt,
        config.map_width,
        config.map_height,
        run_dir,
    )
    logger.info("Config saved to %s", config_path)

    sim = Sim(config=config, metrics_path=metrics_path)
    sim.run()

    logger.info("Simulation done | final_step=%s final_time=%s", sim.step_count, sim.t)
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
