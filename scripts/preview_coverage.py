from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.config import SimConfig
from sim.coverage.coverage_grid import CoverageGrid, check_resolution
from sim.environment.environment import Environment2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview coverage grid and info decay.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=30.0, help="seconds per simulation step")
    parser.add_argument("--runs-dir", type=str, default="runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SimConfig(seed=args.seed, runs_dir=args.runs_dir)
    runs_dir = Path(config.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    env = Environment2D(config)
    grid = CoverageGrid(config)

    uav_diag = check_resolution(config.cell_size, config.uav_sensor_radius)
    usv_diag = check_resolution(config.cell_size, config.usv_sensor_radius)
    print(
        f"[RESOLUTION][UAV] level={uav_diag['level']} r={uav_diag['ratio']:.2f} "
        f"(R={config.uav_sensor_radius:.1f}, CELL={config.cell_size:.1f}) {uav_diag['message']}"
    )
    print(
        f"[RESOLUTION][USV] level={usv_diag['level']} r={usv_diag['ratio']:.2f} "
        f"(R={config.usv_sensor_radius:.1f}, CELL={config.cell_size:.1f}) {usv_diag['message']}"
    )

    times: list[float] = []
    mean_vals: list[float] = []
    min_visited_vals: list[float] = []
    p5_visited_vals: list[float] = []

    for step in range(args.steps):
        t = step * args.dt

        # Virtual observations only (no agent dynamics yet).
        for uav_idx in range(config.num_uav):
            region = "offshore" if (step + uav_idx) % 2 == 0 else "risk_zone"
            p = env.sample_point(region)
            grid.observe(position=p, radius=config.uav_sensor_radius, t=t)

        for usv_idx in range(config.num_usv):
            region = "nearshore" if usv_idx < 2 else "risk_zone"
            p = env.sample_point(region)
            grid.observe(position=p, radius=config.usv_sensor_radius, t=t)

        times.append(t)
        mean_vals.append(grid.mean_info(t, mode="all"))
        min_visited_vals.append(grid.min_info(t, mode="visited"))
        p5_visited_vals.append(grid.percentile_info(t, 5.0, mode="visited"))

    final_t = times[-1] if times else 0.0
    info = grid.info_map(final_t)

    heatmap_path = runs_dir / "coverage_heatmap.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(
        info,
        origin="lower",
        extent=[0.0, config.map_width, 0.0, config.map_height],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.axhline(y=config.nearshore_y_max, color="white", linestyle="--", linewidth=1.2)
    ax.axhline(y=config.risk_zone_y_max, color="white", linestyle="--", linewidth=1.2)
    uav_ratio = config.uav_sensor_radius / config.cell_size
    usv_ratio = config.usv_sensor_radius / config.cell_size
    ax.set_title(
        "Coverage Info Heatmap "
        f"(t={final_t:.1f}s)\n"
        f"CELL_SIZE={config.cell_size:.1f}m, UAV_R={config.uav_sensor_radius:.1f}m, "
        f"USV_R={config.usv_sensor_radius:.1f}m, "
        f"R/CELL(UAV)={uav_ratio:.2f}, R/CELL(USV)={usv_ratio:.2f}"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("info value")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    curve_path = runs_dir / "coverage_curve.png"
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(times, mean_vals, label="mean_all", color="tab:blue", linewidth=2.0)
    ax2.plot(times, min_visited_vals, label="min_visited", color="tab:red", linewidth=2.0)
    ax2.plot(times, p5_visited_vals, label="p5_visited", color="tab:green", linewidth=1.8)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("info value")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_title("Coverage Info Curve")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(curve_path, dpi=150)
    plt.close(fig2)

    print(f"HEATMAP={heatmap_path}")
    print(f"CURVE={curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
