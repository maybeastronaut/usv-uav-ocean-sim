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
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

from sim.config import SimConfig
from sim.environment.environment import Environment2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview Environment2D layout.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-width", type=float, default=10000.0)
    parser.add_argument("--map-height", type=float, default=10000.0)
    parser.add_argument("--base-x", type=float, default=None)
    parser.add_argument("--base-y", type=float, default=None)
    parser.add_argument("--nearshore-y-max", type=float, default=2000.0)
    parser.add_argument("--risk-zone-y-min", type=float, default=2000.0)
    parser.add_argument("--risk-zone-y-max", type=float, default=6000.0)
    parser.add_argument("--offshore-y-min", type=float, default=6000.0)
    parser.add_argument("--offshore-y-max", type=float, default=10000.0)
    parser.add_argument("--risk-obstacle-count", type=int, default=20)
    parser.add_argument("--risk-obstacle-radius-min", type=float, default=80.0)
    parser.add_argument("--risk-obstacle-radius-max", type=float, default=220.0)
    parser.add_argument("--current-noise-amplitude", type=float, default=0.05)
    parser.add_argument("--current-time-variation-amplitude", type=float, default=0.08)
    parser.add_argument("--current-time-variation-period", type=float, default=600.0)
    parser.add_argument("--wind-y-gradient", type=float, default=0.35)
    parser.add_argument("--wind-time-variation-amplitude", type=float, default=0.8)
    parser.add_argument("--wind-time-variation-period", type=float, default=900.0)
    parser.add_argument("--comm-radius", type=float, default=3000.0)
    parser.add_argument("--quiver-step", type=float, default=1000.0)
    parser.add_argument("--quiver-scale", type=float, default=140.0)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--runs-dir", type=str, default="runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SimConfig(
        seed=args.seed,
        map_width=args.map_width,
        map_height=args.map_height,
        base_x=args.base_x,
        base_y=args.base_y,
        nearshore_y_max=args.nearshore_y_max,
        risk_zone_y_min=args.risk_zone_y_min,
        risk_zone_y_max=args.risk_zone_y_max,
        offshore_y_min=args.offshore_y_min,
        offshore_y_max=args.offshore_y_max,
        risk_obstacle_count=args.risk_obstacle_count,
        risk_obstacle_radius_min=args.risk_obstacle_radius_min,
        risk_obstacle_radius_max=args.risk_obstacle_radius_max,
        current_noise_amplitude=args.current_noise_amplitude,
        current_time_variation_amplitude=args.current_time_variation_amplitude,
        current_time_variation_period=args.current_time_variation_period,
        wind_y_gradient=args.wind_y_gradient,
        wind_time_variation_amplitude=args.wind_time_variation_amplitude,
        wind_time_variation_period=args.wind_time_variation_period,
        comm_radius=args.comm_radius,
        quiver_step=args.quiver_step,
        quiver_scale=args.quiver_scale,
        runs_dir=args.runs_dir,
    )

    runs_dir = Path(config.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    env = Environment2D(config)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axhspan(0.0, env.bands["nearshore"].y_max, color="#add8e6", alpha=0.15)
    ax.axhspan(env.bands["risk_zone"].y_min, env.bands["risk_zone"].y_max, color="#ffd8a8", alpha=0.15)
    ax.axhspan(env.bands["offshore"].y_min, env.bands["offshore"].y_max, color="#b2f7ef", alpha=0.15)
    sea_rect = Rectangle((0, 0), env.map_width, env.map_height, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(sea_rect)
    ax.add_line(Line2D([0.0, env.map_width], [0.0, 0.0], color="#1f1f1f", linewidth=2.0, label="coastline y=0"))

    # zone boundaries
    ax.axhline(y=env.bands["nearshore"].y_max, color="#2E8B57", linestyle="--", linewidth=1.8, label="nearshore upper")
    ax.axhline(y=env.bands["risk_zone"].y_min, color="#FF8C00", linestyle="--", linewidth=1.8, label="risk lower")
    ax.axhline(y=env.bands["risk_zone"].y_max, color="#FF8C00", linestyle="--", linewidth=1.8, label="risk upper")
    ax.axhline(y=env.bands["offshore"].y_min, color="#1E90FF", linestyle="--", linewidth=1.8, label="offshore lower")

    for ob in env.obstacles:
        ax.add_patch(
            Circle(
                (ob.center_x, ob.center_y),
                ob.radius,
                facecolor="tomato",
                edgecolor="darkred",
                alpha=0.6,
                linewidth=0.8,
            )
        )

    bx, by = env.base_position
    comm_circle = Circle(
        (bx, by),
        env.comm_radius,
        fill=False,
        edgecolor="purple",
        linewidth=2.0,
        linestyle="--",
        alpha=0.8,
    )
    # Clip communication circle to ocean area only (y >= 0 and map bounds).
    comm_circle.set_clip_path(sea_rect)
    ax.add_patch(comm_circle)

    # current quiver
    points = env.field_grid_points(config.quiver_step)
    cur_x: list[float] = []
    cur_y: list[float] = []
    cur_u: list[float] = []
    cur_v: list[float] = []
    for px, py in points:
        vx, vy = env.current_at(px, py, args.time)
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < 0.05:
            continue
        cur_x.append(px)
        cur_y.append(py)
        cur_u.append(vx * config.quiver_scale)
        cur_v.append(vy * config.quiver_scale)
    ax.quiver(
        cur_x,
        cur_y,
        cur_u,
        cur_v,
        color="blue",
        alpha=0.8,
        angles="xy",
        scale_units="xy",
        scale=2.0,
        width=0.002,
        linewidths=1.0,
    )

    # wind quiver (sparser)
    wind_points = env.field_grid_points(config.quiver_step * 2.0)
    wind_x: list[float] = []
    wind_y: list[float] = []
    wind_u: list[float] = []
    wind_v: list[float] = []
    for px, py in wind_points:
        wx, wy = env.wind_at(px, py, args.time)
        speed = (wx * wx + wy * wy) ** 0.5
        if speed < 0.05:
            continue
        wind_x.append(px)
        wind_y.append(py)
        wind_u.append(wx * config.quiver_scale * 0.6)
        wind_v.append(wy * config.quiver_scale * 0.6)
    ax.quiver(
        wind_x,
        wind_y,
        wind_u,
        wind_v,
        color="green",
        alpha=0.8,
        angles="xy",
        scale_units="xy",
        scale=2.0,
        width=0.0018,
        linewidths=1.0,
    )

    ax.scatter([bx], [by], marker="*", s=200, color="black", label="base")
    ax.text(bx + 120.0, by + 120.0, "Base", fontsize=10, color="black")
    ax.set_xlim(0, env.map_width)
    ax.set_ylim(0, env.map_height)
    ax.set_aspect("equal")
    ax.set_title("Environment2D Preview (Current / Wind / Comm Range)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    legend_handles = [
        Line2D([0], [0], color="blue", lw=2, label="Current field"),
        Line2D([0], [0], color="green", lw=2, alpha=0.7, label="Wind field"),
        Line2D([0], [0], color="purple", lw=2, linestyle="--", label="Communication range"),
        Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=10, label="Base"),
        Line2D([0], [0], marker="o", markerfacecolor="tomato", markeredgecolor="darkred", linestyle="None", markersize=8, label="Obstacle"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    out_path = runs_dir / "environment_preview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"PREVIEW={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
