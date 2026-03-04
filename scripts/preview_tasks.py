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
from sim.coverage.coverage_grid import CoverageGrid
from sim.environment.environment import Environment2D
from sim.tasks.region_map import RegionMap
from sim.tasks.task_generator import TaskGenerator


def print_task_status(current_time: float, pending_tasks) -> None:
    print("---- TASK STATUS ----")
    print("time:", round(current_time, 3))
    print("pending tasks:", len(pending_tasks))
    sorted_tasks = sorted(pending_tasks, key=lambda t: t.priority, reverse=True)
    for task in sorted_tasks[:5]:
        print("region:", task.region_id, "priority:", round(task.priority, 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview Step4 region tasks.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=30.0, help="seconds")
    parser.add_argument("--runs-dir", type=str, default="runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SimConfig(seed=args.seed, runs_dir=args.runs_dir)
    runs_dir = Path(config.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    env = Environment2D(config)
    grid = CoverageGrid(config)
    region_map = RegionMap(grid, config)
    task_gen = TaskGenerator(region_map, config)
    simulation_time = args.steps * args.dt
    checkpoint_times = {0.0, simulation_time * 0.5, simulation_time}
    printed_keys: set[int] = set()

    for step in range(args.steps):
        t = step * args.dt

        for uav_idx in range(config.num_uav):
            region = "offshore" if (step + uav_idx) % 2 == 0 else "risk_zone"
            p = env.sample_point(region)
            grid.observe(position=p, radius=config.uav_sensor_radius, t=t)

        for usv_idx in range(config.num_usv):
            region = "nearshore" if usv_idx < 2 else "risk_zone"
            p = env.sample_point(region)
            grid.observe(position=p, radius=config.usv_sensor_radius, t=t)

        task_gen.generate_tasks(t)
        pending = task_gen.get_pending_tasks()
        should_print_interval = abs((t % 600.0)) < 1e-9
        should_print_checkpoint = any(abs(t - ct) <= (args.dt * 0.5) for ct in checkpoint_times)
        if should_print_interval or should_print_checkpoint:
            time_key = int(round(t * 1000))
            if time_key not in printed_keys:
                print_task_status(t, pending)
                printed_keys.add(time_key)

    # Ensure status print at exact simulation end time.
    end_key = int(round(simulation_time * 1000))
    if end_key not in printed_keys:
        print_task_status(simulation_time, task_gen.get_pending_tasks())
        printed_keys.add(end_key)

    final_t = (args.steps - 1) * args.dt if args.steps > 0 else 0.0
    region_info = region_map.region_info_map(final_t)

    # Region info heatmap
    region_heatmap_path = runs_dir / "region_info_heatmap.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(
        region_info,
        origin="lower",
        extent=[0.0, config.map_width, 0.0, config.map_height],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    # Optional region grid lines for readability.
    for x in range(1, region_map.nrx):
        ax.axvline(x * config.region_cell_size, color="white", linewidth=0.4, alpha=0.3)
    for y in range(1, region_map.nry):
        ax.axhline(y * config.region_cell_size, color="white", linewidth=0.4, alpha=0.3)
    ax.set_title(f"Region Info Heatmap (t={final_t:.1f}s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("region info")
    fig.tight_layout()
    fig.savefig(region_heatmap_path, dpi=150)
    plt.close(fig)

    # Pending tasks scatter plot
    pending_task_path = runs_dir / "pending_tasks.png"
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.set_xlim(0.0, config.map_width)
    ax2.set_ylim(0.0, config.map_height)
    ax2.set_title("Pending Monitor Tasks")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.grid(alpha=0.2)

    pending = task_gen.get_pending_tasks()
    mean_priority = (
        sum(task.priority for task in pending) / len(pending) if pending else 0.0
    )
    if pending:
        xs = [task.target_pos[0] for task in pending]
        ys = [task.target_pos[1] for task in pending]
        priorities = [task.priority for task in pending]
        sizes = [80.0 + 120.0 * p for p in priorities]
        sc = ax2.scatter(xs, ys, c=priorities, s=sizes, cmap="plasma", alpha=0.85, edgecolors="black", linewidths=0.3)
        cbar2 = fig2.colorbar(sc, ax=ax2)
        cbar2.set_label("task priority")
    ax2.set_title(f"Pending Monitor Tasks (N={len(pending)}, mean_priority={mean_priority:.3f})")
    fig2.tight_layout()
    fig2.savefig(pending_task_path, dpi=150)
    plt.close(fig2)

    print(f"REGION_HEATMAP={region_heatmap_path}")
    print(f"PENDING_TASKS={pending_task_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
