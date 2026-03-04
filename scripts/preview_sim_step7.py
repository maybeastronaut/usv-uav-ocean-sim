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
from matplotlib.patches import Circle

from sim.agents.uav import UAVAgent
from sim.agents.usv import USVAgent
from sim.config import SimConfig
from sim.simulator import Simulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step7 simulator preview (energy + recharge).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", type=str, default="nearest", choices=["random", "nearest", "priority"])
    parser.add_argument("--t-end", type=float, default=1200.0)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-every", type=float, default=None)
    return parser.parse_args()


def _draw_common_overlay(sim: Simulator, ax) -> None:
    cfg = sim.config
    env = sim.env
    for y in (cfg.nearshore_y_max, cfg.risk_zone_y_max):
        ax.axhline(y=y, color="white", linestyle="--", linewidth=1.0, alpha=0.8)

    for ob in env.obstacles:
        ax.add_patch(Circle((ob.center_x, ob.center_y), ob.radius, facecolor="tomato", edgecolor="darkred", alpha=0.6))

    for agent in sim.agents:
        if agent.goal_pos is None:
            continue
        path = [agent.pos]
        if agent.current_waypoints:
            idx = min(agent.current_wp_idx, len(agent.current_waypoints) - 1)
            path.extend(agent.current_waypoints[idx:])
        if not path or path[-1] != agent.goal_pos:
            path.append(agent.goal_pos)
        if len(path) < 2:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="white", alpha=0.35, linewidth=0.8)

    for p0, p1 in sim.assigned_task_links():
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="yellow", alpha=0.25, linewidth=0.8)

    for uav_pos, usv_pos, rv in sim.recharge_links():
        ax.plot([uav_pos[0], rv[0]], [uav_pos[1], rv[1]], color="purple", alpha=0.35, linewidth=1.0)
        ax.plot([usv_pos[0], rv[0]], [usv_pos[1], rv[1]], color="purple", alpha=0.35, linewidth=1.0)

    pending = sim.pending_tasks()
    if pending:
        px = [t.target_pos[0] for t in pending]
        py = [t.target_pos[1] for t in pending]
        ax.scatter(px, py, s=16, c="yellow", alpha=0.9, marker="o", label="pending monitor")

    assigned = sim.tasks.get_assigned_tasks(task_type="monitor")
    if assigned:
        ax.scatter(
            [t.target_pos[0] for t in assigned],
            [t.target_pos[1] for t in assigned],
            s=28,
            c="white",
            marker="x",
            alpha=0.9,
            label="assigned monitor",
        )

    recharge_tasks = sim.recharge_tasks()
    if recharge_tasks:
        rx = [t.rendezvous_pos[0] for t in recharge_tasks if t.rendezvous_pos is not None]
        ry = [t.rendezvous_pos[1] for t in recharge_tasks if t.rendezvous_pos is not None]
        if rx:
            ax.scatter(rx, ry, s=54, c="purple", marker="D", alpha=0.85, label="recharge rendezvous")

    uav = [a for a in sim.agents if isinstance(a, UAVAgent)]
    usv = [a for a in sim.agents if isinstance(a, USVAgent)]
    if uav:
        ax.scatter(
            [a.pos[0] for a in uav],
            [a.pos[1] for a in uav],
            s=72,
            c="deepskyblue",
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            label="UAV",
        )
    if usv:
        ax.scatter(
            [a.pos[0] for a in usv],
            [a.pos[1] for a in usv],
            s=66,
            c="orange",
            marker="s",
            edgecolors="black",
            linewidths=0.5,
            label="USV",
        )

    for a in uav + usv:
        if a.task_status == "charging":
            ax.text(a.pos[0] + 40.0, a.pos[1] + 40.0, "CHG", color="purple", fontsize=8, weight="bold")

    bx, by = env.base_position
    ax.scatter([bx], [by], s=180, c="black", marker="*", label="Base")
    ax.set_xlim(0.0, cfg.map_width)
    ax.set_ylim(0.0, cfg.map_height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)


def _render_snapshot(sim: Simulator, t: float, out_path: Path) -> None:
    region_info = sim.region_map.region_info_map(t)
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(
        region_info,
        origin="lower",
        extent=[0.0, sim.config.map_width, 0.0, sim.config.map_height],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    _draw_common_overlay(sim, ax)
    ax.set_title(f"Step7 Snapshot ({sim.config.strategy}, t={t:.1f}s)")
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("region info")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_curves(history: list[dict[str, float]], out_path: Path, strategy: str) -> None:
    times = [row["time"] for row in history]
    pending = [row["pending_count"] for row in history]
    done = [row["done_count"] for row in history]
    mean_all = [row["mean_info_all"] for row in history]
    recharge = [row["recharge_count_cum"] for row in history]
    battery_min = [row["uav_battery_min"] for row in history]
    battery_mean = [row["uav_battery_mean"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, pending, label="pending_count", color="tab:orange", linewidth=1.8)
    ax.plot(times, done, label="done_count_cum", color="tab:green", linewidth=1.8)
    ax.plot(times, mean_all, label="mean_info_all", color="tab:blue", linewidth=2.0)
    ax.plot(times, recharge, label="recharge_count_cum", color="tab:purple", linewidth=1.8)
    ax.plot(times, battery_min, label="uav_battery_min", color="tab:red", linewidth=1.4)
    ax.plot(times, battery_mean, label="uav_battery_mean", color="tab:pink", linewidth=1.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.set_title(f"Step7 Curves ({strategy})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = SimConfig(
        seed=args.seed,
        runs_dir=args.runs_dir,
        sim_dt=args.dt,
        t_end=args.t_end,
        strategy=args.strategy,
    )
    runs_dir = Path(cfg.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    sim = Simulator(cfg)
    save_frames = bool(args.save_frames)
    frame_every = args.frame_every if args.frame_every is not None else cfg.frame_every
    frames_dir = runs_dir / "frames_step7"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    next_frame_time = 0.0

    def on_tick(sim_obj: Simulator, t: float) -> None:
        nonlocal next_frame_time
        if not save_frames or frame_every <= 0.0:
            return
        if t + 1e-9 < next_frame_time:
            return
        frame_path = frames_dir / f"step7_{cfg.strategy}_{int(round(t)):06d}.png"
        _render_snapshot(sim_obj, t=t, out_path=frame_path)
        next_frame_time += frame_every

    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt, on_tick=on_tick)
    final_t = result.history[-1]["time"] if result.history else 0.0
    snapshot_path = runs_dir / "step7_snapshot.png"
    curves_path = runs_dir / "step7_curves.png"
    _render_snapshot(sim, t=final_t, out_path=snapshot_path)
    _render_curves(result.history, curves_path, strategy=cfg.strategy)

    last = result.history[-1] if result.history else {}
    print(f"STRATEGY={cfg.strategy}")
    print(f"SNAPSHOT={snapshot_path}")
    print(f"CURVES={curves_path}")
    if save_frames:
        print(f"FRAMES_DIR={frames_dir}")
    print(
        "METRICS "
        f"mean_info_all_end={last.get('mean_info_all', 0.0):.4f} "
        f"done_count={int(last.get('done_count', 0.0))} "
        f"pending_count_end={int(last.get('pending_count', 0.0))} "
        f"recharge_count_cum={int(last.get('recharge_count_cum', 0.0))} "
        f"uav_battery_min={last.get('uav_battery_min', 0.0):.4f} "
        f"uav_battery_mean={last.get('uav_battery_mean', 0.0):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
