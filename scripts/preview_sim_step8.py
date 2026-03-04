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
    parser = argparse.ArgumentParser(description="Step8 task-layer preview.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", type=str, default="multimetric", choices=["random", "nearest", "priority", "multimetric"])
    parser.add_argument("--t-end", type=float, default=1200.0)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-every", type=float, default=None)
    parser.add_argument("--ablate-softpart", action="store_true")
    parser.add_argument("--ablate-energy", action="store_true")
    parser.add_argument("--ablate-risk", action="store_true")
    return parser.parse_args()


def _draw_softpart(sim: Simulator, ax) -> None:
    layout = sim.usv_softpart_layout()
    if not layout:
        return
    boundaries = sorted({float(item["left"]) for item in layout} | {float(item["right"]) for item in layout})
    for x in boundaries:
        if x <= 0.0 or x >= sim.config.map_width:
            continue
        ax.axvline(x=x, color="white", linestyle=":", linewidth=1.0, alpha=0.5)
    for item in layout:
        ax.scatter(
            [item["center_x"]],
            [item["center_y"]],
            c="white",
            s=28,
            marker="+",
            alpha=0.7,
            linewidths=1.0,
        )


def _draw_common_overlay(sim: Simulator, ax) -> None:
    cfg = sim.config
    env = sim.env
    for y in (cfg.nearshore_y_max, cfg.risk_zone_y_max):
        ax.axhline(y=y, color="white", linestyle="--", linewidth=1.0, alpha=0.8)

    _draw_softpart(sim, ax)

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
        px = [task.target_pos[0] for task in pending]
        py = [task.target_pos[1] for task in pending]
        ax.scatter(px, py, s=16, c="yellow", alpha=0.9, marker="o", label="pending monitor")

    assigned = sim.tasks.get_assigned_tasks(task_type="monitor")
    if assigned:
        ax.scatter(
            [task.target_pos[0] for task in assigned],
            [task.target_pos[1] for task in assigned],
            s=28,
            c="white",
            marker="x",
            alpha=0.9,
            label="assigned monitor",
        )

    recharge_tasks = sim.recharge_tasks()
    if recharge_tasks:
        rx = [task.rendezvous_pos[0] for task in recharge_tasks if task.rendezvous_pos is not None]
        ry = [task.rendezvous_pos[1] for task in recharge_tasks if task.rendezvous_pos is not None]
        if rx:
            ax.scatter(rx, ry, s=52, c="purple", marker="D", alpha=0.85, label="recharge rendezvous")

    uavs = [agent for agent in sim.agents if isinstance(agent, UAVAgent)]
    usvs = [agent for agent in sim.agents if isinstance(agent, USVAgent)]
    if uavs:
        ax.scatter(
            [agent.pos[0] for agent in uavs],
            [agent.pos[1] for agent in uavs],
            s=72,
            c="deepskyblue",
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            label="UAV",
        )
    if usvs:
        ax.scatter(
            [agent.pos[0] for agent in usvs],
            [agent.pos[1] for agent in usvs],
            s=66,
            c="orange",
            marker="s",
            edgecolors="black",
            linewidths=0.5,
            label="USV",
        )

    for agent in uavs + usvs:
        if agent.task_status == "charging":
            ax.text(agent.pos[0] + 40.0, agent.pos[1] + 40.0, "CHG", color="purple", fontsize=8, weight="bold")

    bx, by = env.base_position
    ax.scatter([bx], [by], s=180, c="black", marker="*", label="Base")
    ax.set_xlim(0.0, cfg.map_width)
    ax.set_ylim(0.0, cfg.map_height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)


def _render_snapshot(sim: Simulator, t: float, out_path: Path, title_suffix: str) -> None:
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
    ax.set_title(f"Step8 Snapshot ({title_suffix}, t={t:.1f}s)")
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("region info")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_curves(history: list[dict[str, float]], out_path: Path, title_suffix: str) -> None:
    times = [row["time"] for row in history]
    pending = [row["pending_count"] for row in history]
    assigned = [row["assigned_count"] for row in history]
    done = [row["done_count"] for row in history]
    mean_all = [row["mean_info_all"] for row in history]
    recharge = [row["recharge_count_cum"] for row in history]
    battery_min = [row["uav_battery_min"] for row in history]
    battery_min_hist: list[float] = []
    cur_min = 1.0
    for value in battery_min:
        cur_min = min(cur_min, float(value))
        battery_min_hist.append(cur_min)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, mean_all, label="mean_info_all", color="tab:blue", linewidth=2.0)
    ax.plot(times, pending, label="pending_count", color="tab:orange", linewidth=1.8)
    ax.plot(times, done, label="done_count_cum", color="tab:green", linewidth=1.8)
    ax.plot(times, recharge, label="recharge_count_cum", color="tab:purple", linewidth=1.8)
    ax.plot(times, battery_min_hist, label="uav_battery_min_hist", color="tab:red", linewidth=1.5)
    ax.plot(times, assigned, label="assigned_count", color="tab:brown", linewidth=1.3)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.set_title(f"Step8 Curves ({title_suffix})")
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
        strategy=args.policy,
        task_policy=args.policy,
        ablate_softpart=bool(args.ablate_softpart),
        ablate_energy_term=bool(args.ablate_energy),
        ablate_risk_term=bool(args.ablate_risk),
    )
    runs_dir = Path(cfg.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    tag_parts = [args.policy]
    if args.ablate_softpart:
        tag_parts.append("nosoft")
    if args.ablate_energy:
        tag_parts.append("noenergy")
    if args.ablate_risk:
        tag_parts.append("norisk")
    tag = "_".join(tag_parts)

    sim = Simulator(cfg)
    save_frames = bool(args.save_frames)
    frame_every = args.frame_every if args.frame_every is not None else cfg.frame_every
    frames_dir = runs_dir / f"frames_step8_{tag}"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    next_frame_time = 0.0

    def on_tick(sim_obj: Simulator, t: float) -> None:
        nonlocal next_frame_time
        if not save_frames or frame_every <= 0.0:
            return
        if t + 1e-9 < next_frame_time:
            return
        frame_path = frames_dir / f"step8_{tag}_{int(round(t)):06d}.png"
        _render_snapshot(sim_obj, t=t, out_path=frame_path, title_suffix=tag)
        next_frame_time += frame_every

    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt, on_tick=on_tick)

    final_t = result.history[-1]["time"] if result.history else 0.0
    snapshot_path = runs_dir / f"step8_snapshot_{tag}.png"
    curves_path = runs_dir / f"step8_curves_{tag}.png"
    _render_snapshot(sim, t=final_t, out_path=snapshot_path, title_suffix=tag)
    _render_curves(result.history, out_path=curves_path, title_suffix=tag)

    final = result.history[-1] if result.history else {}
    min_hist = min((row["uav_battery_min"] for row in result.history), default=0.0)
    hit_rate = sim.usv_preference_hit_rate()

    print(f"POLICY={args.policy}")
    print(f"SNAPSHOT={snapshot_path}")
    print(f"CURVES={curves_path}")
    if save_frames:
        print(f"FRAMES_DIR={frames_dir}")
    print(
        "METRICS "
        f"mean_info_all_end={final.get('mean_info_all', 0.0):.4f} "
        f"done_count={int(final.get('done_count', 0.0))} "
        f"pending_count_end={int(final.get('pending_count', 0.0))} "
        f"recharge_count_cum={int(final.get('recharge_count_cum', 0.0))} "
        f"uav_dead_count={int(final.get('uav_dead_count', 0.0))} "
        f"uav_battery_min_hist={min_hist:.4f} "
        f"uav_battery_mean={final.get('uav_battery_mean', 0.0):.4f} "
        f"usv_preference_hit_rate={hit_rate:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
