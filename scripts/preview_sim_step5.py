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

from sim.config import SimConfig
from sim.simulator import Simulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step5 simulator preview.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-end", type=float, default=1200.0)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-every", type=float, default=None)
    return parser.parse_args()


def _render_snapshot(sim: Simulator, t: float, out_path: Path) -> None:
    region_info = sim.region_map.region_info_map(t)
    cfg = sim.config
    env = sim.env

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(
        region_info,
        origin="lower",
        extent=[0.0, cfg.map_width, 0.0, cfg.map_height],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    for y in (cfg.nearshore_y_max, cfg.risk_zone_y_max):
        ax.axhline(y=y, color="white", linestyle="--", linewidth=1.0, alpha=0.8)

    for ob in env.obstacles:
        ax.add_patch(
            Circle((ob.center_x, ob.center_y), ob.radius, facecolor="tomato", edgecolor="darkred", alpha=0.6)
        )

    for p0, p1 in sim.assigned_task_links():
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", alpha=0.25, linewidth=0.8)

    pending = sim.pending_tasks()
    if pending:
        px = [t.target_pos[0] for t in pending]
        py = [t.target_pos[1] for t in pending]
        ax.scatter(px, py, s=18, c="yellow", alpha=0.8, marker="o", label="pending tasks")

    uav_x = [a.pos[0] for a in sim.agents if a.agent_type == "UAV"]
    uav_y = [a.pos[1] for a in sim.agents if a.agent_type == "UAV"]
    usv_x = [a.pos[0] for a in sim.agents if a.agent_type == "USV"]
    usv_y = [a.pos[1] for a in sim.agents if a.agent_type == "USV"]
    if uav_x:
        ax.scatter(uav_x, uav_y, s=70, c="deepskyblue", marker="^", edgecolors="black", linewidths=0.5, label="UAV")
    if usv_x:
        ax.scatter(usv_x, usv_y, s=65, c="orange", marker="s", edgecolors="black", linewidths=0.5, label="USV")

    bx, by = env.base_position
    ax.scatter([bx], [by], s=180, c="black", marker="*", label="Base")
    ax.set_xlim(0.0, cfg.map_width)
    ax.set_ylim(0.0, cfg.map_height)
    ax.set_title(f"Step5 Snapshot (t={t:.1f}s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("region info")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_curves(history: list[dict[str, float]], out_path: Path) -> None:
    times = [row["time"] for row in history]
    pending = [row["pending_count"] for row in history]
    mean_all = [row["mean_info_all"] for row in history]
    min_v = [row["min_info_visited"] for row in history]
    p5_v = [row["p5_info_visited"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, pending, label="pending_count", color="tab:orange", linewidth=1.8)
    ax.plot(times, mean_all, label="mean_info_all", color="tab:blue", linewidth=2.0)
    ax.plot(times, min_v, label="min_visited", color="tab:red", linewidth=1.8)
    ax.plot(times, p5_v, label="p5_visited", color="tab:green", linewidth=1.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.set_title("Step5 Curves")
    ax.grid(alpha=0.25)
    ax.legend()
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
    )
    runs_dir = Path(cfg.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    sim = Simulator(cfg)
    save_frames = bool(args.save_frames)
    frame_every = args.frame_every if args.frame_every is not None else cfg.frame_every
    frames_dir = runs_dir / "frames"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    next_frame_time = 0.0

    def on_tick(sim_obj: Simulator, t: float) -> None:
        nonlocal next_frame_time
        if not save_frames:
            return
        if frame_every <= 0.0:
            return
        if t + 1e-9 < next_frame_time:
            return
        frame_path = frames_dir / f"frame_{int(round(t)):06d}.png"
        _render_snapshot(sim_obj, t=t, out_path=frame_path)
        next_frame_time += frame_every

    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt, on_tick=on_tick)

    final_t = result.history[-1]["time"] if result.history else 0.0
    snapshot_path = runs_dir / "step5_snapshot.png"
    curves_path = runs_dir / "step5_curves.png"
    _render_snapshot(sim, t=final_t, out_path=snapshot_path)
    _render_curves(result.history, curves_path)

    print(f"SNAPSHOT={snapshot_path}")
    print(f"CURVES={curves_path}")
    if save_frames:
        print(f"FRAMES_DIR={frames_dir}")
    print(f"PENDING_FINAL={len(result.pending_tasks)}")
    print(f"TRANSITIONS={result.transition_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
