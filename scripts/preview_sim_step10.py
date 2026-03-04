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
    parser = argparse.ArgumentParser(description="Step10 failure+robustness preview")
    parser.add_argument("--policy", type=str, default="multimetric", choices=["random", "nearest", "priority", "multimetric"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=1200.0)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--failure-kind", type=str, default="DISABLED", choices=["DISABLED", "DAMAGED"])
    parser.add_argument("--failure-t", type=float, default=600.0)
    parser.add_argument("--failure-usv", type=int, default=1)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-every", type=float, default=None)
    return parser.parse_args()


def _draw_common(sim: Simulator, ax) -> None:
    cfg = sim.config
    env = sim.env

    for y in (cfg.nearshore_y_max, cfg.risk_zone_y_max):
        ax.axhline(y=y, color="white", linestyle="--", linewidth=1.0, alpha=0.8)

    for ob in env.obstacles:
        ax.add_patch(Circle((ob.center_x, ob.center_y), ob.radius, facecolor="tomato", edgecolor="darkred", alpha=0.55))

    pending = sim.pending_tasks()
    if pending:
        ax.scatter([t.target_pos[0] for t in pending], [t.target_pos[1] for t in pending], s=16, c="yellow", alpha=0.9, label="pending")

    assigned = sim.tasks.get_assigned_tasks(task_type="monitor")
    if assigned:
        ax.scatter(
            [t.target_pos[0] for t in assigned],
            [t.target_pos[1] for t in assigned],
            s=28,
            c="white",
            marker="x",
            alpha=0.9,
            label="assigned",
        )

    uavs = [a for a in sim.agents if isinstance(a, UAVAgent)]
    usvs = [a for a in sim.agents if isinstance(a, USVAgent)]
    if uavs:
        ax.scatter([a.pos[0] for a in uavs], [a.pos[1] for a in uavs], s=72, c="deepskyblue", marker="^", edgecolors="black", linewidths=0.5, label="UAV")

    ok_usv = [u for u in usvs if u.health_state == "OK"]
    dmg_usv = [u for u in usvs if u.health_state == "DAMAGED"]
    dis_usv = [u for u in usvs if u.health_state == "DISABLED"]

    if ok_usv:
        ax.scatter([u.pos[0] for u in ok_usv], [u.pos[1] for u in ok_usv], s=66, c="orange", marker="s", edgecolors="black", linewidths=0.5, label="USV OK")
    if dmg_usv:
        ax.scatter([u.pos[0] for u in dmg_usv], [u.pos[1] for u in dmg_usv], s=72, c="gold", marker="s", edgecolors="black", linewidths=0.5, label="USV DAMAGED")
        for u in dmg_usv:
            ax.text(u.pos[0] + 40.0, u.pos[1] + 40.0, "DAMAGED", color="goldenrod", fontsize=8, weight="bold")
    if dis_usv:
        ax.scatter([u.pos[0] for u in dis_usv], [u.pos[1] for u in dis_usv], s=80, c="gray", marker="X", edgecolors="black", linewidths=0.5, label="USV DISABLED")
        for u in dis_usv:
            ax.text(u.pos[0] + 40.0, u.pos[1] + 40.0, "DISABLED", color="gray", fontsize=8, weight="bold")

    bx, by = env.base_position
    ax.scatter([bx], [by], s=180, c="black", marker="*", label="Base")

    ax.set_xlim(0.0, cfg.map_width)
    ax.set_ylim(0.0, cfg.map_height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)


def _render_snapshot(sim: Simulator, t: float, out_path: Path, title: str) -> None:
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
    _draw_common(sim, ax)

    status_lines: list[str] = []
    if sim._failure_forced_relax_active(t):
        status_lines.append("SOFTPART FORCED RELAX")
    if sim._feedback_relax_active(t):
        status_lines.append("FEEDBACK RELAX")
    if sim._feedback_recharge_boost_active(t):
        status_lines.append("RECHARGE BOOST")
    if sim.failure_events:
        last_fail = sim.failure_events[-1]
        status_lines.append(f"FAIL: t={last_fail.get('t'):.0f} {last_fail.get('usv_id')} {last_fail.get('kind')}")

    if status_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(status_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 6, "edgecolor": "none"},
            color="white",
        )

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("region info")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_curves(history: list[dict[str, float]], fail_t: float, out_path: Path, title: str) -> None:
    times = [row["time"] for row in history]
    mean_info = [row["mean_info_all"] for row in history]
    pending = [row["pending_count"] for row in history]
    done = [row["done_count"] for row in history]
    recharge = [row["recharge_count_cum"] for row in history]
    fb_count = [row.get("fb_trigger_count_cum", 0.0) for row in history]
    disabled = [row.get("num_usv_disabled", 0.0) for row in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, mean_info, label="mean_info_all", color="tab:blue", linewidth=2.0)
    ax.plot(times, pending, label="pending_count", color="tab:orange", linewidth=1.6)
    ax.plot(times, done, label="done_count_cum", color="tab:green", linewidth=1.6)
    ax.plot(times, recharge, label="recharge_count_cum", color="tab:purple", linewidth=1.6)
    ax.plot(times, fb_count, label="fb_trigger_count_cum", color="tab:red", linewidth=1.4)
    ax.plot(times, disabled, label="num_usv_disabled", color="black", linewidth=1.4)
    ax.axvline(fail_t, color="magenta", linestyle="--", linewidth=1.2, alpha=0.7, label="failure_t")

    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = SimConfig(
        seed=args.seed,
        strategy=args.policy,
        task_policy=args.policy,
        t_end=args.duration,
        sim_dt=args.dt,
        runs_dir=args.runs_dir,
        enable_feedback=True,
        enable_failures=True,
        failure_mode="scheduled",
        failure_t_sec=args.failure_t,
        failure_usv_id=args.failure_usv,
        failure_kind=args.failure_kind,
        enable_robust_response=True,
        forced_relax_on_failure=True,
    )

    runs_dir = Path(cfg.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sim = Simulator(cfg)

    save_frames = bool(args.save_frames)
    frame_every = args.frame_every if args.frame_every is not None else cfg.frame_every
    frames_dir = runs_dir / "frames_step10"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    next_frame_t = 0.0

    def on_tick(sim_obj: Simulator, t: float) -> None:
        nonlocal next_frame_t
        if not save_frames or frame_every <= 0.0:
            return
        if t + 1e-9 < next_frame_t:
            return
        out = frames_dir / f"step10_{int(round(t)):06d}.png"
        _render_snapshot(sim_obj, t, out, f"Step10 Snapshot (t={t:.1f}s)")
        next_frame_t += frame_every

    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt, on_tick=on_tick)
    final_t = result.history[-1]["time"] if result.history else 0.0

    curves_path = runs_dir / "step10_curves.png"
    snapshot_path = runs_dir / "step10_snapshot.png"

    _render_curves(result.history, cfg.failure_t_sec, curves_path, "Step10 Curves")
    _render_snapshot(sim, final_t, snapshot_path, f"Step10 Snapshot (t={final_t:.1f}s)")

    final = result.history[-1] if result.history else {}
    print(f"CURVES={curves_path}")
    print(f"SNAPSHOT={snapshot_path}")
    print(
        "METRICS "
        f"mean_end={final.get('mean_info_all', 0.0):.4f} "
        f"done={int(final.get('done_count', 0.0))} "
        f"pending_end={int(final.get('pending_count', 0.0))} "
        f"recharge={int(final.get('recharge_count_cum', 0.0))} "
        f"fb_trigger={int(final.get('fb_trigger_count_cum', 0.0))} "
        f"disabled={int(final.get('num_usv_disabled', 0.0))} "
        f"dead={int(final.get('uav_dead_count', 0.0))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
