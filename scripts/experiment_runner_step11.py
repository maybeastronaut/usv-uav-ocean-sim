from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import SimConfig
from sim.simulator import Simulator


RESULT_FIELDS = [
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
    "total_distance_all",
    "total_energy_usv",
    "total_energy_uav",
    "total_energy_all",
    "recharge_count",
    "uav_dead_count",
    "fb_trigger_count",
    "baseline_pre",
    "threshold_recover",
    "recovered",
    "recovery_time_sec",
    "mean_min_after_failure",
    "info_per_distance",
    "done_per_distance",
]

NUMERIC_SUMMARY_FIELDS = [
    "mean_info_all_end",
    "mean_info_all_min",
    "mean_info_all_p5",
    "done_count",
    "pending_end",
    "total_distance_usv",
    "total_distance_uav",
    "total_distance_all",
    "total_energy_usv",
    "total_energy_uav",
    "total_energy_all",
    "recharge_count",
    "uav_dead_count",
    "fb_trigger_count",
    "baseline_pre",
    "threshold_recover",
    "recovery_time_sec",
    "mean_min_after_failure",
    "info_per_distance",
    "done_per_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step11 batch experiment runner")
    parser.add_argument("--outdir", type=str, default="runs/step11")
    parser.add_argument("--exp", type=str, default="all", choices=["exp1", "exp2", "exp3", "all"])
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--parallel", type=int, default=0, choices=[0, 1])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--low-load", action="store_true")
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--timeseries-stride", type=int, default=1)
    parser.add_argument("--energy-proxy-k", type=float, default=1.0)
    parser.add_argument("--recovery-pre-window", type=float, default=300.0)
    parser.add_argument("--recovery-hold-sec", type=float, default=120.0)
    parser.add_argument("--recovery-ratio", type=float, default=0.95)
    return parser.parse_args()


def parse_seed_list(seed_str: str) -> list[int] | None:
    txt = seed_str.strip()
    if not txt:
        return None
    return [int(part.strip()) for part in txt.split(",") if part.strip()]


def build_cases(exp: str, seeds_override: list[int] | None, duration_override: float | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add_case(**kwargs: Any) -> None:
        cases.append(kwargs)

    if exp in ("exp1", "all"):
        seeds = seeds_override if seeds_override is not None else [0, 1, 2, 3, 4]
        duration = duration_override if duration_override is not None else 1800.0
        for policy in ("random", "nearest", "priority", "multimetric"):
            for seed in seeds:
                add_case(
                    exp_name="exp1",
                    policy=policy,
                    seed=seed,
                    duration=duration,
                    enable_feedback=True,
                    enable_failures=False,
                    enable_robust_response=False,
                    failure_mode="scheduled",
                    failure_kind="",
                    failure_t="",
                    failure_usv="",
                )

    if exp in ("exp2", "all"):
        seeds = seeds_override if seeds_override is not None else [0, 1, 2, 3, 4]
        duration = duration_override if duration_override is not None else 1800.0
        for robust in (False, True):
            for seed in seeds:
                add_case(
                    exp_name="exp2",
                    policy="multimetric",
                    seed=seed,
                    duration=duration,
                    enable_feedback=True,
                    enable_failures=True,
                    enable_robust_response=robust,
                    failure_mode="scheduled",
                    failure_kind="DISABLED",
                    failure_t=600.0,
                    failure_usv=1,
                )

        weak_seed = seeds[0]
        add_case(
            exp_name="exp2",
            policy="nearest",
            seed=weak_seed,
            duration=duration,
            enable_feedback=True,
            enable_failures=True,
            enable_robust_response=False,
            failure_mode="scheduled",
            failure_kind="DISABLED",
            failure_t=600.0,
            failure_usv=1,
        )

    if exp in ("exp3", "all"):
        seeds = seeds_override if seeds_override is not None else [0, 1, 2]
        duration = duration_override if duration_override is not None else 3600.0
        for seed in seeds:
            add_case(
                exp_name="exp3",
                policy="multimetric",
                seed=seed,
                duration=duration,
                enable_feedback=True,
                enable_failures=False,
                enable_robust_response=True,
                failure_mode="scheduled",
                failure_kind="",
                failure_t="",
                failure_usv="",
            )

    return cases


def _sample_indices(n: int, stride: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    step = max(1, int(stride))
    idx = np.arange(0, n, step, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _compute_recovery_metrics(
    history: list[dict[str, float]],
    fail_t: float,
    duration: float,
    *,
    pre_window: float,
    hold_sec: float,
    recover_ratio: float,
) -> dict[str, Any]:
    times = np.array([row["time"] for row in history], dtype=float)
    means = np.array([row["mean_info_all"] for row in history], dtype=float)
    if times.size == 0:
        return {
            "baseline_pre": 0.0,
            "threshold_recover": 0.0,
            "recovered": False,
            "recovery_time_sec": max(0.0, duration - fail_t),
        }

    pre_mask = (times >= fail_t - pre_window) & (times < fail_t)
    pre_vals = means[pre_mask]
    if pre_vals.size == 0:
        older = means[times < fail_t]
        pre_vals = older[-5:] if older.size > 0 else means[:1]

    baseline_pre = float(np.mean(pre_vals)) if pre_vals.size > 0 else float(means[0])
    threshold = float(recover_ratio * baseline_pre)

    recovered = False
    recovery_time = max(0.0, duration - fail_t)

    for i, t0 in enumerate(times):
        if t0 <= fail_t + 1e-9:
            continue
        tend = t0 + hold_sec
        if tend > times[-1] + 1e-9:
            break
        hold_mask = (times >= t0 - 1e-9) & (times <= tend + 1e-9)
        if not np.any(hold_mask):
            continue
        if np.all(means[hold_mask] >= threshold):
            recovered = True
            recovery_time = float(max(0.0, t0 - fail_t))
            break

    return {
        "baseline_pre": baseline_pre,
        "threshold_recover": threshold,
        "recovered": recovered,
        "recovery_time_sec": recovery_time,
    }


def _series_min_after_failure(history: list[dict[str, float]], fail_t: float, window: float = 200.0) -> float:
    vals = [row["mean_info_all"] for row in history if fail_t <= row["time"] <= fail_t + window]
    if not vals:
        return 0.0
    return float(min(vals))


def _timeseries_filename(case: dict[str, Any]) -> str:
    robust = int(bool(case["enable_robust_response"]))
    fail = int(bool(case["enable_failures"]))
    return (
        f"{case['exp_name']}__{case['policy']}__seed{case['seed']}__dur{int(case['duration'])}"
        f"__fail{fail}__rob{robust}.npz"
    )


def run_single_case(
    case: dict[str, Any],
    outdir: str,
    *,
    sim_dt: float,
    timeseries_stride: int,
    energy_proxy_k: float,
    recovery_pre_window: float,
    recovery_hold_sec: float,
    recovery_ratio: float,
) -> dict[str, Any]:
    cfg = SimConfig(
        seed=int(case["seed"]),
        t_end=float(case["duration"]),
        sim_dt=float(sim_dt),
        strategy=str(case["policy"]),
        task_policy=str(case["policy"]),
        enable_feedback=bool(case["enable_feedback"]),
        enable_failures=bool(case["enable_failures"]),
        enable_robust_response=bool(case["enable_robust_response"]),
        forced_relax_on_failure=bool(case["enable_robust_response"]),
        failure_mode=str(case["failure_mode"]),
        failure_kind=str(case["failure_kind"] or "DISABLED"),
        failure_t_sec=float(case["failure_t"] or 600.0),
        failure_usv_id=int(case["failure_usv"] or 1),
        runs_dir=outdir,
    )

    sim = Simulator(cfg)
    result = sim.run(t_end=cfg.t_end, dt=cfg.sim_dt)
    if not result.history:
        raise RuntimeError("empty simulation history")

    history = result.history
    final = history[-1]
    mean_series = np.array([row["mean_info_all"] for row in history], dtype=float)

    baseline_pre_val: float | str = ""
    threshold_recover_val: float | str = ""
    recovered_val: int | str = ""
    recovery_val: float | str = ""
    mean_after_fail_val: float | str = ""
    if bool(case["enable_failures"]):
        fail_t = float(case["failure_t"] or 600.0)
        recovery = _compute_recovery_metrics(
            history,
            fail_t,
            float(case["duration"]),
            pre_window=recovery_pre_window,
            hold_sec=recovery_hold_sec,
            recover_ratio=recovery_ratio,
        )
        mean_after_fail = _series_min_after_failure(history, fail_t, window=200.0)
        baseline_pre_val = float(recovery["baseline_pre"])
        threshold_recover_val = float(recovery["threshold_recover"])
        recovered_val = int(bool(recovery["recovered"]))
        recovery_val = float(recovery["recovery_time_sec"])
        mean_after_fail_val = float(mean_after_fail)

    total_distance_uav = float(sum(agent.stats.get("distance", 0.0) for agent in sim._uavs()))
    total_distance_usv = float(sum(agent.stats.get("distance", 0.0) for agent in sim._usvs()))
    total_distance_all = float(total_distance_uav + total_distance_usv)

    # Step12: if no explicit energy model is exposed here, use distance-based proxy.
    e_k = float(energy_proxy_k)
    total_energy_uav = float(total_distance_uav * e_k)
    total_energy_usv = float(total_distance_usv * e_k)
    total_energy_all = float(total_distance_all * e_k)

    eps = 1e-9
    info_per_distance = float(final["mean_info_all"]) / (total_distance_all + eps)
    done_per_distance = float(final["done_count"]) / (total_distance_all + eps)

    outdir_path = Path(outdir)
    ts_dir = outdir_path / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts_path = ts_dir / _timeseries_filename(case)

    idx = _sample_indices(len(history), timeseries_stride)
    time_arr = np.array([history[i]["time"] for i in idx], dtype=float)
    mean_arr = np.array([history[i]["mean_info_all"] for i in idx], dtype=float)
    pending_arr = np.array([history[i]["pending_count"] for i in idx], dtype=float)
    done_arr = np.array([history[i]["done_count"] for i in idx], dtype=float)
    recharge_arr = np.array([history[i]["recharge_count_cum"] for i in idx], dtype=float)
    fb_arr = np.array([history[i].get("fb_trigger_count_cum", 0.0) for i in idx], dtype=float)
    disabled_arr = np.array([history[i].get("num_usv_disabled", 0.0) for i in idx], dtype=float)

    np.savez_compressed(
        ts_path,
        time=time_arr,
        mean_info_all=mean_arr,
        pending_count=pending_arr,
        done_count=done_arr,
        recharge_count=recharge_arr,
        fb_trigger_count=fb_arr,
        num_usv_disabled=disabled_arr,
        exp_name=np.array(str(case["exp_name"])),
        policy=np.array(str(case["policy"])),
        seed=np.array(int(case["seed"])),
        enable_robust_response=np.array(int(bool(case["enable_robust_response"]))),
        failure_t=np.array(float(case["failure_t"] or -1.0)),
        baseline_pre=np.array(float(baseline_pre_val) if baseline_pre_val != "" else np.nan),
        threshold_recover=np.array(float(threshold_recover_val) if threshold_recover_val != "" else np.nan),
        recovered=np.array(int(recovered_val) if recovered_val != "" else -1),
        recovery_time_sec=np.array(float(recovery_val) if recovery_val != "" else np.nan),
    )

    row = {
        "exp_name": case["exp_name"],
        "policy": case["policy"],
        "seed": int(case["seed"]),
        "duration": float(case["duration"]),
        "enable_feedback": int(bool(case["enable_feedback"])),
        "enable_failures": int(bool(case["enable_failures"])),
        "enable_robust_response": int(bool(case["enable_robust_response"])),
        "failure_kind": case["failure_kind"],
        "failure_t": case["failure_t"],
        "failure_usv": case["failure_usv"],
        "mean_info_all_end": float(final["mean_info_all"]),
        "mean_info_all_min": float(np.min(mean_series)),
        "mean_info_all_p5": float(np.percentile(mean_series, 5)),
        "done_count": int(final["done_count"]),
        "pending_end": int(final["pending_count"]),
        "total_distance_usv": total_distance_usv,
        "total_distance_uav": total_distance_uav,
        "total_distance_all": total_distance_all,
        "total_energy_usv": total_energy_usv,
        "total_energy_uav": total_energy_uav,
        "total_energy_all": total_energy_all,
        "recharge_count": int(final.get("recharge_count_cum", 0.0)),
        "uav_dead_count": int(final.get("uav_dead_count", 0.0)),
        "fb_trigger_count": int(final.get("fb_trigger_count_cum", 0.0)),
        "baseline_pre": baseline_pre_val,
        "threshold_recover": threshold_recover_val,
        "recovered": recovered_val,
        "recovery_time_sec": recovery_val,
        "mean_min_after_failure": mean_after_fail_val,
        "info_per_distance": info_per_distance,
        "done_per_distance": done_per_distance,
    }
    return row


def write_results_csv(rows: list[dict[str, Any]], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["exp_name"],
            row["policy"],
            int(row["enable_robust_response"]),
            int(row["enable_failures"]),
            float(row["duration"]),
        )
        groups[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        exp_name, policy, robust, failures, duration = key
        out: dict[str, Any] = {
            "exp_name": exp_name,
            "policy": policy,
            "enable_robust_response": robust,
            "enable_failures": failures,
            "duration": duration,
            "n_runs": len(items),
        }

        recovered_values = [v for v in (_to_float(item.get("recovered")) for item in items) if v is not None]
        out["recovered_rate"] = float(np.mean(np.array(recovered_values, dtype=float))) if recovered_values else ""

        for metric in NUMERIC_SUMMARY_FIELDS:
            values = [v for v in (_to_float(item.get(metric)) for item in items) if v is not None]
            if not values:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
                out[f"{metric}_sem"] = ""
                out[f"{metric}_ci95"] = ""
                continue
            arr = np.array(values, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            sem = std / math.sqrt(arr.size) if arr.size > 0 else 0.0
            ci95 = 1.96 * sem
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_sem"] = sem
            out[f"{metric}_ci95"] = ci95

        summary_rows.append(out)

    return summary_rows


def write_summary_csv(summary_rows: list[dict[str, Any]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    base_fields = [
        "exp_name",
        "policy",
        "enable_robust_response",
        "enable_failures",
        "duration",
        "n_runs",
        "recovered_rate",
    ]
    metric_fields: list[str] = []
    for metric in NUMERIC_SUMMARY_FIELDS:
        metric_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_sem", f"{metric}_ci95"])
    fieldnames = base_fields + metric_fields

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def run_experiments(
    *,
    outdir: Path,
    exp: str,
    seeds_override: list[int] | None,
    duration_override: float | None,
    parallel: bool,
    no_plots: bool,
    low_load: bool,
    sim_dt_override: float | None,
    timeseries_stride: int,
    energy_proxy_k: float,
    recovery_pre_window: float,
    recovery_hold_sec: float,
    recovery_ratio: float,
) -> tuple[Path, Path]:
    cases = build_cases(exp=exp, seeds_override=seeds_override, duration_override=duration_override)
    if not cases:
        raise RuntimeError("no experiment cases generated")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "timeseries").mkdir(parents=True, exist_ok=True)

    sim_dt = float(sim_dt_override if sim_dt_override is not None else (10.0 if low_load else 5.0))
    stride = max(1, int(timeseries_stride))
    if low_load:
        stride = max(stride, 2)

    rows: list[dict[str, Any]] = []
    total = len(cases)
    if parallel:
        max_workers = min(4, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    run_single_case,
                    case,
                    str(outdir),
                    sim_dt=sim_dt,
                    timeseries_stride=stride,
                    energy_proxy_k=energy_proxy_k,
                    recovery_pre_window=recovery_pre_window,
                    recovery_hold_sec=recovery_hold_sec,
                    recovery_ratio=recovery_ratio,
                )
                for case in cases
            ]
            for idx, fut in enumerate(futures, start=1):
                row = fut.result()
                rows.append(row)
                print(
                    f"[STEP11] {idx}/{total} {row['exp_name']} policy={row['policy']} "
                    f"seed={row['seed']} mean_end={float(row['mean_info_all_end']):.4f}"
                )
    else:
        for idx, case in enumerate(cases, start=1):
            row = run_single_case(
                case,
                str(outdir),
                sim_dt=sim_dt,
                timeseries_stride=stride,
                energy_proxy_k=energy_proxy_k,
                recovery_pre_window=recovery_pre_window,
                recovery_hold_sec=recovery_hold_sec,
                recovery_ratio=recovery_ratio,
            )
            rows.append(row)
            print(
                f"[STEP11] {idx}/{total} {row['exp_name']} policy={row['policy']} "
                f"seed={row['seed']} mean_end={float(row['mean_info_all_end']):.4f}"
            )

    results_path = outdir / "results.csv"
    write_results_csv(rows, results_path)

    summary_rows = build_summary(rows)
    summary_path = outdir / "summary.csv"
    write_summary_csv(summary_rows, summary_path)

    if not no_plots:
        env = dict(os.environ)
        env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/pycache")
        env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
        plot_cmd = [sys.executable, str(ROOT / "scripts" / "plot_step11.py"), "--outdir", str(outdir)]
        subprocess.run(plot_cmd, check=True, env=env)

    return results_path, summary_path


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    seeds_override = parse_seed_list(args.seeds)
    duration_override = args.duration
    parallel = bool(args.parallel)

    results_path, summary_path = run_experiments(
        outdir=outdir,
        exp=args.exp,
        seeds_override=seeds_override,
        duration_override=duration_override,
        parallel=parallel,
        no_plots=bool(args.no_plots),
        low_load=bool(args.low_load),
        sim_dt_override=args.sim_dt,
        timeseries_stride=args.timeseries_stride,
        energy_proxy_k=float(args.energy_proxy_k),
        recovery_pre_window=float(args.recovery_pre_window),
        recovery_hold_sec=float(args.recovery_hold_sec),
        recovery_ratio=float(args.recovery_ratio),
    )

    print(f"RESULTS={results_path}")
    print(f"SUMMARY={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
