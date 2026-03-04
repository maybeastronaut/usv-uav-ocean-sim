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
    "recharge_count",
    "uav_dead_count",
    "fb_trigger_count",
    "recovery_time_sec",
    "mean_min_after_failure",
]

NUMERIC_SUMMARY_FIELDS = [
    "mean_info_all_end",
    "mean_info_all_min",
    "mean_info_all_p5",
    "done_count",
    "pending_end",
    "total_distance_usv",
    "total_distance_uav",
    "recharge_count",
    "uav_dead_count",
    "fb_trigger_count",
    "recovery_time_sec",
    "mean_min_after_failure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step11 batch experiment runner")
    parser.add_argument("--outdir", type=str, default="runs/step11")
    parser.add_argument("--exp", type=str, default="all", choices=["exp1", "exp2", "exp3", "all"])
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--parallel", type=int, default=0, choices=[0, 1])
    parser.add_argument("--no-plots", action="store_true")
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

        # optional weak baseline (single seed)
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


def _series_recovery_time(history: list[dict[str, float]], fail_t: float) -> float:
    pre = [row["mean_info_all"] for row in history if row["time"] < fail_t]
    if not pre:
        return math.inf
    pre_mean = float(sum(pre[-5:]) / min(5, len(pre)))
    target = 0.95 * pre_mean
    for row in history:
        if row["time"] < fail_t:
            continue
        if row["mean_info_all"] >= target:
            return float(row["time"] - fail_t)
    return math.inf


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


def run_single_case(case: dict[str, Any], outdir: str) -> dict[str, Any]:
    cfg = SimConfig(
        seed=int(case["seed"]),
        t_end=float(case["duration"]),
        sim_dt=5.0,
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

    recovery_val: float | str = ""
    mean_after_fail_val: float | str = ""
    if bool(case["enable_failures"]):
        fail_t = float(case["failure_t"] or 600.0)
        recovery = _series_recovery_time(history, fail_t)
        mean_after_fail = _series_min_after_failure(history, fail_t, window=200.0)
        recovery_val = "" if math.isinf(recovery) else float(recovery)
        mean_after_fail_val = float(mean_after_fail)

    total_distance_uav = sum(agent.stats.get("distance", 0.0) for agent in sim._uavs())
    total_distance_usv = sum(agent.stats.get("distance", 0.0) for agent in sim._usvs())

    outdir_path = Path(outdir)
    ts_dir = outdir_path / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts_path = ts_dir / _timeseries_filename(case)
    np.savez_compressed(
        ts_path,
        time=np.array([row["time"] for row in history], dtype=float),
        mean_info_all=mean_series,
        pending_count=np.array([row["pending_count"] for row in history], dtype=float),
        done_count=np.array([row["done_count"] for row in history], dtype=float),
        recharge_count=np.array([row["recharge_count_cum"] for row in history], dtype=float),
        fb_trigger_count=np.array([row.get("fb_trigger_count_cum", 0.0) for row in history], dtype=float),
        num_usv_disabled=np.array([row.get("num_usv_disabled", 0.0) for row in history], dtype=float),
        exp_name=np.array(str(case["exp_name"])),
        policy=np.array(str(case["policy"])),
        seed=np.array(int(case["seed"])),
        enable_robust_response=np.array(int(bool(case["enable_robust_response"]))),
        failure_t=np.array(float(case["failure_t"] or -1.0)),
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
        "total_distance_usv": float(total_distance_usv),
        "total_distance_uav": float(total_distance_uav),
        "recharge_count": int(final.get("recharge_count_cum", 0.0)),
        "uav_dead_count": int(final.get("uav_dead_count", 0.0)),
        "fb_trigger_count": int(final.get("fb_trigger_count_cum", 0.0)),
        "recovery_time_sec": recovery_val,
        "mean_min_after_failure": mean_after_fail_val,
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

    base_fields = ["exp_name", "policy", "enable_robust_response", "enable_failures", "duration", "n_runs"]
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
) -> tuple[Path, Path]:
    cases = build_cases(exp=exp, seeds_override=seeds_override, duration_override=duration_override)
    if not cases:
        raise RuntimeError("no experiment cases generated")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "timeseries").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(cases)
    if parallel:
        max_workers = min(4, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_single_case, case, str(outdir)) for case in cases]
            for idx, fut in enumerate(futures, start=1):
                row = fut.result()
                rows.append(row)
                print(
                    f"[STEP11] {idx}/{total} {row['exp_name']} policy={row['policy']} "
                    f"seed={row['seed']} mean_end={float(row['mean_info_all_end']):.4f}"
                )
    else:
        for idx, case in enumerate(cases, start=1):
            row = run_single_case(case, str(outdir))
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
        plot_cmd = [sys.executable, str(ROOT / "scripts" / "plot_step11.py"), "--outdir", str(outdir)]
        subprocess.run(plot_cmd, check=True)

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
    )

    print(f"RESULTS={results_path}")
    print(f"SUMMARY={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
