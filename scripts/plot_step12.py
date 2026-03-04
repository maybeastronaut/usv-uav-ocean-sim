from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step12 plotting")
    parser.add_argument("--outdir", type=str, default="runs/step12")
    return parser.parse_args()


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return (mean, 0.0)
    std = float(np.std(arr, ddof=1))
    sem = std / math.sqrt(arr.size)
    return (mean, 1.96 * sem)


def _plot_exp1_bar(rows: list[dict[str, str]], metric: str, ylabel: str, title: str, out_path: Path) -> None:
    exp1 = [r for r in rows if r.get("exp_name") == "exp1"]
    if not exp1:
        return

    order = ["random", "nearest", "priority", "multimetric"]
    policies = [p for p in order if any(r.get("policy") == p for r in exp1)]
    if not policies:
        return

    means: list[float] = []
    cis: list[float] = []
    for p in policies:
        vals = [_to_float(r.get(metric)) for r in exp1 if r.get("policy") == p]
        vals = [float(v) for v in vals if v is not None]
        m, ci = _mean_ci(vals)
        means.append(m)
        cis.append(ci)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(policies))
    ax.bar(x, means, yerr=cis, capsize=4, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_npz_records(timeseries_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not timeseries_dir.exists():
        return records
    for path in sorted(timeseries_dir.glob("*.npz")):
        data = np.load(path)
        rec = {
            "path": str(path),
            "time": np.array(data["time"], dtype=float),
            "mean_info_all": np.array(data["mean_info_all"], dtype=float),
            "pending_count": np.array(data["pending_count"], dtype=float),
            "done_count": np.array(data["done_count"], dtype=float),
            "recharge_count": np.array(data["recharge_count"], dtype=float),
            "exp_name": str(data["exp_name"]),
            "policy": str(data["policy"]),
            "seed": int(data["seed"]),
            "enable_robust_response": int(data["enable_robust_response"]),
            "failure_t": float(data["failure_t"]),
            "baseline_pre": float(data["baseline_pre"]) if "baseline_pre" in data else np.nan,
            "threshold_recover": float(data["threshold_recover"]) if "threshold_recover" in data else np.nan,
        }
        records.append(rec)
    return records


def _stack_series(records: list[dict[str, object]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not records:
        return None
    times = np.array(records[0]["time"], dtype=float)
    mats: list[np.ndarray] = []
    for rec in records:
        t = np.array(rec["time"], dtype=float)
        v = np.array(rec[key], dtype=float)
        if t.shape != times.shape or not np.allclose(t, times):
            return None
        mats.append(v)
    mat = np.stack(mats, axis=0)
    return (times, np.mean(mat, axis=0), np.std(mat, axis=0))


def _plot_exp2_recovery_v2(rows: list[dict[str, str]], records: list[dict[str, object]], out_path: Path) -> None:
    exp2_rows = [r for r in rows if r.get("exp_name") == "exp2" and r.get("policy") == "multimetric"]
    exp2_records = [r for r in records if r["exp_name"] == "exp2" and r["policy"] == "multimetric"]
    if not exp2_rows or not exp2_records:
        return

    baseline_recs = [r for r in exp2_records if int(r["enable_robust_response"]) == 0]
    robust_recs = [r for r in exp2_records if int(r["enable_robust_response"]) == 1]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for recs, label, color in (
        (baseline_recs, "baseline", "tab:red"),
        (robust_recs, "robust", "tab:blue"),
    ):
        stacked = _stack_series(recs, "mean_info_all")
        if stacked is None:
            continue
        t, mean, std = stacked
        ax.plot(t, mean, label=f"{label} mean", color=color, linewidth=2.0)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18)

    fail_t_vals = [_to_float(r.get("failure_t")) for r in exp2_rows]
    fail_t_vals = [float(v) for v in fail_t_vals if v is not None]
    fail_t = float(np.mean(np.array(fail_t_vals))) if fail_t_vals else 600.0
    ax.axvline(fail_t, color="magenta", linestyle="--", linewidth=1.2, alpha=0.8, label="t_fail")

    baseline_pre_vals = [_to_float(r.get("baseline_pre")) for r in exp2_rows]
    baseline_pre_vals = [float(v) for v in baseline_pre_vals if v is not None]
    threshold_vals = [_to_float(r.get("threshold_recover")) for r in exp2_rows]
    threshold_vals = [float(v) for v in threshold_vals if v is not None]

    if baseline_pre_vals:
        base_line = float(np.mean(np.array(baseline_pre_vals)))
        ax.axhline(base_line, color="black", linestyle=":", linewidth=1.2, alpha=0.85, label="baseline_pre")
    if threshold_vals:
        thr_line = float(np.mean(np.array(threshold_vals)))
        ax.axhline(thr_line, color="gray", linestyle="-.", linewidth=1.2, alpha=0.85, label="threshold_recover")

    ax.set_title("Exp2 Recovery Curve V2 (mean ± std)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean_info_all")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_plots(outdir: Path) -> list[Path]:
    results_path = outdir / "results.csv"
    rows = _read_results(results_path)
    figs_dir = outdir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    records = _load_npz_records(outdir / "timeseries")

    created: list[Path] = []

    targets = [
        ("total_distance_all", "total distance (m)", "Exp1 Distance (mean ± 95% CI)", figs_dir / "exp1_distance_all.png"),
        ("total_energy_all", "total energy (proxy)", "Exp1 Energy Proxy (mean ± 95% CI)", figs_dir / "exp1_energy_all.png"),
        ("info_per_distance", "mean_end / distance", "Exp1 Info per Distance (mean ± 95% CI)", figs_dir / "exp1_info_per_distance.png"),
        ("done_per_distance", "done / distance", "Exp1 Done per Distance (mean ± 95% CI)", figs_dir / "exp1_done_per_distance.png"),
    ]
    for metric, ylabel, title, out_path in targets:
        _plot_exp1_bar(rows, metric, ylabel, title, out_path)
        if out_path.exists():
            created.append(out_path)

    exp2_path = figs_dir / "exp2_recovery_curve_v2.png"
    _plot_exp2_recovery_v2(rows, records, exp2_path)
    if exp2_path.exists():
        created.append(exp2_path)

    return created


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    created = generate_plots(outdir)
    for path in created:
        print(f"FIG={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
