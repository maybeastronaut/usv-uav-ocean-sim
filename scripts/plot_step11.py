from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step11 plotting")
    parser.add_argument("--outdir", type=str, default="runs/step11")
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
        reader = csv.DictReader(f)
        return list(reader)


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


def _plot_exp1_metric(rows: list[dict[str, str]], metric: str, ylabel: str, out_path: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(policies))
    ax.bar(x, means, yerr=cis, capsize=4, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Exp1 {metric} (mean ± 95% CI)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_npz_records(timeseries_dir: Path) -> list[dict[str, np.ndarray | str | int | float]]:
    records: list[dict[str, np.ndarray | str | int | float]] = []
    if not timeseries_dir.exists():
        return records
    for path in sorted(timeseries_dir.glob("*.npz")):
        data = np.load(path)
        rec: dict[str, np.ndarray | str | int | float] = {
            "path": str(path),
            "time": data["time"],
            "mean_info_all": data["mean_info_all"],
            "pending_count": data["pending_count"],
            "done_count": data["done_count"],
            "recharge_count": data["recharge_count"],
            "exp_name": str(data["exp_name"]),
            "policy": str(data["policy"]),
            "seed": int(data["seed"]),
            "enable_robust_response": int(data["enable_robust_response"]),
            "failure_t": float(data["failure_t"]),
        }
        records.append(rec)
    return records


def _stack_series(records: list[dict[str, np.ndarray | str | int | float]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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


def _plot_exp2_recovery(records: list[dict[str, np.ndarray | str | int | float]], out_path: Path) -> None:
    exp2 = [r for r in records if r["exp_name"] == "exp2" and r["policy"] == "multimetric"]
    if not exp2:
        return
    baseline = [r for r in exp2 if int(r["enable_robust_response"]) == 0]
    robust = [r for r in exp2 if int(r["enable_robust_response"]) == 1]
    if not baseline and not robust:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for recs, label, color in (
        (baseline, "baseline", "tab:red"),
        (robust, "robust", "tab:blue"),
    ):
        stacked = _stack_series(recs, "mean_info_all")
        if stacked is None:
            continue
        t, mean, std = stacked
        ax.plot(t, mean, label=label, color=color, linewidth=2.0)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18)

    fail_t = float(exp2[0]["failure_t"])
    if fail_t >= 0:
        ax.axvline(fail_t, color="magenta", linestyle="--", linewidth=1.2, alpha=0.8, label="failure")

    ax.set_title("Exp2 Recovery Curve (mean_info_all)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean_info_all")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_exp3_longrun(records: list[dict[str, np.ndarray | str | int | float]], key: str, ylabel: str, out_path: Path) -> None:
    exp3 = [r for r in records if r["exp_name"] == "exp3" and r["policy"] == "multimetric"]
    stacked = _stack_series(exp3, key)
    if stacked is None:
        return
    t, mean, std = stacked

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(t, mean, color="tab:blue", linewidth=2.0)
    ax.fill_between(t, mean - std, mean + std, color="tab:blue", alpha=0.18)
    ax.set_title(f"Exp3 Long-run {key}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_plots(outdir: Path) -> list[Path]:
    results_path = outdir / "results.csv"
    rows = _read_results(results_path)
    figs_dir = outdir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    fig1 = figs_dir / "exp1_mean_end.png"
    _plot_exp1_metric(rows, "mean_info_all_end", "mean_info_all_end", fig1)
    if fig1.exists():
        created.append(fig1)

    fig2 = figs_dir / "exp1_done_count.png"
    _plot_exp1_metric(rows, "done_count", "done_count", fig2)
    if fig2.exists():
        created.append(fig2)

    fig3 = figs_dir / "exp1_pending_end.png"
    _plot_exp1_metric(rows, "pending_end", "pending_end", fig3)
    if fig3.exists():
        created.append(fig3)

    ts_records = _load_npz_records(outdir / "timeseries")

    fig4 = figs_dir / "exp2_recovery_curve.png"
    _plot_exp2_recovery(ts_records, fig4)
    if fig4.exists():
        created.append(fig4)

    fig5 = figs_dir / "exp3_longrun_mean.png"
    _plot_exp3_longrun(ts_records, "mean_info_all", "mean_info_all", fig5)
    if fig5.exists():
        created.append(fig5)

    fig6 = figs_dir / "exp3_longrun_pending.png"
    _plot_exp3_longrun(ts_records, "pending_count", "pending_count", fig6)
    if fig6.exists():
        created.append(fig6)

    fig7 = figs_dir / "exp3_longrun_recharge.png"
    _plot_exp3_longrun(ts_records, "recharge_count", "recharge_count_cum", fig7)
    if fig7.exists():
        created.append(fig7)

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
