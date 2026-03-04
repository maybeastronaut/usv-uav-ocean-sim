# usv-uav-ocean-sim

面向海洋环境监测的异构多智能体区域覆盖方法研究（当前阶段：工程骨架 + 空仿真循环）。

## 1. Requirements

```bash
python3 --version
```

Python 3.10+ is required.

## 2. Run

```bash
python -m sim.run --seed 0 --steps 200
# if your environment does not have "python" alias:
python3 -m sim.run --seed 0 --steps 200
```

Example with custom output root:

```bash
python -m sim.run --seed 0 --steps 50 --runs-dir ./runs
```

Run artifacts are written to:

```text
runs/<timestamp-uuid>/
├── config.json
├── run.log
└── metrics.csv
```

## 3. Preview environment

```bash
python3 scripts/preview_environment.py --seed 0
```

This generates:

```text
runs/<timestamp-uuid>/environment_preview.png
```

## 4. One-click smoke run

```bash
bash scripts/smoke_run.sh
```

## 5. Test

```bash
python3 -m pytest
```
