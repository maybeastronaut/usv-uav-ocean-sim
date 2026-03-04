from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

# =========================
# System-Wide Fixed Parameters
# =========================

# Reproducibility
RANDOM_SEED: int = 0

# Platform counts
NUM_UAV: int = 2
NUM_USV: int = 3

# Physical body size (meters)
UAV_SIZE: float = 1.0
USV_SIZE: float = 4.0

# Visualization draw radius on a 10km x 10km map (meters)
UAV_DRAW_RADIUS: float = 50.0
USV_DRAW_RADIUS: float = 100.0

# Platform speed (meters / second)
UAV_SPEED: float = 20.0
USV_SPEED: float = 5.0

# Sensor radius (meters)
UAV_SENSOR_RADIUS: float = 400.0
USV_SENSOR_RADIUS: float = 150.0

# UAV energy model
UAV_MAX_ENERGY: float = 3600.0  # seconds of flight time
UAV_ENERGY_CONSUMPTION: float = 1.0  # energy units per second
UAV_LOW_ENERGY_THRESHOLD: float = 0.2  # fraction in [0, 1]

# USV-assisted recharge parameters
UAV_RECHARGE_TIME: float = 300.0  # seconds

# Map and zone parameters (meters)
MAP_WIDTH: float = 10000.0
MAP_HEIGHT: float = 10000.0
NEARSHORE_LIMIT: float = 2000.0
RISK_LIMIT: float = 6000.0
OFFSHORE_LIMIT: float = 10000.0

# Ocean current field base vectors by band (meters / second)
CURRENT_NEAR: tuple[float, float] = (0.15, 0.05)
CURRENT_RISK: tuple[float, float] = (0.45, 0.12)
CURRENT_OFFSHORE: tuple[float, float] = (0.80, 0.20)
CURRENT_NOISE_AMPLITUDE: float = 0.05  # meters / second
CURRENT_TIME_VARIATION_AMPLITUDE: float = 0.08  # meters / second
CURRENT_TIME_VARIATION_PERIOD: float = 600.0  # seconds

# Wind field parameters
WIND_BASE: tuple[float, float] = (6.0, 1.0)  # meters / second
WIND_Y_GRADIENT: float = 0.35  # relative gain from y=0 to y=MAP_HEIGHT
WIND_TIME_VARIATION_AMPLITUDE: float = 0.80  # meters / second
WIND_TIME_VARIATION_PERIOD: float = 900.0  # seconds

# Communication range for any pair of nodes (Base/UAV/USV) in peer-to-peer checks (meters)
COMM_RADIUS: float = 3000.0

# Quiver rendering
QUIVER_STEP: float = 500.0  # meters
QUIVER_SCALE: float = 140.0  # display gain (arrow length multiplier)

# Coverage grid parameters
CELL_SIZE: float = 100.0  # meters
DECAY_MODE: str = "exponential"  # "exponential" | "linear"
DECAY_TAU: float = 1800.0  # seconds


@dataclass
class SimConfig:
    seed: int = RANDOM_SEED
    dt: float = 1.0
    steps: int = 200
    num_uav: int = NUM_UAV
    num_usv: int = NUM_USV
    map_width: float = MAP_WIDTH
    map_height: float = MAP_HEIGHT
    base_x: float | None = None
    base_y: float | None = None
    nearshore_y_max: float = NEARSHORE_LIMIT
    risk_zone_y_min: float = NEARSHORE_LIMIT
    risk_zone_y_max: float = RISK_LIMIT
    offshore_y_min: float = RISK_LIMIT
    offshore_y_max: float = OFFSHORE_LIMIT
    risk_obstacle_count: int = 20
    risk_obstacle_radius_min: float = 80.0
    risk_obstacle_radius_max: float = 220.0
    uav_sensor_radius: float = UAV_SENSOR_RADIUS
    usv_sensor_radius: float = USV_SENSOR_RADIUS
    current_near: tuple[float, float] = CURRENT_NEAR
    current_risk: tuple[float, float] = CURRENT_RISK
    current_offshore: tuple[float, float] = CURRENT_OFFSHORE
    current_noise_amplitude: float = CURRENT_NOISE_AMPLITUDE
    current_time_variation_amplitude: float = CURRENT_TIME_VARIATION_AMPLITUDE
    current_time_variation_period: float = CURRENT_TIME_VARIATION_PERIOD
    wind_base: tuple[float, float] = WIND_BASE
    wind_y_gradient: float = WIND_Y_GRADIENT
    wind_time_variation_amplitude: float = WIND_TIME_VARIATION_AMPLITUDE
    wind_time_variation_period: float = WIND_TIME_VARIATION_PERIOD
    comm_radius: float = COMM_RADIUS
    quiver_step: float = QUIVER_STEP
    quiver_scale: float = QUIVER_SCALE
    cell_size: float = CELL_SIZE
    decay_mode: str = DECAY_MODE
    decay_tau: float = DECAY_TAU
    runs_dir: str = "runs"
    run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_run_dir(config: SimConfig) -> Path:
    base = Path(config.runs_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_id = config.run_name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    run_dir = base / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir
