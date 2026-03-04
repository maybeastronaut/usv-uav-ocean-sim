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
USV_SPEED: float = 40.0

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

# Region-task aggregation parameters
REGION_CELL_SIZE: float = 1000.0  # meters
REGION_AGG_MODE: str = "mean"  # "mean" | "min" | "p5"
TASK_INFO_THRESHOLD: float = 0.3  # generate task when region_info < threshold
REGION_TASK_COOLDOWN: float = 300.0  # seconds
TASK_COOLDOWN: float = REGION_TASK_COOLDOWN  # backward-compatible alias
MAX_PENDING_TASKS: int = 100
MAX_NEW_TASKS_PER_TICK: int = 5
NEARSHORE_TASK_WEIGHT: float = 1.2
RISK_TASK_WEIGHT: float = 1.0
OFFSHORE_TASK_WEIGHT: float = 0.8

# Step5 simulator execution parameters
SIM_DT: float = 5.0  # seconds
T_END: float = 1200.0  # seconds
TASK_TIMEOUT: float = 1200.0  # seconds
TASK_COMPLETE_THRESHOLD: float = 0.4
WORK_RADIUS_FACTOR: float = 0.5  # work radius = sensor_radius * factor
FRAME_EVERY: float = 10.0  # seconds
WIND_EFFECT_UAV: float = 0.0  # wind contribution scaling for UAV
CURRENT_EFFECT_USV: float = 1.0  # current contribution scaling for USV
USV_TURN_RATE_DEG: float = 20.0  # degrees per second
OBSTACLE_AVOIDANCE_ANGLES: tuple[float, ...] = (15.0, 30.0, 45.0)  # degrees

# Step6 path/policy parameters
SAFE_MARGIN: float = 150.0  # meters, obstacle bypass margin
WP_REACHED_EPS: float = 80.0  # meters, waypoint reached threshold
STRATEGY: str = "priority"  # "random" | "nearest" | "priority"

# Step8 task layer parameters
TASK_POLICY: str = "priority"  # "random" | "nearest" | "priority" | "multimetric"
ABLATE_SOFTPART: bool = False
ABLATE_ENERGY_TERM: bool = False
ABLATE_RISK_TERM: bool = False
W_NEED: float = 0.45
W_PRIO: float = 0.30
W_DIST: float = 0.90
W_RISK: float = 0.00
W_ENERGY: float = 0.90
W_SOFTPART: float = 0.20
SOFTPART_SIGMA: float = 1200.0  # meters
PENDING_CROSS_THRESHOLD: int = 85
PENDING_CROSS_THRESHOLD_FRAC: float = 0.95  # if pending > frac * MAX_PENDING_TASKS, soften partition
MEANINFO_CROSS_THRESHOLD: float = 0.03
SOFTPART_CROSS_SCALE: float = 0.70  # effective soft-partition weight scale under high pressure
RISK_ZONE_PENALTY: float = 1.0

# Step7 recharge parameters
UAV_BATTERY_MAX: float = 620.0  # abstract energy units
UAV_DISCHARGE_RATE: float = 1.75  # energy units / second
UAV_LOW_BATTERY_FRAC: float = 0.35  # trigger recharge below this fraction
UAV_CRITICAL_BATTERY_FRAC: float = 0.6  # emergency level fraction
USV_CHARGE_RATE: float = 14.0  # energy units / second
RENDEZVOUS_EPS: float = 150.0  # meters
RECHARGE_TARGET_FRAC: float = 0.75  # recharge complete at this fraction
RECHARGE_TASK_PRIORITY: float = 2.0  # higher than monitor tasks
ALLOW_USV_PREEMPT_MONITOR_FOR_RECHARGE: bool = True

# Step9 feedback layer parameters
ENABLE_FEEDBACK: bool = True
FB_COOLDOWN_SEC: float = 120.0
FB_COOLDOWN_RELAX: float = 300.0
FB_COOLDOWN_REASSIGN: float = 300.0
FB_COOLDOWN_RECHARGE_BOOST: float = 300.0
FB_PENDING_HIGH_FRAC: float = 0.90
FB_MEANINFO_LOW: float = 0.06
FB_P5INFO_LOW: float = 0.03
FB_ENERGY_PRESSURE_BATT: float = 0.35
FB_RELAX_DURATION_SEC: float = 180.0
FB_RECHARGE_BOOST_DURATION_SEC: float = 120.0
FB_REASSIGN_MODE: str = "soft"  # "soft" | "hard"

# Step10 failure + robust-response parameters
ENABLE_FAILURES: bool = False
FAILURE_MODE: str = "scheduled"  # "scheduled" | "random"
FAILURE_T_SEC: float = 600.0  # seconds
FAILURE_USV_ID: int = 1
FAILURE_KIND: str = "DISABLED"  # "DAMAGED" | "DISABLED"
DAMAGE_SPEED_SCALE: float = 0.6
DAMAGE_TURN_SCALE: float = 0.7
DAMAGE_CHARGE_SCALE: float = 0.5
FAILURE_RANDOM_PROB_PER_SEC: float = 0.0005
FAILURE_RECOVERY: bool = False
FAILURE_EVENT_COOLDOWN: float = 999999.0
ENABLE_ROBUST_RESPONSE: bool = True
FORCED_RELAX_ON_FAILURE: bool = True
FORCED_RELAX_DURATION_SEC: float = 300.0


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
    uav_speed: float = UAV_SPEED
    usv_speed: float = USV_SPEED
    uav_sensor_radius: float = UAV_SENSOR_RADIUS
    usv_sensor_radius: float = USV_SENSOR_RADIUS
    uav_max_energy: float = UAV_MAX_ENERGY
    uav_energy_consumption: float = UAV_ENERGY_CONSUMPTION
    uav_low_energy_threshold: float = UAV_LOW_ENERGY_THRESHOLD
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
    region_cell_size: float = REGION_CELL_SIZE
    region_agg_mode: str = REGION_AGG_MODE
    task_info_threshold: float = TASK_INFO_THRESHOLD
    region_task_cooldown: float = REGION_TASK_COOLDOWN
    task_cooldown: float = TASK_COOLDOWN
    max_pending_tasks: int = MAX_PENDING_TASKS
    max_new_tasks_per_tick: int = MAX_NEW_TASKS_PER_TICK
    nearshore_task_weight: float = NEARSHORE_TASK_WEIGHT
    risk_task_weight: float = RISK_TASK_WEIGHT
    offshore_task_weight: float = OFFSHORE_TASK_WEIGHT
    sim_dt: float = SIM_DT
    t_end: float = T_END
    task_timeout: float = TASK_TIMEOUT
    task_complete_threshold: float = TASK_COMPLETE_THRESHOLD
    work_radius_factor: float = WORK_RADIUS_FACTOR
    frame_every: float = FRAME_EVERY
    wind_effect_uav: float = WIND_EFFECT_UAV
    current_effect_usv: float = CURRENT_EFFECT_USV
    usv_turn_rate_deg: float = USV_TURN_RATE_DEG
    obstacle_avoidance_angles: tuple[float, ...] = OBSTACLE_AVOIDANCE_ANGLES
    safe_margin: float = SAFE_MARGIN
    wp_reached_eps: float = WP_REACHED_EPS
    strategy: str = STRATEGY
    task_policy: str = TASK_POLICY
    ablate_softpart: bool = ABLATE_SOFTPART
    ablate_energy_term: bool = ABLATE_ENERGY_TERM
    ablate_risk_term: bool = ABLATE_RISK_TERM
    w_need: float = W_NEED
    w_prio: float = W_PRIO
    w_dist: float = W_DIST
    w_risk: float = W_RISK
    w_energy: float = W_ENERGY
    w_softpart: float = W_SOFTPART
    softpart_sigma: float = SOFTPART_SIGMA
    pending_cross_threshold: int = PENDING_CROSS_THRESHOLD
    pending_cross_threshold_frac: float = PENDING_CROSS_THRESHOLD_FRAC
    meaninfo_cross_threshold: float = MEANINFO_CROSS_THRESHOLD
    softpart_cross_scale: float = SOFTPART_CROSS_SCALE
    risk_zone_penalty: float = RISK_ZONE_PENALTY
    uav_battery_max: float = UAV_BATTERY_MAX
    uav_discharge_rate: float = UAV_DISCHARGE_RATE
    uav_low_battery_frac: float = UAV_LOW_BATTERY_FRAC
    uav_critical_battery_frac: float = UAV_CRITICAL_BATTERY_FRAC
    usv_charge_rate: float = USV_CHARGE_RATE
    rendezvous_eps: float = RENDEZVOUS_EPS
    recharge_target_frac: float = RECHARGE_TARGET_FRAC
    recharge_task_priority: float = RECHARGE_TASK_PRIORITY
    allow_usv_preempt_monitor_for_recharge: bool = ALLOW_USV_PREEMPT_MONITOR_FOR_RECHARGE
    enable_feedback: bool = ENABLE_FEEDBACK
    fb_cooldown_sec: float = FB_COOLDOWN_SEC
    fb_cooldown_relax: float = FB_COOLDOWN_RELAX
    fb_cooldown_reassign: float = FB_COOLDOWN_REASSIGN
    fb_cooldown_recharge_boost: float = FB_COOLDOWN_RECHARGE_BOOST
    fb_pending_high_frac: float = FB_PENDING_HIGH_FRAC
    fb_meaninfo_low: float = FB_MEANINFO_LOW
    fb_p5info_low: float = FB_P5INFO_LOW
    fb_energy_pressure_batt: float = FB_ENERGY_PRESSURE_BATT
    fb_relax_duration_sec: float = FB_RELAX_DURATION_SEC
    fb_recharge_boost_duration_sec: float = FB_RECHARGE_BOOST_DURATION_SEC
    fb_reassign_mode: str = FB_REASSIGN_MODE
    enable_failures: bool = ENABLE_FAILURES
    failure_mode: str = FAILURE_MODE
    failure_t_sec: float = FAILURE_T_SEC
    failure_usv_id: int = FAILURE_USV_ID
    failure_kind: str = FAILURE_KIND
    damage_speed_scale: float = DAMAGE_SPEED_SCALE
    damage_turn_scale: float = DAMAGE_TURN_SCALE
    damage_charge_scale: float = DAMAGE_CHARGE_SCALE
    failure_random_prob_per_sec: float = FAILURE_RANDOM_PROB_PER_SEC
    failure_recovery: bool = FAILURE_RECOVERY
    failure_event_cooldown: float = FAILURE_EVENT_COOLDOWN
    enable_robust_response: bool = ENABLE_ROBUST_RESPONSE
    forced_relax_on_failure: bool = FORCED_RELAX_ON_FAILURE
    forced_relax_duration_sec: float = FORCED_RELAX_DURATION_SEC
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
