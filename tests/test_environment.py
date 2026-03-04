from __future__ import annotations

from sim.config import SimConfig
from sim.environment.environment import Environment2D


def test_obstacles_stay_in_risk_zone() -> None:
    cfg = SimConfig(
        seed=7,
        map_width=10000.0,
        map_height=10000.0,
        nearshore_y_max=2000.0,
        risk_zone_y_min=2000.0,
        risk_zone_y_max=6000.0,
        offshore_y_min=6000.0,
        offshore_y_max=10000.0,
        risk_obstacle_count=12,
        risk_obstacle_radius_min=60.0,
        risk_obstacle_radius_max=120.0,
    )
    env = Environment2D(cfg)
    risk_y_min = cfg.risk_zone_y_min
    risk_y_max = cfg.risk_zone_y_max

    assert len(env.obstacles) == cfg.risk_obstacle_count
    for ob in env.obstacles:
        assert risk_y_min <= ob.center_y <= risk_y_max
        assert 0.0 <= ob.center_x - ob.radius
        assert ob.center_x + ob.radius <= cfg.map_width
        assert 0.0 <= ob.center_y - ob.radius
        assert ob.center_y + ob.radius <= cfg.map_height


def test_sample_point_for_each_region() -> None:
    cfg = SimConfig(
        seed=3,
        map_width=9000.0,
        map_height=9000.0,
        nearshore_y_max=1800.0,
        risk_zone_y_min=1800.0,
        risk_zone_y_max=5000.0,
        offshore_y_min=5000.0,
        offshore_y_max=9000.0,
        risk_obstacle_count=8,
        risk_obstacle_radius_min=40.0,
        risk_obstacle_radius_max=80.0,
    )
    env = Environment2D(cfg)

    for region in ("nearshore", "risk_zone", "offshore"):
        x, y = env.sample_point(region)
        assert env.is_inside_map(x, y)
        assert env.is_in_region(x, y, region)
        assert not env.is_in_obstacle(x, y)
