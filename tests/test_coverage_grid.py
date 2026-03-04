from __future__ import annotations

from sim.config import SimConfig
from sim.coverage.coverage_grid import CoverageGrid, check_resolution


def test_world_to_cell_stays_in_bounds() -> None:
    cfg = SimConfig(cell_size=100.0)
    grid = CoverageGrid(cfg)

    samples = [
        (0.0, 0.0),
        (cfg.map_width - 1e-6, cfg.map_height - 1e-6),
        (-10.0, -20.0),
        (cfg.map_width + 100.0, cfg.map_height + 100.0),
    ]
    for x, y in samples:
        ix, iy = grid.world_to_cell(x, y)
        assert 0 <= ix < grid.nx
        assert 0 <= iy < grid.ny


def test_observe_updates_some_cells() -> None:
    cfg = SimConfig(cell_size=100.0)
    grid = CoverageGrid(cfg)
    updated = grid.observe(
        position=(cfg.map_width * 0.5, cfg.map_height * 0.5),
        radius=cfg.uav_sensor_radius,
        t=10.0,
    )
    assert updated > 0


def test_decay_monotonic_non_increasing() -> None:
    cfg = SimConfig(cell_size=100.0, decay_mode="exponential", decay_tau=1200.0)
    grid = CoverageGrid(cfg)
    t0 = 100.0
    grid.observe(position=(5000.0, 5000.0), radius=500.0, t=t0)

    m1 = grid.mean_info(t0 + 300.0)
    m2 = grid.mean_info(t0 + 1200.0)
    assert m2 <= m1


def test_visited_mask_updates() -> None:
    cfg = SimConfig(cell_size=100.0)
    grid = CoverageGrid(cfg)
    assert grid.visited_count == 0
    grid.observe(position=(5000.0, 5000.0), radius=300.0, t=0.0)
    assert grid.visited_count > 0


def test_min_visited_not_zero_after_observe() -> None:
    cfg = SimConfig(cell_size=100.0, decay_mode="exponential", decay_tau=1800.0)
    grid = CoverageGrid(cfg)
    t_obs = 50.0
    grid.observe(position=(5000.0, 5000.0), radius=400.0, t=t_obs)

    min_all = grid.min_info(t_obs, mode="all")
    min_visited = grid.min_info(t_obs, mode="visited")
    assert min_all <= 1e-6
    assert min_visited > 0.8


def test_resolution_check() -> None:
    d1 = check_resolution(cell_size=100.0, sensor_radius=100.0)  # ratio=1
    d2 = check_resolution(cell_size=100.0, sensor_radius=300.0)  # ratio=3
    d3 = check_resolution(cell_size=100.0, sensor_radius=1200.0)  # ratio=12

    assert d1["level"] == "WARNING"
    assert d2["level"] == "OK"
    assert d3["level"] == "NOTICE"
