from __future__ import annotations

from sim.config import SimConfig
from sim.coverage.coverage_grid import CoverageGrid
from sim.tasks.region_map import RegionMap
from sim.tasks.task_generator import TaskGenerator


def test_region_mapping() -> None:
    cfg = SimConfig(region_cell_size=1000.0)
    grid = CoverageGrid(cfg)
    region_map = RegionMap(grid, cfg)

    rx, ry = region_map.nrx - 1, region_map.nry - 1
    cx, cy = region_map.region_center(rx, ry)
    assert 0.0 <= cx <= cfg.map_width
    assert 0.0 <= cy <= cfg.map_height

    cells = region_map.grid_cells_in_region(rx, ry)
    assert len(cells) > 0
    for ix, iy in cells:
        assert 0 <= ix < grid.nx
        assert 0 <= iy < grid.ny


def test_task_generation_threshold() -> None:
    cfg = SimConfig(region_cell_size=1000.0, task_info_threshold=0.3)
    grid = CoverageGrid(cfg)
    region_map = RegionMap(grid, cfg)
    task_gen = TaskGenerator(region_map, cfg)

    low_tasks = task_gen.generate_tasks(t=0.0)
    assert len(low_tasks) > 0

    for ry in range(region_map.nry):
        for rx in range(region_map.nrx):
            center = region_map.region_center(rx, ry)
            grid.observe(center, radius=cfg.region_cell_size * 0.8, t=1.0)
    high_tasks = task_gen.generate_tasks(t=cfg.task_cooldown + 2.0)
    assert len(high_tasks) == 0


def test_task_cooldown_and_dedup() -> None:
    cfg = SimConfig(region_cell_size=1000.0, task_cooldown=600.0, max_pending_tasks=200)
    grid = CoverageGrid(cfg)
    region_map = RegionMap(grid, cfg)
    task_gen = TaskGenerator(region_map, cfg)

    first = task_gen.generate_tasks(t=0.0)
    assert len(first) > 0
    second = task_gen.generate_tasks(t=10.0)
    assert len(second) <= cfg.max_new_tasks_per_tick
    first_regions = {task.region_id for task in first}
    second_regions = {task.region_id for task in second}
    assert first_regions.isdisjoint(second_regions)

    pending = task_gen.get_pending_tasks()
    region_ids = [t.region_id for t in pending]
    assert len(region_ids) == len(set(region_ids))


def test_priority_monotonic() -> None:
    cfg = SimConfig(region_cell_size=1000.0)
    grid = CoverageGrid(cfg)
    region_map = RegionMap(grid, cfg)
    task_gen = TaskGenerator(region_map, cfg)

    task_gen.generate_tasks(t=0.0)
    pending = task_gen.get_pending_tasks()
    assert len(pending) > 0
    target = pending[0]
    p0 = target.priority

    grid.observe(position=target.target_pos, radius=cfg.region_cell_size * 0.9, t=1.0)
    task_gen.generate_tasks(t=cfg.task_cooldown + 1.0)
    refreshed = [t for t in task_gen.get_pending_tasks() if t.task_id == target.task_id][0]
    assert refreshed.priority <= p0
