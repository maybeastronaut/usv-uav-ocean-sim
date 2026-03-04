from __future__ import annotations

import math


def plan_rendezvous(uav, usv, env, safety_margin: float = 50.0) -> tuple[float, float]:
    """Plan a simple feasible rendezvous point in sea space.

    Step7 minimum rule: default to current USV position, then project outside obstacles
    and clamp into map bounds.
    """
    _ = uav
    x, y = usv.pos
    x = min(max(x, 0.0), env.map_width)
    y = min(max(y, 0.0), env.map_height)

    if not env.is_in_obstacle(x, y):
        return (x, y)

    for obstacle in env.obstacles:
        dx = x - obstacle.center_x
        dy = y - obstacle.center_y
        dist = math.hypot(dx, dy)
        if dist <= 1e-9:
            dx, dy, dist = 1.0, 0.0, 1.0
        target_r = obstacle.radius + safety_margin
        if dist < target_r:
            scale = target_r / dist
            x = obstacle.center_x + dx * scale
            y = obstacle.center_y + dy * scale

    x = min(max(x, 0.0), env.map_width)
    y = min(max(y, 0.0), env.map_height)
    return (x, y)
