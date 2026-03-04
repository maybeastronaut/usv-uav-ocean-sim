from __future__ import annotations

import math
from dataclasses import dataclass


def segment_circle_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = center
    dx = x2 - x1
    dy = y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-12:
        return (x1 - cx) ** 2 + (y1 - cy) ** 2 <= radius * radius
    t = ((cx - x1) * dx + (cy - y1) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    px = x1 + t * dx
    py = y1 + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2 <= radius * radius


def segment_hits_any_obstacle(
    p1: tuple[float, float],
    p2: tuple[float, float],
    obstacles,
    margin: float = 0.0,
) -> bool:
    for ob in obstacles:
        r = ob.radius + margin
        if segment_circle_intersect(p1, p2, (ob.center_x, ob.center_y), r):
            return True
    return False


@dataclass
class PathPlanner:
    safe_margin: float = 0.0

    def plan(self, agent, goal_pos: tuple[float, float], env, t: float) -> list[tuple[float, float]]:
        if agent.agent_type == "UAV":
            return UAVPlanner(safe_margin=self.safe_margin).plan(agent, goal_pos, env, t)
        return USVPlanner(safe_margin=self.safe_margin).plan(agent, goal_pos, env, t)

    @staticmethod
    def next_waypoint(agent, waypoints: list[tuple[float, float]], t: float) -> tuple[float, float] | None:
        _ = t
        if not waypoints:
            return None
        idx = min(max(agent.current_wp_idx, 0), len(waypoints) - 1)
        return waypoints[idx]


@dataclass
class UAVPlanner:
    safe_margin: float = 0.0

    def plan(self, agent, goal_pos: tuple[float, float], env, t: float) -> list[tuple[float, float]]:
        _ = (agent, env, t, self.safe_margin)
        return [goal_pos]


@dataclass
class USVPlanner:
    safe_margin: float = 150.0

    def plan(self, agent, goal_pos: tuple[float, float], env, t: float) -> list[tuple[float, float]]:
        _ = t
        start = agent.pos
        if not segment_hits_any_obstacle(start, goal_pos, env.obstacles, margin=self.safe_margin):
            return [goal_pos]

        hit_obstacles = []
        for ob in env.obstacles:
            inflated = ob.radius + self.safe_margin
            if segment_circle_intersect(start, goal_pos, (ob.center_x, ob.center_y), inflated):
                hit_obstacles.append(ob)

        best_wp: tuple[float, float] | None = None
        best_cost = float("inf")
        for ob in hit_obstacles:
            inflated = ob.radius + self.safe_margin
            candidates = self._bypass_candidates(start, goal_pos, (ob.center_x, ob.center_y), inflated)
            for wp in candidates:
                if not env.is_inside_map(wp[0], wp[1]):
                    continue
                if env.is_in_obstacle(wp[0], wp[1]):
                    continue
                if segment_hits_any_obstacle(start, wp, env.obstacles, margin=self.safe_margin):
                    continue
                if segment_hits_any_obstacle(wp, goal_pos, env.obstacles, margin=self.safe_margin):
                    continue
                cost = _dist(start, wp) + _dist(wp, goal_pos)
                if cost < best_cost:
                    best_cost = cost
                    best_wp = wp

        if best_wp is None:
            fallback = self._fallback_corridor_path(start, goal_pos, env, hit_obstacles)
            if fallback is not None:
                return fallback
            return [goal_pos]
        return [best_wp, goal_pos]

    def _bypass_candidates(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        center: tuple[float, float],
        r: float,
    ) -> list[tuple[float, float]]:
        cx, cy = center
        _ = (start, goal)
        # Put candidate waypoints slightly outside inflated obstacle boundary,
        # then pick the shortest valid two-segment route.
        extra = max(30.0, 0.2 * self.safe_margin)
        d = r + extra
        candidates: list[tuple[float, float]] = []
        n = 16
        for k in range(n):
            theta = 2.0 * math.pi * k / n
            candidates.append((cx + d * math.cos(theta), cy + d * math.sin(theta)))
        return candidates

    def _fallback_corridor_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        env,
        hit_obstacles,
    ) -> list[tuple[float, float]] | None:
        if not hit_obstacles:
            return None
        mean_y = sum(ob.center_y for ob in hit_obstacles) / len(hit_obstacles)
        max_r = max(ob.radius for ob in hit_obstacles)
        band = max_r + self.safe_margin + 120.0
        for sign in (1.0, -1.0):
            y = mean_y + sign * band
            y = min(max(y, 0.0), env.map_height)
            wp1 = (start[0], y)
            wp2 = (goal[0], y)
            if not env.is_inside_map(wp1[0], wp1[1]) or not env.is_inside_map(wp2[0], wp2[1]):
                continue
            segments = [(start, wp1), (wp1, wp2), (wp2, goal)]
            ok = True
            for a, b in segments:
                if segment_hits_any_obstacle(a, b, env.obstacles, margin=self.safe_margin):
                    ok = False
                    break
            if ok:
                return [wp1, wp2, goal]
        return None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
