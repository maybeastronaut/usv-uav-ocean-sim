from __future__ import annotations

import math

from sim.agents.base_agent import BaseAgent


class USVAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        pos: tuple[float, float],
        max_speed: float,
        sensor_radius: float,
        comm_radius: float,
        turn_rate_deg: float,
        heading: float = 0.0,
        charge_rate: float = 0.0,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type="USV",
            pos=pos,
            max_speed=max_speed,
            sensor_radius=sensor_radius,
            comm_radius=comm_radius,
        )
        self.turn_rate_deg = turn_rate_deg
        self.heading = heading
        self.charge_rate = max(0.0, charge_rate)
        self.can_charge = True
        self.charging_slots = 1
        self.charging_uav_id: str | None = None

    def step_toward(
        self,
        target_pos: tuple[float, float],
        env,
        t: float,
        dt: float,
        current_effect: float,
        avoidance_angles_deg: tuple[float, ...],
    ) -> None:
        old_pos = self.pos
        desired = math.atan2(target_pos[1] - self.pos[1], target_pos[0] - self.pos[0])
        max_turn = math.radians(self.turn_rate_deg) * dt
        delta = _wrap_pi(desired - self.heading)
        turn = max(-max_turn, min(max_turn, delta))
        base_heading = _wrap_pi(self.heading + turn)

        new_pos, new_vel, chosen_heading = self._find_safe_step(
            env=env,
            t=t,
            dt=dt,
            heading=base_heading,
            current_effect=current_effect,
            avoidance_angles_deg=avoidance_angles_deg,
        )

        self.heading = chosen_heading
        self.pos = new_pos
        self.clamp_to_map(env.map_width, env.map_height)
        # Safety clamp can push into edge; if edge is obstacle, stay put.
        if env.is_in_obstacle(self.pos[0], self.pos[1]):
            self.pos = old_pos
            self.vel = (0.0, 0.0)
            self.stats["avoid_failures"] = self.stats.get("avoid_failures", 0.0) + 1.0
        else:
            self.vel = new_vel
            self._update_distance_stat(old_pos, self.pos)
        self.task_status = "moving"

    def _find_safe_step(
        self,
        env,
        t: float,
        dt: float,
        heading: float,
        current_effect: float,
        avoidance_angles_deg: tuple[float, ...],
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        candidates = [heading]
        for deg in avoidance_angles_deg:
            off = math.radians(deg)
            candidates.append(_wrap_pi(heading + off))
            candidates.append(_wrap_pi(heading - off))

        for h in candidates:
            pos, vel = self._predict_step(env=env, t=t, dt=dt, heading=h, current_effect=current_effect)
            if not env.is_inside_map(pos[0], pos[1]):
                continue
            if env.is_in_obstacle(pos[0], pos[1]):
                continue
            return (pos, vel, h)

        # If every candidate collides, wait in place.
        self.stats["avoid_failures"] = self.stats.get("avoid_failures", 0.0) + 1.0
        return (self.pos, (0.0, 0.0), heading)

    def _predict_step(
        self,
        env,
        t: float,
        dt: float,
        heading: float,
        current_effect: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        body_vx = self.max_speed * math.cos(heading)
        body_vy = self.max_speed * math.sin(heading)
        cx, cy = env.current_at(self.pos[0], self.pos[1], t)
        vx = body_vx + current_effect * cx
        vy = body_vy + current_effect * cy
        pos = (self.pos[0] + vx * dt, self.pos[1] + vy * dt)
        return (pos, (vx, vy))


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
