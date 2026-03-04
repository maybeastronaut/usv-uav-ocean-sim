from __future__ import annotations

import math

from sim.agents.base_agent import BaseAgent


class UAVAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        pos: tuple[float, float],
        max_speed: float,
        sensor_radius: float,
        comm_radius: float,
        energy: float,
        low_energy_threshold: float,
        battery_max: float,
        discharge_rate: float,
        critical_battery_threshold: float,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type="UAV",
            pos=pos,
            max_speed=max_speed,
            sensor_radius=sensor_radius,
            comm_radius=comm_radius,
            energy=energy,
            low_energy_threshold=low_energy_threshold,
        )
        self.battery_max = max(1e-6, battery_max)
        self.battery = min(max(energy, 0.0), self.battery_max)
        self.discharge_rate = max(0.0, discharge_rate)
        self.critical_battery_threshold = max(0.0, min(1.0, critical_battery_threshold))
        self.emergency_events = 0

    @property
    def battery_frac(self) -> float:
        return max(0.0, min(1.0, self.battery / self.battery_max))

    def is_low_battery(self) -> bool:
        return self.battery_frac <= self.low_energy_threshold

    def is_critical(self) -> bool:
        return self.battery_frac <= self.critical_battery_threshold

    def discharge(self, dt: float, moving: bool = True) -> float:
        if not moving or dt <= 0.0:
            return 0.0
        consumed = self.discharge_rate * dt
        before = self.battery
        self.battery = max(0.0, self.battery - consumed)
        return before - self.battery

    def recharge(self, dt: float, charge_rate: float) -> float:
        if dt <= 0.0 or charge_rate <= 0.0:
            return 0.0
        before = self.battery
        self.battery = min(self.battery_max, self.battery + charge_rate * dt)
        return self.battery - before

    def step_toward(
        self,
        target_pos: tuple[float, float],
        env,
        t: float,
        dt: float,
        wind_effect: float,
    ) -> None:
        old_pos = self.pos
        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist <= 1e-9:
            ctrl_vx, ctrl_vy = 0.0, 0.0
        else:
            ux = dx / dist
            uy = dy / dist
            ctrl_vx = self.max_speed * ux
            ctrl_vy = self.max_speed * uy

        wx, wy = env.wind_at(self.pos[0], self.pos[1], t)
        vx = ctrl_vx + wind_effect * wx
        vy = ctrl_vy + wind_effect * wy
        new_pos = (self.pos[0] + vx * dt, self.pos[1] + vy * dt)
        self.pos = new_pos
        self.clamp_to_map(env.map_width, env.map_height)
        self.vel = (vx, vy)
        self.task_status = "moving"
        self._update_distance_stat(old_pos, self.pos)
