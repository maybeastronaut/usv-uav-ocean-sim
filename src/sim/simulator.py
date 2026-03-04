from __future__ import annotations

import math
import random
from dataclasses import dataclass

from sim.agents.uav import UAVAgent
from sim.agents.usv import USVAgent
from sim.config import SimConfig
from sim.coverage.coverage_grid import CoverageGrid
from sim.environment.environment import Environment2D
from sim.feedback.feedback import FeedbackController, FeedbackMonitor
from sim.pathing.path_planner import PathPlanner
from sim.policy.strategy import create_strategy
from sim.recharge.rendezvous import plan_rendezvous
from sim.tasks.region_map import RegionMap
from sim.tasks.task_generator import Task, TaskGenerator


@dataclass
class SimulatorResult:
    history: list[dict[str, float]]
    transition_counts: dict[str, int]
    pending_tasks: list[Task]


class Simulator:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.env = Environment2D(config)
        self.coverage = CoverageGrid(config)
        self.region_map = RegionMap(self.coverage, config)
        self.tasks = TaskGenerator(self.region_map, config)
        self._rng = random.Random(config.seed + 311)
        self.path_planner = PathPlanner(safe_margin=config.safe_margin)
        self.policy_name = self._resolve_policy_name()
        self.strategy = create_strategy(self.policy_name, seed=config.seed, config=config)

        self.agents: list[UAVAgent | USVAgent] = self._init_agents()
        self.agent_by_id = {agent.agent_id: agent for agent in self.agents}
        self.history: list[dict[str, float]] = []
        self.transition_counts: dict[str, int] = {
            "created": 0,
            "assigned": 0,
            "done": 0,
            "timeout_rollback": 0,
            "observe_calls_total": 0,
            "observe_updates": 0,
            "recharge_created": 0,
            "recharge_done": 0,
            "uav_emergency_events": 0,
            "fb_trigger_count_cum": 0,
        }
        self.usv_collision_count = 0
        self.out_of_bounds_count = 0
        self.current_time = 0.0
        self.enable_feedback = bool(getattr(config, "enable_feedback", False))
        self.feedback_monitor = FeedbackMonitor() if self.enable_feedback else None
        self.feedback_controller = FeedbackController(config) if self.enable_feedback else None
        self.feedback_softpart_override_until = -math.inf
        self.feedback_softpart_override_scale: float | None = None
        self.feedback_recharge_boost_until = -math.inf
        self.feedback_recharge_boost_mult = 1.0
        self.feedback_events: list[dict[str, object]] = []

    def run(
        self,
        t_end: float | None = None,
        dt: float | None = None,
        on_tick=None,
    ) -> SimulatorResult:
        sim_dt = dt if dt is not None else self.config.sim_dt
        end_time = t_end if t_end is not None else self.config.t_end
        if sim_dt <= 0.0:
            raise ValueError("sim dt must be > 0")
        if end_time < 0.0:
            raise ValueError("t_end must be >= 0")

        t = 0.0
        while t <= end_time + 1e-9:
            self.current_time = t
            rolled = self.tasks.update_timeouts(t, timeout=self.config.task_timeout)
            self.transition_counts["timeout_rollback"] += len(rolled)
            if rolled:
                rolled_set = set(rolled)
                for agent in self.agents:
                    if agent.current_task_id in rolled_set:
                        agent.clear_task()

            self._update_recharge_needs(t)
            self._assign_recharge_tasks(t)

            observe_calls_tick = self._step_agents(t=t, dt=sim_dt)
            self._complete_tasks_by_region_info(t)

            created = self.tasks.generate_tasks(t)
            self.transition_counts["created"] += len(created)

            self._update_recharge_needs(t)
            self._assign_recharge_tasks(t)
            self._assign_idle_agents(t, log_select=_is_multiple(t, 60.0))
            self._feedback_tick(t)

            self._log_tick(t, observe_calls_tick=observe_calls_tick)
            if _is_multiple(t, 60.0):
                summary = self.task_summary()
                mean_info = self.history[-1]["mean_info_all"] if self.history else 0.0
                print(
                    f"[SIM] t={t:.1f}s pending={summary['pending']} "
                    f"assigned={summary['assigned']} done={summary['done']} "
                    f"mean_info_all={mean_info:.4f}"
                )
                print(
                    f"[SIM] t={t:.1f}s observe_calls={observe_calls_tick} "
                    f"recharge_done={self.transition_counts['recharge_done']}"
                )
            if on_tick is not None:
                on_tick(self, t)
            t += sim_dt

        return SimulatorResult(
            history=self.history,
            transition_counts=self.transition_counts,
            pending_tasks=self.tasks.get_pending_tasks(task_type="monitor"),
        )

    def _init_agents(self) -> list[UAVAgent | USVAgent]:
        agents: list[UAVAgent | USVAgent] = []
        bx, by = self.env.base_position
        for i in range(self.config.num_uav):
            jx = self._rng.uniform(-40.0, 40.0)
            jy = self._rng.uniform(0.0, 40.0)
            pos = (min(max(bx + jx, 0.0), self.config.map_width), min(max(by + jy, 0.0), self.config.map_height))
            agents.append(
                UAVAgent(
                    agent_id=f"UAV-{i+1}",
                    pos=pos,
                    max_speed=self.config.uav_speed,
                    sensor_radius=self.config.uav_sensor_radius,
                    comm_radius=self.config.comm_radius,
                    energy=self.config.uav_battery_max,
                    low_energy_threshold=self.config.uav_low_battery_frac,
                    battery_max=self.config.uav_battery_max,
                    discharge_rate=self.config.uav_discharge_rate,
                    critical_battery_threshold=self.config.uav_critical_battery_frac,
                )
            )

        for i in range(self.config.num_usv):
            jx = self._rng.uniform(-60.0, 60.0)
            jy = self._rng.uniform(0.0, 30.0)
            pos = (min(max(bx + jx, 0.0), self.config.map_width), min(max(by + jy, 0.0), self.config.map_height))
            agents.append(
                USVAgent(
                    agent_id=f"USV-{i+1}",
                    pos=pos,
                    max_speed=self.config.usv_speed,
                    sensor_radius=self.config.usv_sensor_radius,
                    comm_radius=self.config.comm_radius,
                    turn_rate_deg=self.config.usv_turn_rate_deg,
                    heading=0.0,
                    charge_rate=self.config.usv_charge_rate,
                )
            )
        agents.sort(key=lambda a: a.agent_id)
        usvs = [agent for agent in agents if isinstance(agent, USVAgent)]
        n_usv = max(1, len(usvs))
        for idx, usv in enumerate(sorted(usvs, key=lambda a: a.agent_id)):
            left = idx * self.config.map_width / n_usv
            right = (idx + 1) * self.config.map_width / n_usv
            cx = 0.5 * (left + right)
            cy = 0.5 * self.config.map_height
            usv.stats["preferred_band_idx"] = float(idx)
            usv.stats["preferred_band_left"] = float(left)
            usv.stats["preferred_band_right"] = float(right)
            usv.stats["preferred_center_x"] = float(cx)
            usv.stats["preferred_center_y"] = float(cy)
            usv.stats["monitor_assign_count"] = 0.0
            usv.stats["softpart_hits"] = 0.0
            usv.stats["pref_distance_sum"] = 0.0
        return agents

    def _update_recharge_needs(self, t: float) -> None:
        for uav in self._uavs():
            if not uav.alive:
                continue
            active = self.tasks.find_recharge_task(uav.agent_id)
            if uav.battery <= 0.0 and uav.stats.get("emergency_logged", 0.0) < 1.0:
                self.transition_counts["uav_emergency_events"] += 1
                uav.stats["emergency_logged"] = 1.0

            if not uav.is_low_battery() and active is None:
                continue

            target_energy = self.config.recharge_target_frac * uav.battery_max
            if active is None:
                usv = self._select_recharge_usv(uav, exclude_task_id=None)
                rendezvous = self._safe_rendezvous(uav=uav, usv=usv)
                recharge_priority = self.config.recharge_task_priority * self._current_recharge_boost_mult(t)
                task, created = self.tasks.create_recharge_task(
                    uav_id=uav.agent_id,
                    usv_id=usv.agent_id if usv is not None else None,
                    rendezvous_pos=rendezvous,
                    t=t,
                    required_energy=target_energy,
                    priority=recharge_priority,
                )
                if created:
                    self.transition_counts["recharge_created"] += 1
                task.metadata["created_for_frac"] = uav.battery_frac
                continue

            # Keep rendezvous updated for critical UAV: USV should come to UAV.
            if active.status in ("pending", "assigned", "in_progress"):
                usv = self._select_recharge_usv(uav, exclude_task_id=active.task_id)
                if active.usv_id is None and usv is not None:
                    active.usv_id = usv.agent_id
                if active.usv_id is not None:
                    cur_usv = self.agent_by_id.get(active.usv_id)
                    if isinstance(cur_usv, USVAgent):
                        active.rendezvous_pos = self._safe_rendezvous(uav=uav, usv=cur_usv)
                        active.target_pos = active.rendezvous_pos
                if uav.is_critical():
                    active.rendezvous_pos = (uav.pos[0], uav.pos[1])
                    active.target_pos = active.rendezvous_pos
                active.required_energy = target_energy
                active.priority = self.config.recharge_task_priority * self._current_recharge_boost_mult(t)
                active.last_update_time = t

    def _assign_recharge_tasks(self, t: float) -> None:
        tasks = [
            task
            for task in self.tasks.all_tasks(task_type="recharge")
            if task.status in ("pending", "assigned", "in_progress")
        ]
        tasks.sort(key=lambda task: (-task.priority, task.created_time, task.task_id))

        for task in tasks:
            if task.uav_id is None:
                self.tasks.cancel_task(task.task_id, t)
                continue
            uav = self.agent_by_id.get(task.uav_id)
            if not isinstance(uav, UAVAgent) or not uav.alive:
                self.tasks.cancel_task(task.task_id, t)
                continue

            required = task.required_energy if task.required_energy is not None else self.config.recharge_target_frac * uav.battery_max
            if uav.battery >= required and task.status == "pending":
                self.tasks.mark_task_done(task.task_id, t=t)
                continue

            if task.usv_id is None:
                usv = self._select_recharge_usv(uav, exclude_task_id=task.task_id)
                if usv is None:
                    continue
                task.usv_id = usv.agent_id
            usv = self.agent_by_id.get(task.usv_id)
            if not isinstance(usv, USVAgent) or not usv.alive:
                task.usv_id = None
                continue
            if self._usv_has_other_recharge_task(usv.agent_id, task.task_id):
                continue

            if not self._prepare_agent_for_recharge(uav, t, task.task_id, allow_preempt=True):
                continue
            if not self._prepare_agent_for_recharge(
                usv,
                t,
                task.task_id,
                allow_preempt=self.config.allow_usv_preempt_monitor_for_recharge,
            ):
                continue

            task.status = "assigned"
            task.assigned_to = f"{uav.agent_id}|{usv.agent_id}"
            if task.assigned_time is None:
                task.assigned_time = t
            task.last_update_time = t
            if task.rendezvous_pos is None:
                task.rendezvous_pos = self._safe_rendezvous(uav=uav, usv=usv)
                task.target_pos = task.rendezvous_pos

            if uav.current_task_id != task.task_id:
                uav.set_task(task.task_id)
                uav.goal_pos = None
                uav.current_waypoints = []
                uav.current_wp_idx = 0
            if usv.current_task_id != task.task_id:
                usv.set_task(task.task_id)
                usv.goal_pos = None
                usv.current_waypoints = []
                usv.current_wp_idx = 0

    def _prepare_agent_for_recharge(self, agent: UAVAgent | USVAgent, t: float, recharge_task_id: int, allow_preempt: bool) -> bool:
        if agent.current_task_id is None:
            return True
        if agent.current_task_id == recharge_task_id:
            return True

        current = self.tasks.get_task(agent.current_task_id)
        if current is None:
            agent.clear_task()
            return True

        if current.task_type == "monitor" and allow_preempt:
            self.tasks.release_task(current.task_id, t)
            agent.clear_task()
            return True

        if current.task_type == "recharge":
            return current.task_id == recharge_task_id

        return False

    def _assign_idle_agents(self, t: float, log_select: bool = False) -> None:
        region_info = self.region_map.region_info_map(t)
        self.tasks.refresh_pending_priorities(region_info, t)
        pending_monitor = self.tasks.get_pending_tasks(task_type="monitor")
        pending_count = len(pending_monitor)
        mean_info_now = self.coverage.mean_info(t, mode="all")
        soft_scale = 1.0
        pending_frac_threshold = float(
            getattr(
                self.config,
                "pending_cross_threshold_frac",
                self.config.pending_cross_threshold / max(1.0, float(self.config.max_pending_tasks)),
            )
        )
        pending_trigger = pending_count > int(round(pending_frac_threshold * self.config.max_pending_tasks))
        cross_trigger = pending_trigger or mean_info_now < self.config.meaninfo_cross_threshold
        if cross_trigger:
            soft_scale = self.config.softpart_cross_scale
        if self._feedback_relax_active(t):
            cross_trigger = True
            if self.feedback_softpart_override_scale is not None:
                soft_scale = min(soft_scale, float(self.feedback_softpart_override_scale))
        if self.config.ablate_softpart:
            soft_scale = 0.0

        low_or_recharge_uav = [
            uav
            for uav in self._uavs()
            if uav.is_low_battery() or self.tasks.find_recharge_task(uav.agent_id) is not None
        ]
        recharge_pressure = len(low_or_recharge_uav) / max(1.0, float(self.config.num_uav))
        for usv in self._usvs():
            usv.stats["recharge_pressure"] = recharge_pressure
            usv.stats["softpart_scale"] = soft_scale

        idle_agents = [agent for agent in self.agents if agent.alive and agent.current_task_id is None]
        filtered: list[UAVAgent | USVAgent] = []
        for agent in idle_agents:
            if isinstance(agent, UAVAgent):
                if agent.is_low_battery() or self.tasks.find_recharge_task(agent.agent_id) is not None:
                    continue
            if isinstance(agent, USVAgent):
                if self._usv_reserved_by_recharge(agent.agent_id):
                    continue
            filtered.append(agent)
        idle_agents = filtered

        idle_agents.sort(key=lambda agent: agent.agent_id)

        if not idle_agents or not pending_monitor:
            return

        free_tasks: dict[int, Task] = {task.task_id: task for task in pending_monitor}
        free_agents: dict[str, UAVAgent | USVAgent] = {agent.agent_id: agent for agent in idle_agents}
        while free_tasks and free_agents:
            best: tuple[float, float, str, int] | None = None
            best_agent: UAVAgent | USVAgent | None = None
            best_task: Task | None = None

            for agent_id in sorted(free_agents):
                agent = free_agents[agent_id]
                candidate_task_ids = sorted(free_tasks)
                if isinstance(agent, USVAgent) and (not self.config.ablate_softpart) and (not cross_trigger):
                    preferred_ids = [
                        task_id
                        for task_id in candidate_task_ids
                        if self._task_in_usv_preferred_band(agent, free_tasks[task_id])
                    ]
                    if preferred_ids:
                        candidate_task_ids = preferred_ids
                for task_id in candidate_task_ids:
                    task = free_tasks[task_id]
                    pair_score = float(self.strategy.pair_score(agent, task, region_info, t))
                    if not math.isfinite(pair_score):
                        continue
                    dist = agent.distance_to(task.target_pos)
                    rank = (pair_score, -dist, -task_id, agent_id)
                    if best is None or rank > best:
                        best = rank
                        best_agent = agent
                        best_task = task

            if best_agent is None or best_task is None:
                break

            if log_select:
                dist = best_agent.distance_to(best_task.target_pos)
                print(
                    f"[select] t={t:.1f}, agent={best_agent.agent_id}, strategy={self.policy_name}, "
                    f"pick task_id={best_task.task_id}, region={best_task.region_id}, "
                    f"priority={best_task.priority:.3f}, dist={dist:.1f}"
                )

            ok = self.tasks.assign_task(task_id=best_task.task_id, agent_id=best_agent.agent_id, t=t)
            if not ok:
                del free_tasks[best_task.task_id]
                continue

            best_agent.set_task(best_task.task_id)
            best_agent.task_status = "moving"
            best_agent.goal_pos = None
            best_agent.current_waypoints = []
            best_agent.current_wp_idx = 0
            if best_task.region_id is not None:
                best_agent.stats["last_monitor_rx"] = float(best_task.region_id[0])
                best_agent.stats["last_monitor_ry"] = float(best_task.region_id[1])
            if isinstance(best_agent, USVAgent):
                self._update_usv_softpart_stats(best_agent, best_task)

            self.transition_counts["assigned"] += 1
            del free_tasks[best_task.task_id]
            del free_agents[best_agent.agent_id]

    def _step_agents(self, t: float, dt: float) -> int:
        observe_calls = 0

        recharge_tasks = [
            task
            for task in self.tasks.all_tasks(task_type="recharge")
            if task.status in ("assigned", "in_progress") and task.uav_id is not None and task.usv_id is not None
        ]
        handled_agents: set[str] = set()
        for task in sorted(recharge_tasks, key=lambda item: item.task_id):
            uav = self.agent_by_id.get(task.uav_id)
            usv = self.agent_by_id.get(task.usv_id)
            if not isinstance(uav, UAVAgent) or not isinstance(usv, USVAgent):
                self.tasks.cancel_task(task.task_id, t)
                continue
            if not uav.alive or not usv.alive:
                self.tasks.cancel_task(task.task_id, t)
                continue
            self._step_recharge_pair(task, uav, usv, t=t, dt=dt)
            handled_agents.add(uav.agent_id)
            handled_agents.add(usv.agent_id)

        for agent in sorted(self.agents, key=lambda item: item.agent_id):
            if not agent.alive:
                continue
            if agent.agent_id in handled_agents:
                continue
            agent.last_seen_time = t
            if agent.current_task_id is None:
                self._step_idle_agent(agent, t=t, dt=dt)
                if isinstance(agent, UAVAgent):
                    self._update_uav_energy_post_move(agent, dt)
                continue

            task = self.tasks.get_task(agent.current_task_id)
            if task is None:
                agent.clear_task()
                continue
            if task.task_type != "monitor":
                continue
            if task.status != "assigned" or task.assigned_to != agent.agent_id:
                agent.clear_task()
                continue

            goal_pos = self.strategy.select_goal(agent, task, self.env, t) or task.target_pos
            self._move_agent_to_goal(agent, goal_pos, t=t, dt=dt)

            if not self.env.is_inside_map(agent.pos[0], agent.pos[1]):
                self.out_of_bounds_count += 1
                agent.clamp_to_map(self.config.map_width, self.config.map_height)

            if isinstance(agent, UAVAgent):
                self._update_uav_energy_post_move(agent, dt)

            if isinstance(agent, UAVAgent) and agent.battery <= 0.0:
                agent.task_status = "emergency"
                continue

            if agent.distance_to(task.target_pos) <= agent.sensor_radius:
                observe_calls += 1
                self.transition_counts["observe_calls_total"] += 1
                changed = self.coverage.observe(agent.pos, agent.sensor_radius, t)
                agent.task_status = "working"
                agent.stats["work_ticks"] = agent.stats.get("work_ticks", 0.0) + 1.0
                if changed > 0:
                    self.transition_counts["observe_updates"] += int(changed)
            else:
                agent.task_status = "moving"

        return observe_calls

    def _step_recharge_pair(self, task: Task, uav: UAVAgent, usv: USVAgent, t: float, dt: float) -> None:
        rendezvous = task.rendezvous_pos or task.target_pos
        if uav.is_critical():
            rendezvous = (uav.pos[0], uav.pos[1])
            task.rendezvous_pos = rendezvous
            task.target_pos = rendezvous

        self._move_agent_to_goal(usv, rendezvous, t=t, dt=dt)

        if uav.battery > 0.0:
            self._move_agent_to_goal(uav, rendezvous, t=t, dt=dt)
            self._update_uav_energy_post_move(uav, dt)
        else:
            uav.vel = (0.0, 0.0)
            uav.task_status = "emergency"

        pair_dist = _distance(uav.pos, usv.pos)
        if pair_dist > self.config.rendezvous_eps:
            uav.task_status = "waiting_recharge"
            usv.task_status = "waiting_recharge"
            task.status = "assigned"
            task.assigned_to = f"{uav.agent_id}|{usv.agent_id}"
            task.last_update_time = t
            return

        # Charging phase.
        task.status = "in_progress"
        task.assigned_to = f"{uav.agent_id}|{usv.agent_id}"
        task.last_update_time = t
        uav.task_status = "charging"
        usv.task_status = "charging"
        usv.charging_uav_id = uav.agent_id
        uav.vel = (0.0, 0.0)
        usv.vel = (0.0, 0.0)

        uav.recharge(dt=dt, charge_rate=usv.charge_rate)
        required = task.required_energy if task.required_energy is not None else self.config.recharge_target_frac * uav.battery_max
        if uav.battery >= required:
            self.tasks.mark_task_done(task.task_id, t=t)
            self.transition_counts["recharge_done"] += 1
            uav.clear_task()
            usv.clear_task()
            usv.charging_uav_id = None

    def _step_idle_agent(self, agent: UAVAgent | USVAgent, t: float, dt: float) -> None:
        goal = None
        if isinstance(agent, UAVAgent) and agent.is_low_battery():
            usv = self._nearest_usv(agent.pos)
            goal = usv.pos if usv is not None else self.env.base_position
        else:
            goal = self.strategy.select_goal(agent, None, self.env, t)

        if goal is None:
            agent.task_status = "idle"
            agent.vel = (0.0, 0.0)
            return

        self._move_agent_to_goal(agent, goal, t=t, dt=dt)

    def _move_agent_to_goal(self, agent: UAVAgent | USVAgent, goal: tuple[float, float], t: float, dt: float) -> None:
        if agent.goal_pos != goal or not agent.current_waypoints:
            agent.goal_pos = goal
            agent.current_waypoints = self.path_planner.plan(agent=agent, goal_pos=goal, env=self.env, t=t)
            agent.current_wp_idx = 0

        wp = self.path_planner.next_waypoint(agent, agent.current_waypoints, t)
        if wp is None:
            agent.task_status = "idle"
            agent.vel = (0.0, 0.0)
            return
        if agent.distance_to(wp) <= self.config.wp_reached_eps:
            if agent.current_wp_idx < len(agent.current_waypoints) - 1:
                agent.current_wp_idx += 1
                wp = agent.current_waypoints[agent.current_wp_idx]
            else:
                wp = goal

        if isinstance(agent, UAVAgent):
            agent.step_toward(
                target_pos=wp,
                env=self.env,
                t=t,
                dt=dt,
                wind_effect=self.config.wind_effect_uav,
            )
        else:
            old_pos = agent.pos
            agent.step_toward(
                target_pos=wp,
                env=self.env,
                t=t,
                dt=dt,
                current_effect=self.config.current_effect_usv,
                avoidance_angles_deg=self.config.obstacle_avoidance_angles,
            )
            if self.env.is_in_obstacle(agent.pos[0], agent.pos[1]):
                self.usv_collision_count += 1
                agent.pos = old_pos

        if not self.env.is_inside_map(agent.pos[0], agent.pos[1]):
            self.out_of_bounds_count += 1
            agent.clamp_to_map(self.config.map_width, self.config.map_height)

    def _update_uav_energy_post_move(self, uav: UAVAgent, dt: float) -> None:
        speed = math.hypot(uav.vel[0], uav.vel[1])
        moved = speed > 1e-6 and uav.task_status != "charging"
        if moved:
            uav.discharge(dt=dt, moving=True)
        if uav.battery <= 0.0:
            uav.battery = 0.0
            uav.vel = (0.0, 0.0)
            uav.task_status = "emergency"

    def _complete_tasks_by_region_info(self, t: float) -> None:
        info = self.region_map.region_info_map(t)
        to_complete_regions: set[tuple[int, int]] = set()
        for task in self.tasks.get_assigned_tasks(task_type="monitor"):
            if task.region_id is None:
                continue
            rx, ry = task.region_id
            if float(info[ry, rx]) >= self.config.task_complete_threshold:
                to_complete_regions.add((rx, ry))

        if not to_complete_regions:
            return

        done_task_ids: set[int] = set()
        for region_id in sorted(to_complete_regions):
            done_ids = self.tasks.complete_tasks_for_region(region_id, t=t)
            done_task_ids.update(done_ids)
            self.transition_counts["done"] += len(done_ids)

        if done_task_ids:
            for agent in self.agents:
                if agent.current_task_id in done_task_ids:
                    agent.clear_task()

    def _log_tick(self, t: float, observe_calls_tick: int) -> None:
        pending_count = len(self.tasks.get_pending_tasks(task_type="monitor"))
        assigned_count = len(self.tasks.get_assigned_tasks(task_type="monitor"))
        done_count = len([task for task in self.tasks.all_tasks(task_type="monitor") if task.status == "done"])
        pending_recharge = len(self.tasks.get_pending_tasks(task_type="recharge"))
        assigned_recharge = len(self.tasks.get_assigned_tasks(task_type="recharge")) + len(
            self.tasks.get_in_progress_tasks(task_type="recharge")
        )
        uavs = self._uavs()
        batteries = [uav.battery_frac for uav in uavs]
        battery_min = min(batteries) if batteries else 0.0
        battery_mean = sum(batteries) / len(batteries) if batteries else 0.0

        row = {
            "time": float(t),
            "pending_count": float(pending_count),
            "assigned_count": float(assigned_count),
            "done_count": float(done_count),
            "pending_recharge_count": float(pending_recharge),
            "assigned_recharge_count": float(assigned_recharge),
            "observe_calls_tick": float(observe_calls_tick),
            "mean_info_all": self.coverage.mean_info(t, mode="all"),
            "min_info_visited": self.coverage.min_info(t, mode="visited"),
            "p5_info_visited": self.coverage.percentile_info(t, 5.0, mode="visited"),
            "recharge_count_cum": float(self.transition_counts["recharge_done"]),
            "uav_battery_min": float(battery_min),
            "uav_battery_mean": float(battery_mean),
            "uav_dead_count": float(self.transition_counts["uav_emergency_events"]),
            "fb_trigger_count_cum": float(self.transition_counts["fb_trigger_count_cum"]),
            "fb_relax_active": 1.0 if self._feedback_relax_active(t) else 0.0,
            "fb_recharge_boost_active": 1.0 if self._feedback_recharge_boost_active(t) else 0.0,
            "usv_preference_hit_rate": float(self.usv_preference_hit_rate()),
            "usv_cross_band_ratio": float(self.usv_cross_band_ratio()),
        }
        self.history.append(row)

    def _feedback_tick(self, t: float) -> None:
        if not self.enable_feedback or self.feedback_monitor is None or self.feedback_controller is None:
            return

        metrics = self.feedback_monitor.update(self, t)
        actions = self.feedback_controller.step(metrics, self)
        for action in actions:
            self._apply_feedback_action(action, t)

    def _apply_feedback_action(self, action: dict[str, object], t: float) -> None:
        action_type = str(action.get("type", "UNKNOWN"))
        reason = str(action.get("reason", "n/a"))

        if action_type == "RELAX_SOFTPART":
            self.feedback_softpart_override_scale = float(action.get("scale", 0.2))
            duration = float(action.get("duration", 0.0))
            self.feedback_softpart_override_until = max(self.feedback_softpart_override_until, t + duration)
            self.feedback_events.append(
                {"t": float(t), "action": action_type, "reason": reason, "duration": duration}
            )
            self.transition_counts["fb_trigger_count_cum"] += 1
            cooldown_remaining = self._feedback_cooldown_remaining("RELAX_SOFTPART", t)
            duration_remaining = max(0.0, self.feedback_softpart_override_until - t)
            print(
                f"[FB] t={t:.1f}s action={action_type} reason={reason} "
                f"duration={duration:.1f}s scale={self.feedback_softpart_override_scale:.2f} "
                f"cooldown_remaining={cooldown_remaining:.1f}s duration_remaining={duration_remaining:.1f}s"
            )
            return

        if action_type == "BOOST_RECHARGE_PRIORITY":
            self.feedback_recharge_boost_mult = float(action.get("mult", 1.0))
            duration = float(action.get("duration", 0.0))
            self.feedback_recharge_boost_until = max(self.feedback_recharge_boost_until, t + duration)
            self.feedback_events.append(
                {"t": float(t), "action": action_type, "reason": reason, "duration": duration}
            )
            self.transition_counts["fb_trigger_count_cum"] += 1
            cooldown_remaining = self._feedback_cooldown_remaining("BOOST_RECHARGE_PRIORITY", t)
            duration_remaining = max(0.0, self.feedback_recharge_boost_until - t)
            print(
                f"[FB] t={t:.1f}s action={action_type} reason={reason} "
                f"duration={duration:.1f}s mult={self.feedback_recharge_boost_mult:.2f} "
                f"cooldown_remaining={cooldown_remaining:.1f}s duration_remaining={duration_remaining:.1f}s"
            )
            return

        if action_type == "GLOBAL_REASSIGN":
            mode = str(action.get("mode", "soft")).lower()
            released = 0
            if mode == "hard":
                for task in self.tasks.get_assigned_tasks(task_type="monitor"):
                    if self.tasks.release_task(task.task_id, t):
                        released += 1
                for agent in self.agents:
                    if agent.current_task_id is None:
                        continue
                    task = self.tasks.get_task(agent.current_task_id)
                    if task is not None and task.task_type == "monitor":
                        agent.clear_task()
            else:
                for agent in self.agents:
                    if agent.current_task_id is None:
                        agent.goal_pos = None
                        agent.current_waypoints = []
                        agent.current_wp_idx = 0
                        continue
                    task = self.tasks.get_task(agent.current_task_id)
                    if task is None or task.task_type != "monitor":
                        continue
                    agent.goal_pos = None
                    agent.current_waypoints = []
                    agent.current_wp_idx = 0

            self._assign_idle_agents(t, log_select=False)
            self.feedback_events.append(
                {"t": float(t), "action": action_type, "reason": reason, "mode": mode, "released": released}
            )
            self.transition_counts["fb_trigger_count_cum"] += 1
            cooldown_remaining = self._feedback_cooldown_remaining("GLOBAL_REASSIGN", t)
            print(
                f"[FB] t={t:.1f}s action={action_type} reason={reason} "
                f"mode={mode} released={released} "
                f"cooldown_remaining={cooldown_remaining:.1f}s duration_remaining=0.0s"
            )
            return

    def _feedback_relax_active(self, t: float) -> bool:
        if not self.enable_feedback:
            return False
        return t <= self.feedback_softpart_override_until

    def _feedback_recharge_boost_active(self, t: float) -> bool:
        if not self.enable_feedback:
            return False
        return t <= self.feedback_recharge_boost_until

    def _current_recharge_boost_mult(self, t: float) -> float:
        if not self._feedback_recharge_boost_active(t):
            self.feedback_recharge_boost_mult = 1.0
            return 1.0
        return max(1.0, float(self.feedback_recharge_boost_mult))

    def latest_feedback_event(self) -> dict[str, object] | None:
        if not self.feedback_events:
            return None
        return self.feedback_events[-1]

    def _feedback_cooldown_remaining(self, action_type: str, t: float) -> float:
        if self.feedback_controller is None:
            return 0.0
        until = float(self.feedback_controller.cooldown_until.get(action_type, t))
        return max(0.0, until - t)

    def _safe_rendezvous(self, uav: UAVAgent, usv: USVAgent | None) -> tuple[float, float]:
        if usv is None or uav.is_critical():
            return (uav.pos[0], uav.pos[1])
        return plan_rendezvous(uav=uav, usv=usv, env=self.env, safety_margin=self.config.safe_margin)

    def _resolve_policy_name(self) -> str:
        policy = (getattr(self.config, "task_policy", "") or "").strip().lower()
        strategy = (getattr(self.config, "strategy", "") or "").strip().lower()
        if strategy and strategy != "priority" and (not policy or policy == "priority"):
            return strategy
        if policy:
            return policy
        if strategy:
            return strategy
        return "priority"

    def _update_usv_softpart_stats(self, usv: USVAgent, task: Task) -> None:
        if task.region_id is None:
            return
        stats = usv.stats
        stats["monitor_assign_count"] = stats.get("monitor_assign_count", 0.0) + 1.0
        tx = float(task.target_pos[0])
        hit = self._x_in_usv_preferred_band(usv, tx)
        if hit:
            stats["softpart_hits"] = stats.get("softpart_hits", 0.0) + 1.0
        cx = float(stats.get("preferred_center_x", tx))
        stats["pref_distance_sum"] = stats.get("pref_distance_sum", 0.0) + abs(tx - cx)

    def _x_in_usv_preferred_band(self, usv: USVAgent, x: float) -> bool:
        left = float(usv.stats.get("preferred_band_left", 0.0))
        right = float(usv.stats.get("preferred_band_right", self.config.map_width))
        return left <= x < right or (abs(x - self.config.map_width) <= 1e-6 and abs(right - self.config.map_width) <= 1e-6)

    def _task_in_usv_preferred_band(self, usv: USVAgent, task: Task) -> bool:
        return self._x_in_usv_preferred_band(usv, float(task.target_pos[0]))

    def _select_recharge_usv(self, uav: UAVAgent, exclude_task_id: int | None) -> USVAgent | None:
        busy_usv_ids: set[str] = set()
        for task in self.tasks.all_tasks(task_type="recharge"):
            if task.status not in ("pending", "assigned", "in_progress"):
                continue
            if task.task_id == exclude_task_id:
                continue
            if task.usv_id is not None:
                busy_usv_ids.add(task.usv_id)

        candidates: list[USVAgent] = []
        for usv in self._usvs():
            if not usv.alive:
                continue
            if usv.agent_id in busy_usv_ids:
                continue
            candidates.append(usv)
        if not candidates:
            return None
        candidates.sort(key=lambda usv: (_distance(uav.pos, usv.pos), usv.agent_id))
        return candidates[0]

    def _nearest_usv(self, pos: tuple[float, float]) -> USVAgent | None:
        usvs = [usv for usv in self._usvs() if usv.alive]
        if not usvs:
            return None
        usvs.sort(key=lambda usv: (_distance(pos, usv.pos), usv.agent_id))
        return usvs[0]

    def _usv_reserved_by_recharge(self, usv_id: str) -> bool:
        for task in self.tasks.all_tasks(task_type="recharge"):
            if task.status not in ("pending", "assigned", "in_progress"):
                continue
            if task.usv_id == usv_id:
                return True
        return False

    def _usv_has_other_recharge_task(self, usv_id: str, task_id: int) -> bool:
        for task in self.tasks.all_tasks(task_type="recharge"):
            if task.task_id == task_id:
                continue
            if task.status not in ("assigned", "in_progress"):
                continue
            if task.usv_id == usv_id:
                return True
        return False

    def _uavs(self) -> list[UAVAgent]:
        return [agent for agent in self.agents if isinstance(agent, UAVAgent)]

    def _usvs(self) -> list[USVAgent]:
        return [agent for agent in self.agents if isinstance(agent, USVAgent)]

    def agent_positions(self) -> dict[str, tuple[float, float]]:
        return {agent.agent_id: agent.pos for agent in self.agents}

    def task_summary(self) -> dict[str, int]:
        return {
            "pending": len(self.tasks.get_pending_tasks(task_type="monitor")),
            "assigned": len(self.tasks.get_assigned_tasks(task_type="monitor")),
            "done": len([task for task in self.tasks.all_tasks(task_type="monitor") if task.status == "done"]),
        }

    def pending_tasks(self) -> list[Task]:
        return self.tasks.get_pending_tasks(task_type="monitor")

    def recharge_tasks(self) -> list[Task]:
        return [
            task
            for task in self.tasks.all_tasks(task_type="recharge")
            if task.status in ("pending", "assigned", "in_progress")
        ]

    def assigned_task_links(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        links: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for agent in self.agents:
            if agent.current_task_id is None:
                continue
            task = self.tasks.get_task(agent.current_task_id)
            if task is None:
                continue
            if task.task_type != "monitor":
                continue
            if task.status != "assigned":
                continue
            links.append((agent.pos, task.target_pos))
        return links

    def recharge_links(self) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        links: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
        for task in self.recharge_tasks():
            if task.uav_id is None or task.usv_id is None:
                continue
            uav = self.agent_by_id.get(task.uav_id)
            usv = self.agent_by_id.get(task.usv_id)
            if not isinstance(uav, UAVAgent) or not isinstance(usv, USVAgent):
                continue
            rv = task.rendezvous_pos or task.target_pos
            links.append((uav.pos, usv.pos, rv))
        return links

    def usv_preference_hit_rate(self) -> float:
        hits = 0.0
        total = 0.0
        for usv in self._usvs():
            hits += usv.stats.get("softpart_hits", 0.0)
            total += usv.stats.get("monitor_assign_count", 0.0)
        if total <= 0.0:
            return 0.0
        return float(hits / total)

    def usv_cross_band_ratio(self) -> float:
        hits = 0.0
        total = 0.0
        for usv in self._usvs():
            hits += usv.stats.get("softpart_hits", 0.0)
            total += usv.stats.get("monitor_assign_count", 0.0)
        if total <= 0.0:
            return 0.0
        cross = max(0.0, total - hits)
        return float(cross / total)

    def usv_softpart_layout(self) -> list[dict[str, float]]:
        layout: list[dict[str, float]] = []
        for usv in sorted(self._usvs(), key=lambda a: a.agent_id):
            layout.append(
                {
                    "left": float(usv.stats.get("preferred_band_left", 0.0)),
                    "right": float(usv.stats.get("preferred_band_right", self.config.map_width)),
                    "center_x": float(usv.stats.get("preferred_center_x", 0.0)),
                    "center_y": float(usv.stats.get("preferred_center_y", 0.0)),
                }
            )
        return layout

    def final_region_info(self) -> tuple[float, object]:
        final_t = self.history[-1]["time"] if self.history else 0.0
        return (final_t, self.region_map.region_info_map(final_t))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_multiple(t: float, base: float, tol: float = 1e-6) -> bool:
    if base <= 0.0:
        return False
    k = round(t / base)
    return abs(t - k * base) <= tol
