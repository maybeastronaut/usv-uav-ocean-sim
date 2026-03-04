from __future__ import annotations

import math
from typing import Any


def maybe_trigger_failure(sim_state: Any, t: float, dt: float) -> dict[str, object] | None:
    cfg = sim_state.config
    if not bool(getattr(cfg, "enable_failures", False)):
        return None

    last_t = float(getattr(sim_state, "failure_last_trigger_t", -math.inf))
    cooldown = float(getattr(cfg, "failure_event_cooldown", 999999.0))
    if t - last_t < cooldown:
        return None

    mode = str(getattr(cfg, "failure_mode", "scheduled")).strip().lower()
    usvs = [usv for usv in sim_state._usvs() if usv.health_state != "DISABLED"]
    if not usvs:
        return None

    target_usv = None
    if mode == "scheduled":
        failure_t = float(getattr(cfg, "failure_t_sec", 600.0))
        if t + 1e-9 < failure_t:
            return None
        target_usv = _find_scheduled_target(sim_state, usvs)
    elif mode == "random":
        p = float(getattr(cfg, "failure_random_prob_per_sec", 0.0))
        trigger_prob = max(0.0, min(1.0, p * max(0.0, dt)))
        if sim_state._rng.random() > trigger_prob:
            return None
        target_usv = sorted(usvs, key=lambda a: a.agent_id)[sim_state._rng.randrange(len(usvs))]
    else:
        return None

    if target_usv is None:
        return None

    kind = str(getattr(cfg, "failure_kind", "DISABLED")).strip().upper()
    if kind not in ("DAMAGED", "DISABLED"):
        kind = "DISABLED"

    return {
        "t": float(t),
        "usv_id": target_usv.agent_id,
        "kind": kind,
    }


def _find_scheduled_target(sim_state: Any, candidates: list[Any]) -> Any | None:
    target = getattr(sim_state.config, "failure_usv_id", 1)
    target_str = str(target)
    wanted = target_str if target_str.startswith("USV-") else f"USV-{target_str}"
    for usv in candidates:
        if usv.agent_id == wanted:
            return usv
    ordered = sorted(candidates, key=lambda a: a.agent_id)
    return ordered[0] if ordered else None
