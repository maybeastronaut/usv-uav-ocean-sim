from __future__ import annotations

from sim.config import SimConfig
from sim.feedback.feedback import FeedbackController


def _metrics(
    *,
    t: float,
    mean_info_all: float = 0.2,
    p5_info_all: float = 0.2,
    pending_count: float = 10.0,
    uav_battery_mean: float = 0.8,
    uav_battery_min: float = 0.7,
) -> dict[str, float]:
    return {
        "t": float(t),
        "mean_info_all": float(mean_info_all),
        "p5_info_all": float(p5_info_all),
        "min_info_all": float(min(mean_info_all, p5_info_all)),
        "pending_count": float(pending_count),
        "assigned_count": 0.0,
        "done_count_cum": 0.0,
        "recharge_count_cum": 0.0,
        "uav_battery_min": float(uav_battery_min),
        "uav_battery_mean": float(uav_battery_mean),
        "usv_preference_hit_rate": 0.0,
        "usv_cross_band_ratio": 0.0,
    }


def test_relax_softpart_trigger() -> None:
    cfg = SimConfig(enable_feedback=True, fb_cooldown_sec=120.0, fb_meaninfo_low=0.08, fb_p5info_low=0.05)
    controller = FeedbackController(cfg)

    actions = controller.step(
        _metrics(t=0.0, mean_info_all=0.04, p5_info_all=0.02, pending_count=20.0),
        sim_state=None,
    )
    types = {a["type"] for a in actions}
    assert "RELAX_SOFTPART" in types


def test_cooldown() -> None:
    cfg = SimConfig(
        enable_feedback=True,
        fb_cooldown_sec=120.0,
        fb_cooldown_relax=120.0,
        fb_meaninfo_low=0.08,
        fb_p5info_low=0.05,
    )
    controller = FeedbackController(cfg)

    first = controller.step(
        _metrics(t=0.0, mean_info_all=0.04, p5_info_all=0.02, pending_count=20.0),
        sim_state=None,
    )
    assert any(a["type"] == "RELAX_SOFTPART" for a in first)

    second = controller.step(
        _metrics(t=60.0, mean_info_all=0.03, p5_info_all=0.02, pending_count=20.0),
        sim_state=None,
    )
    assert not any(a["type"] == "RELAX_SOFTPART" for a in second)

    third = controller.step(
        _metrics(t=130.0, mean_info_all=0.03, p5_info_all=0.02, pending_count=20.0),
        sim_state=None,
    )
    assert any(a["type"] == "RELAX_SOFTPART" for a in third)
