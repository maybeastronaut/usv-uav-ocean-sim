from __future__ import annotations

import numpy as np


def exponential_decay(delta_t: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    return np.exp(-delta_t / tau)


def linear_decay(delta_t: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    return np.maximum(0.0, 1.0 - delta_t / tau)


def apply_decay(delta_t: np.ndarray, mode: str, tau: float) -> np.ndarray:
    if mode == "exponential":
        values = exponential_decay(delta_t, tau)
    elif mode == "linear":
        values = linear_decay(delta_t, tau)
    else:
        raise ValueError(f"unknown decay mode: {mode}")
    return np.clip(values, 0.0, 1.0)
