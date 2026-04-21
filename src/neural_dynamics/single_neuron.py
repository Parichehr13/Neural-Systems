"""Single-neuron integrate-and-fire models used across experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import (
    build_time_vector,
    firing_rate_hz,
    first_spike_latency_ms,
    inter_spike_intervals,
    steady_state_rate_hz,
)


@dataclass(frozen=True)
class FixedThresholdParams:
    resting_potential_mv: float = -65.0
    threshold_mv: float = -50.0
    membrane_time_constant_ms: float = 30.0
    membrane_resistance_mohm: float = 10.0
    dt_ms: float = 0.05
    duration_ms: float = 300.0


@dataclass(frozen=True)
class DynamicThresholdParams:
    resting_potential_mv: float = -65.0
    baseline_threshold_mv: float = -55.0
    spike_threshold_mv: float = 0.0
    membrane_time_constant_ms: float = 30.0
    threshold_time_constant_ms: float = 10.0
    membrane_resistance_mohm: float = 10.0
    dt_ms: float = 0.05
    duration_ms: float = 300.0


def rectified_sinusoid_current(time_ms: np.ndarray, max_current_na: float = 4.0) -> np.ndarray:
    """Example time-varying current used for visualization."""
    return np.abs(max_current_na * np.sin(np.pi * time_ms / time_ms[-1]))


def _current_vector(
    current_na: float | Sequence[float] | np.ndarray,
    time_ms: np.ndarray,
) -> np.ndarray:
    if np.isscalar(current_na):
        return np.full(time_ms.size, float(current_na), dtype=float)
    current = np.asarray(current_na, dtype=float)
    if current.size != time_ms.size:
        raise ValueError("Current vector must match the time vector length.")
    return current


def simulate_fixed_threshold_neuron(
    current_na: float | Sequence[float] | np.ndarray,
    params: FixedThresholdParams,
    transient_ms: float = 50.0,
) -> dict[str, np.ndarray | float | int]:
    """Simulate a leaky integrate-and-fire neuron with a fixed threshold."""
    time_ms = build_time_vector(params.duration_ms, params.dt_ms)
    current = _current_vector(current_na, time_ms)
    voltage_mv = np.zeros_like(time_ms)
    voltage_mv[0] = params.resting_potential_mv

    spike_indices: list[int] = []
    for index in range(time_ms.size - 1):
        steady_state_voltage_mv = (
            params.resting_potential_mv + params.membrane_resistance_mohm * current[index]
        )
        voltage_mv[index + 1] = (
            (voltage_mv[index] - steady_state_voltage_mv)
            * np.exp(-params.dt_ms / params.membrane_time_constant_ms)
            + steady_state_voltage_mv
        )
        if voltage_mv[index + 1] >= params.threshold_mv:
            voltage_mv[index + 1] = params.resting_potential_mv
            spike_indices.append(index + 1)

    spike_times_ms = time_ms[np.asarray(spike_indices, dtype=int)]
    isi_ms = inter_spike_intervals(spike_times_ms)

    return {
        "time_ms": time_ms,
        "input_current_na": current,
        "voltage_mv": voltage_mv,
        "threshold_mv": np.full(time_ms.size, params.threshold_mv, dtype=float),
        "spike_times_ms": spike_times_ms,
        "spike_count": int(spike_times_ms.size),
        "isi_ms": isi_ms,
        "mean_rate_hz": firing_rate_hz(spike_times_ms, params.duration_ms, transient_ms=transient_ms),
        "steady_rate_hz": steady_state_rate_hz(spike_times_ms),
        "first_spike_latency_ms": first_spike_latency_ms(spike_times_ms),
    }


def simulate_dynamic_threshold_neuron(
    current_na: float,
    params: DynamicThresholdParams,
    transient_ms: float = 50.0,
) -> dict[str, np.ndarray | float | int]:
    """Simulate a leaky integrate-and-fire neuron with a dynamic threshold."""
    time_ms = build_time_vector(params.duration_ms, params.dt_ms)
    voltage_mv = np.zeros_like(time_ms)
    threshold_mv = np.zeros_like(time_ms)
    voltage_mv[0] = params.resting_potential_mv
    threshold_mv[0] = params.baseline_threshold_mv

    spike_indices: list[int] = []
    steady_state_voltage_mv = (
        params.resting_potential_mv + params.membrane_resistance_mohm * current_na
    )
    for index in range(time_ms.size - 1):
        voltage_mv[index + 1] = (
            (voltage_mv[index] - steady_state_voltage_mv)
            * np.exp(-params.dt_ms / params.membrane_time_constant_ms)
            + steady_state_voltage_mv
        )
        threshold_mv[index + 1] = (
            (threshold_mv[index] - params.baseline_threshold_mv)
            * np.exp(-params.dt_ms / params.threshold_time_constant_ms)
            + params.baseline_threshold_mv
        )
        if voltage_mv[index + 1] >= threshold_mv[index + 1]:
            voltage_mv[index + 1] = params.resting_potential_mv
            threshold_mv[index + 1] = params.spike_threshold_mv
            spike_indices.append(index + 1)

    spike_times_ms = time_ms[np.asarray(spike_indices, dtype=int)]
    isi_ms = inter_spike_intervals(spike_times_ms)

    return {
        "time_ms": time_ms,
        "input_current_na": np.full(time_ms.size, current_na, dtype=float),
        "voltage_mv": voltage_mv,
        "threshold_mv": threshold_mv,
        "spike_times_ms": spike_times_ms,
        "spike_count": int(spike_times_ms.size),
        "isi_ms": isi_ms,
        "mean_rate_hz": firing_rate_hz(spike_times_ms, params.duration_ms, transient_ms=transient_ms),
        "steady_rate_hz": steady_state_rate_hz(spike_times_ms),
        "first_spike_latency_ms": first_spike_latency_ms(spike_times_ms),
    }
