"""Adaptive integrate-and-fire neuron model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import (
    adaptation_index,
    build_time_vector,
    firing_rate_hz,
    first_spike_latency_ms,
    inter_spike_intervals,
    steady_state_rate_hz,
)


@dataclass(frozen=True)
class AdaptiveNeuronParams:
    resting_potential_mv: float = -65.0
    membrane_resistance_mohm: float = 10.0
    membrane_time_constant_ms: float = 30.0
    baseline_threshold_mv: float = -55.0
    spike_threshold_mv: float = 50.0
    threshold_time_constant_ms: float = 10.0
    adaptation_reversal_mv: float = -90.0
    adaptation_time_constant_ms: float = 700.0
    adaptation_increment: float = 0.10
    resistance_times_gamax: float = 2.0
    dt_ms: float = 0.05
    duration_ms: float = 800.0

    @property
    def capacitance_nf(self) -> float:
        return self.membrane_time_constant_ms / self.membrane_resistance_mohm

    @property
    def leak_conductance_us(self) -> float:
        return 1.0 / self.membrane_resistance_mohm

    @property
    def max_adaptation_conductance_us(self) -> float:
        return self.resistance_times_gamax / self.membrane_resistance_mohm


def simulate_adaptive_neuron(
    current_na: float,
    params: AdaptiveNeuronParams,
    with_adaptation: bool = True,
    transient_ms: float = 100.0,
) -> dict[str, np.ndarray | float | int]:
    """Simulate a neuron with dynamic threshold and optional adaptation."""
    time_ms = build_time_vector(params.duration_ms, params.dt_ms)
    voltage_mv = np.empty(time_ms.size)
    threshold_mv = np.empty(time_ms.size)
    adaptation_state = np.empty(time_ms.size)
    spike_train = np.zeros(time_ms.size, dtype=int)

    voltage_mv[0] = params.resting_potential_mv
    threshold_mv[0] = params.baseline_threshold_mv
    adaptation_state[0] = 0.0

    spike_indices: list[int] = []
    for index in range(time_ms.size - 1):
        adaptation_conductance_us = (
            params.max_adaptation_conductance_us * adaptation_state[index]
            if with_adaptation
            else 0.0
        )
        equivalent_conductance_us = params.leak_conductance_us + adaptation_conductance_us
        equivalent_potential_mv = (
            params.leak_conductance_us * params.resting_potential_mv
            + adaptation_conductance_us * params.adaptation_reversal_mv
        ) / equivalent_conductance_us
        equivalent_resistance_mohm = 1.0 / equivalent_conductance_us
        steady_state_voltage_mv = equivalent_potential_mv + equivalent_resistance_mohm * current_na
        effective_time_constant_ms = params.capacitance_nf * equivalent_resistance_mohm

        voltage_mv[index + 1] = steady_state_voltage_mv + (
            voltage_mv[index] - steady_state_voltage_mv
        ) * np.exp(-params.dt_ms / effective_time_constant_ms)
        threshold_mv[index + 1] = params.baseline_threshold_mv + (
            threshold_mv[index] - params.baseline_threshold_mv
        ) * np.exp(-params.dt_ms / params.threshold_time_constant_ms)
        adaptation_state[index + 1] = adaptation_state[index] * np.exp(
            -params.dt_ms / params.adaptation_time_constant_ms
        )

        if voltage_mv[index + 1] >= threshold_mv[index + 1]:
            spike_train[index + 1] = 1
            spike_indices.append(index + 1)
            voltage_mv[index + 1] = params.resting_potential_mv
            threshold_mv[index + 1] = params.spike_threshold_mv
            if with_adaptation:
                adaptation_state[index + 1] += params.adaptation_increment * (
                    1.0 - adaptation_state[index + 1]
                )

    spike_times_ms = time_ms[np.asarray(spike_indices, dtype=int)]
    isi_ms = inter_spike_intervals(spike_times_ms)

    return {
        "time_ms": time_ms,
        "voltage_mv": voltage_mv,
        "threshold_mv": threshold_mv,
        "adaptation_state": adaptation_state,
        "spike_train": spike_train,
        "spike_times_ms": spike_times_ms,
        "spike_count": int(spike_times_ms.size),
        "isi_ms": isi_ms,
        "mean_rate_hz": firing_rate_hz(spike_times_ms, params.duration_ms, transient_ms=transient_ms),
        "steady_rate_hz": steady_state_rate_hz(spike_times_ms),
        "first_spike_latency_ms": first_spike_latency_ms(spike_times_ms),
        "adaptation_index": adaptation_index(isi_ms),
        "max_adaptation_state": float(np.max(adaptation_state)),
    }
