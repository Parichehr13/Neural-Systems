"""Coupled integrate-and-fire neuron model with synaptic interaction metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import (
    build_time_vector,
    coincidence_fraction,
    firing_rate_hz,
    lag_distribution_ms,
    mean_nearest_lag_ms,
    mean_signed_lag_ms,
    membrane_correlation,
    steady_state_rate_hz,
)


@dataclass(frozen=True)
class CoupledNeuronParams:
    resting_potential_mv: float = -65.0
    membrane_time_constant_ms: float = 30.0
    threshold_time_constant_ms: float = 5.0
    membrane_resistance_mohm: float = 10.0
    dt_ms: float = 0.01
    duration_ms: float = 150.0
    baseline_threshold_mv: float = -55.0
    spike_threshold_mv: float = 50.0
    input_current_1_na: float = 2.5
    input_current_2_na: float = 2.5
    synapse_12_tau_ms: float = 10.0
    synapse_21_tau_ms: float = 10.0
    synapse_12_increment: float = 0.20
    synapse_21_increment: float = 0.20
    synapse_12_reversal_mv: float = 0.0
    synapse_21_reversal_mv: float = -70.0
    synapse_12_strength_ratio: float = 2.0
    synapse_21_strength_ratio: float = 2.0

    @property
    def capacitance_nf(self) -> float:
        return self.membrane_time_constant_ms / self.membrane_resistance_mohm

    @property
    def leak_conductance_us(self) -> float:
        return 1.0 / self.membrane_resistance_mohm

    @property
    def synapse_12_conductance_us(self) -> float:
        return self.synapse_12_strength_ratio / self.membrane_resistance_mohm

    @property
    def synapse_21_conductance_us(self) -> float:
        return self.synapse_21_strength_ratio / self.membrane_resistance_mohm


def _exact_update(value: float, steady_state_value: float, tau_ms: float, dt_ms: float) -> float:
    return (value - steady_state_value) * np.exp(-dt_ms / tau_ms) + steady_state_value


def simulate_coupled_neurons(
    params: CoupledNeuronParams,
    transient_ms: float = 50.0,
) -> dict[str, np.ndarray | float | int]:
    """Simulate a pair of synaptically coupled neurons."""
    time_ms = build_time_vector(params.duration_ms, params.dt_ms)
    voltage_1_mv = np.zeros_like(time_ms)
    voltage_2_mv = np.zeros_like(time_ms)
    threshold_1_mv = np.zeros_like(time_ms)
    threshold_2_mv = np.zeros_like(time_ms)
    synapse_12_state = np.zeros_like(time_ms)
    synapse_21_state = np.zeros_like(time_ms)

    voltage_1_mv[0] = params.resting_potential_mv
    voltage_2_mv[0] = params.resting_potential_mv
    threshold_1_mv[0] = params.baseline_threshold_mv
    threshold_2_mv[0] = params.baseline_threshold_mv

    spike_indices_1: list[int] = []
    spike_indices_2: list[int] = []

    for index in range(time_ms.size - 1):
        gsyn_to_1 = params.synapse_21_conductance_us * synapse_21_state[index]
        gsyn_to_2 = params.synapse_12_conductance_us * synapse_12_state[index]

        geq_1 = params.leak_conductance_us + gsyn_to_1
        geq_2 = params.leak_conductance_us + gsyn_to_2
        req_1 = 1.0 / geq_1
        req_2 = 1.0 / geq_2

        eq_potential_1_mv = (
            params.leak_conductance_us * params.resting_potential_mv
            + gsyn_to_1 * params.synapse_21_reversal_mv
        ) / geq_1
        eq_potential_2_mv = (
            params.leak_conductance_us * params.resting_potential_mv
            + gsyn_to_2 * params.synapse_12_reversal_mv
        ) / geq_2

        tau_1_ms = params.capacitance_nf * req_1
        tau_2_ms = params.capacitance_nf * req_2
        steady_state_1_mv = eq_potential_1_mv + req_1 * params.input_current_1_na
        steady_state_2_mv = eq_potential_2_mv + req_2 * params.input_current_2_na

        threshold_1_next_mv = _exact_update(
            threshold_1_mv[index],
            params.baseline_threshold_mv,
            params.threshold_time_constant_ms,
            params.dt_ms,
        )
        threshold_2_next_mv = _exact_update(
            threshold_2_mv[index],
            params.baseline_threshold_mv,
            params.threshold_time_constant_ms,
            params.dt_ms,
        )
        synapse_12_next = synapse_12_state[index] * np.exp(-params.dt_ms / params.synapse_12_tau_ms)
        synapse_21_next = synapse_21_state[index] * np.exp(-params.dt_ms / params.synapse_21_tau_ms)
        voltage_1_next_mv = _exact_update(voltage_1_mv[index], steady_state_1_mv, tau_1_ms, params.dt_ms)
        voltage_2_next_mv = _exact_update(voltage_2_mv[index], steady_state_2_mv, tau_2_ms, params.dt_ms)

        if voltage_1_next_mv >= threshold_1_next_mv:
            voltage_1_next_mv = params.resting_potential_mv
            threshold_1_next_mv = params.spike_threshold_mv
            synapse_12_next += params.synapse_12_increment * (1.0 - synapse_12_next)
            spike_indices_1.append(index + 1)

        if voltage_2_next_mv >= threshold_2_next_mv:
            voltage_2_next_mv = params.resting_potential_mv
            threshold_2_next_mv = params.spike_threshold_mv
            synapse_21_next += params.synapse_21_increment * (1.0 - synapse_21_next)
            spike_indices_2.append(index + 1)

        voltage_1_mv[index + 1] = voltage_1_next_mv
        voltage_2_mv[index + 1] = voltage_2_next_mv
        threshold_1_mv[index + 1] = threshold_1_next_mv
        threshold_2_mv[index + 1] = threshold_2_next_mv
        synapse_12_state[index + 1] = synapse_12_next
        synapse_21_state[index + 1] = synapse_21_next

    spike_times_1_ms = time_ms[np.asarray(spike_indices_1, dtype=int)]
    spike_times_2_ms = time_ms[np.asarray(spike_indices_2, dtype=int)]
    lag_ms = lag_distribution_ms(spike_times_1_ms, spike_times_2_ms)
    start_index = int(transient_ms / params.dt_ms)

    return {
        "time_ms": time_ms,
        "voltage_1_mv": voltage_1_mv,
        "voltage_2_mv": voltage_2_mv,
        "threshold_1_mv": threshold_1_mv,
        "threshold_2_mv": threshold_2_mv,
        "synapse_12_state": synapse_12_state,
        "synapse_21_state": synapse_21_state,
        "spike_times_1_ms": spike_times_1_ms,
        "spike_times_2_ms": spike_times_2_ms,
        "spike_count_1": int(spike_times_1_ms.size),
        "spike_count_2": int(spike_times_2_ms.size),
        "rate_1_hz": firing_rate_hz(spike_times_1_ms, params.duration_ms, transient_ms=transient_ms),
        "rate_2_hz": firing_rate_hz(spike_times_2_ms, params.duration_ms, transient_ms=transient_ms),
        "steady_rate_1_hz": steady_state_rate_hz(spike_times_1_ms),
        "steady_rate_2_hz": steady_state_rate_hz(spike_times_2_ms),
        "coincidence_fraction": coincidence_fraction(spike_times_1_ms, spike_times_2_ms),
        "mean_nearest_lag_ms": mean_nearest_lag_ms(spike_times_1_ms, spike_times_2_ms),
        "mean_signed_lag_ms": mean_signed_lag_ms(spike_times_1_ms, spike_times_2_ms),
        "lag_distribution_ms": lag_ms,
        "membrane_correlation": membrane_correlation(voltage_1_mv, voltage_2_mv, start_index=start_index),
    }
