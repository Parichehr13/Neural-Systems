from __future__ import annotations

from dataclasses import replace

from neural_dynamics.adaptive_neuron import AdaptiveNeuronParams, simulate_adaptive_neuron
from neural_dynamics.coupled_neurons import CoupledNeuronParams, simulate_coupled_neurons
from neural_dynamics.neural_mass import NeuralMassParams, simulate_neural_mass
from neural_dynamics.single_neuron import (
    DynamicThresholdParams,
    FixedThresholdParams,
    simulate_dynamic_threshold_neuron,
    simulate_fixed_threshold_neuron,
)


def test_fixed_threshold_neuron_does_not_spike_without_current() -> None:
    result = simulate_fixed_threshold_neuron(0.0, FixedThresholdParams())
    assert result["spike_count"] == 0


def test_dynamic_threshold_neuron_spikes_for_suprathreshold_drive() -> None:
    result = simulate_dynamic_threshold_neuron(4.0, DynamicThresholdParams())
    assert result["spike_count"] > 0


def test_adaptation_reduces_steady_state_rate() -> None:
    params = AdaptiveNeuronParams()
    without_adaptation = simulate_adaptive_neuron(4.0, params, with_adaptation=False)
    with_adaptation = simulate_adaptive_neuron(4.0, params, with_adaptation=True)
    assert with_adaptation["steady_rate_hz"] < without_adaptation["steady_rate_hz"]


def test_coupled_neuron_metrics_are_finite_in_base_configuration() -> None:
    result = simulate_coupled_neurons(CoupledNeuronParams())
    assert result["spike_count_1"] > 0
    assert result["spike_count_2"] > 0
    assert result["coincidence_fraction"] >= 0.0


def test_neural_mass_returns_positive_peak_power() -> None:
    params = replace(NeuralMassParams(duration_s=8.0, psd_segment_length_s=2.0, psd_overlap_s=1.0))
    result = simulate_neural_mass(params)
    assert result["peak_power"] > 0.0
