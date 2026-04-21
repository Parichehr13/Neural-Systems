"""Population-level Jansen-Rit neural mass model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import dominant_frequency, welch_psd


@dataclass(frozen=True)
class NeuralMassParams:
    coupling_ep: float = 135.0
    coupling_pe_ratio: float = 0.8
    coupling_ip_ratio: float = 0.25
    coupling_pi_ratio: float = 0.25
    excitatory_gain_mv: float = 3.25
    inhibitory_gain_mv: float = 22.0
    excitatory_rate_hz: float = 100.0
    inhibitory_rate_hz: float = 50.0
    sigmoid_slope: float = 0.56
    sigmoid_midpoint_mv: float = 6.0
    max_firing_rate_s: float = 5.0
    dt_s: float = 1e-4
    duration_s: float = 20.0
    transient_s: float = 2.0
    noise_mean: float = 160.0
    noise_std: float = 200.0
    random_seed: int = 42
    psd_segment_length_s: float = 4.0
    psd_overlap_s: float = 2.0
    min_peak_frequency_hz: float = 3.0

    @property
    def coupling_pe(self) -> float:
        return self.coupling_pe_ratio * self.coupling_ep

    @property
    def coupling_ip(self) -> float:
        return self.coupling_ip_ratio * self.coupling_ep

    @property
    def coupling_pi(self) -> float:
        return self.coupling_pi_ratio * self.coupling_ep


def _sigmoid(voltage_mv: np.ndarray | float, params: NeuralMassParams) -> np.ndarray | float:
    return params.max_firing_rate_s / (
        1.0 + np.exp(-params.sigmoid_slope * (voltage_mv - params.sigmoid_midpoint_mv))
    )


def simulate_neural_mass(params: NeuralMassParams) -> dict[str, np.ndarray | float]:
    """Simulate a Jansen-Rit neural mass model and compute spectral metrics."""
    time_s = np.arange(0.0, params.duration_s + params.dt_s, params.dt_s)
    n_samples = time_s.size
    rng = np.random.default_rng(params.random_seed)
    noise = rng.normal(loc=params.noise_mean, scale=params.noise_std, size=n_samples - 1)

    pyramidal_y = np.zeros(n_samples)
    pyramidal_z = np.zeros(n_samples)
    excitatory_y = np.zeros(n_samples)
    excitatory_z = np.zeros(n_samples)
    inhibitory_y = np.zeros(n_samples)
    inhibitory_z = np.zeros(n_samples)

    for index in range(n_samples - 1):
        pyramidal_input = params.coupling_pe * excitatory_y[index] - params.coupling_pi * inhibitory_y[index]
        excitatory_input = params.coupling_ep * pyramidal_y[index]
        inhibitory_input = params.coupling_ip * pyramidal_y[index]

        pyramidal_rate = _sigmoid(pyramidal_input, params)
        excitatory_rate = _sigmoid(excitatory_input, params)
        inhibitory_rate = _sigmoid(inhibitory_input, params)

        dpyramidal_y = pyramidal_z[index]
        dpyramidal_z = (
            params.excitatory_gain_mv * params.excitatory_rate_hz * pyramidal_rate
            - 2.0 * params.excitatory_rate_hz * pyramidal_z[index]
            - (params.excitatory_rate_hz ** 2) * pyramidal_y[index]
        )

        dexcitatory_y = excitatory_z[index]
        dexcitatory_z = (
            params.excitatory_gain_mv
            * params.excitatory_rate_hz
            * (excitatory_rate + noise[index] / params.coupling_ep)
            - 2.0 * params.excitatory_rate_hz * excitatory_z[index]
            - (params.excitatory_rate_hz ** 2) * excitatory_y[index]
        )

        dinhibitory_y = inhibitory_z[index]
        dinhibitory_z = (
            params.inhibitory_gain_mv * params.inhibitory_rate_hz * inhibitory_rate
            - 2.0 * params.inhibitory_rate_hz * inhibitory_z[index]
            - (params.inhibitory_rate_hz ** 2) * inhibitory_y[index]
        )

        pyramidal_y[index + 1] = pyramidal_y[index] + dpyramidal_y * params.dt_s
        pyramidal_z[index + 1] = pyramidal_z[index] + dpyramidal_z * params.dt_s
        excitatory_y[index + 1] = excitatory_y[index] + dexcitatory_y * params.dt_s
        excitatory_z[index + 1] = excitatory_z[index] + dexcitatory_z * params.dt_s
        inhibitory_y[index + 1] = inhibitory_y[index] + dinhibitory_y * params.dt_s
        inhibitory_z[index + 1] = inhibitory_z[index] + dinhibitory_z * params.dt_s

    eeg_signal = params.coupling_pe * excitatory_y - params.coupling_pi * inhibitory_y
    transient_index = int(params.transient_s / params.dt_s)
    steady_time_s = time_s[transient_index:]
    steady_signal = eeg_signal[transient_index:]
    centered_signal = steady_signal - np.mean(steady_signal)

    fs_hz = 1.0 / params.dt_s
    frequencies_hz, power = welch_psd(
        centered_signal,
        fs_hz=fs_hz,
        segment_length_s=params.psd_segment_length_s,
        overlap_s=params.psd_overlap_s,
    )
    peak_frequency_hz, peak_power = dominant_frequency(
        frequencies_hz,
        power,
        min_frequency_hz=params.min_peak_frequency_hz,
    )

    return {
        "time_s": time_s,
        "eeg_signal": eeg_signal,
        "steady_time_s": steady_time_s,
        "steady_signal": steady_signal,
        "steady_signal_centered": centered_signal,
        "frequencies_hz": frequencies_hz,
        "power": power,
        "peak_frequency_hz": peak_frequency_hz,
        "peak_power": peak_power,
        "rms_amplitude": float(np.sqrt(np.mean(centered_signal ** 2))),
    }
