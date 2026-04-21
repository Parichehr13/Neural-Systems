from __future__ import annotations

import math

import numpy as np

from neural_dynamics.metrics import (
    adaptation_index,
    coincidence_fraction,
    dominant_frequency,
    steady_state_rate_hz,
    welch_psd,
)


def test_steady_state_rate_hz_uses_last_interval_window() -> None:
    spike_times_ms = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    assert math.isclose(steady_state_rate_hz(spike_times_ms), 100.0, rel_tol=1e-6)


def test_adaptation_index_increases_with_longer_late_isi() -> None:
    isi_ms = np.array([10.0, 12.0, 15.0])
    assert adaptation_index(isi_ms) > 0.0


def test_coincidence_fraction_detects_tight_pairs() -> None:
    spike_times_1 = np.array([10.0, 20.0, 30.0])
    spike_times_2 = np.array([10.5, 19.5, 50.0])
    assert math.isclose(coincidence_fraction(spike_times_1, spike_times_2, tolerance_ms=1.0), 2.0 / 3.0)


def test_welch_psd_detects_sine_peak() -> None:
    fs_hz = 1000.0
    time_s = np.arange(0.0, 4.0, 1.0 / fs_hz)
    signal = np.sin(2.0 * np.pi * 10.0 * time_s)
    frequencies_hz, power = welch_psd(signal, fs_hz=fs_hz, segment_length_s=1.0, overlap_s=0.5)
    peak_frequency_hz, _ = dominant_frequency(frequencies_hz, power, min_frequency_hz=1.0)
    assert abs(peak_frequency_hz - 10.0) <= 1.0
