"""Shared quantitative metrics used across simulations."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def build_time_vector(duration: float, dt: float) -> np.ndarray:
    """Return an inclusive time vector."""
    return np.arange(0.0, duration + dt, dt)


def inter_spike_intervals(spike_times_ms: np.ndarray) -> np.ndarray:
    """Return inter-spike intervals in milliseconds."""
    return np.diff(spike_times_ms) if spike_times_ms.size >= 2 else np.array([], dtype=float)


def firing_rate_hz(
    spike_times_ms: np.ndarray,
    end_time_ms: float,
    transient_ms: float = 0.0,
) -> float:
    """Compute the mean firing rate over a specified time window."""
    valid = spike_times_ms[spike_times_ms >= transient_ms]
    duration_s = max((end_time_ms - transient_ms) / 1000.0, 0.0)
    if duration_s <= 0.0:
        return math.nan
    return float(valid.size / duration_s)


def steady_state_rate_hz(spike_times_ms: np.ndarray, n_last_isi: int = 3) -> float:
    """Estimate steady-state rate from the last few inter-spike intervals."""
    isi_ms = inter_spike_intervals(spike_times_ms)
    if isi_ms.size == 0:
        return 0.0
    window = isi_ms[-n_last_isi:] if isi_ms.size >= n_last_isi else isi_ms
    return float(1000.0 / np.mean(window))


def first_spike_latency_ms(spike_times_ms: np.ndarray, stimulus_start_ms: float = 0.0) -> float:
    """Return the first spike latency or NaN when no spikes occur."""
    if spike_times_ms.size == 0:
        return math.nan
    return float(spike_times_ms[0] - stimulus_start_ms)


def adaptation_index(isi_ms: np.ndarray) -> float:
    """Quantify spike-frequency adaptation from the first and last ISI."""
    if isi_ms.size < 2 or isi_ms[0] == 0.0:
        return math.nan
    return float((isi_ms[-1] - isi_ms[0]) / isi_ms[0])


def _nearest_lags(reference_spikes_ms: np.ndarray, target_spikes_ms: np.ndarray) -> np.ndarray:
    """Return signed lags to the nearest target spike for each reference spike."""
    if reference_spikes_ms.size == 0 or target_spikes_ms.size == 0:
        return np.array([], dtype=float)

    lags = []
    for spike_time in reference_spikes_ms:
        delta = target_spikes_ms - spike_time
        lags.append(delta[np.argmin(np.abs(delta))])
    return np.asarray(lags, dtype=float)


def coincidence_fraction(
    spike_times_1_ms: np.ndarray,
    spike_times_2_ms: np.ndarray,
    tolerance_ms: float = 2.0,
) -> float:
    """Fraction of spikes in neuron 1 with a spike in neuron 2 within tolerance."""
    nearest_lags = _nearest_lags(spike_times_1_ms, spike_times_2_ms)
    if nearest_lags.size == 0:
        return math.nan
    return float(np.mean(np.abs(nearest_lags) <= tolerance_ms))


def mean_nearest_lag_ms(spike_times_1_ms: np.ndarray, spike_times_2_ms: np.ndarray) -> float:
    """Mean absolute nearest-spike lag."""
    nearest_lags = _nearest_lags(spike_times_1_ms, spike_times_2_ms)
    if nearest_lags.size == 0:
        return math.nan
    return float(np.mean(np.abs(nearest_lags)))


def mean_signed_lag_ms(spike_times_1_ms: np.ndarray, spike_times_2_ms: np.ndarray) -> float:
    """Mean signed nearest-spike lag. Negative means neuron 2 tends to lead."""
    nearest_lags = _nearest_lags(spike_times_1_ms, spike_times_2_ms)
    if nearest_lags.size == 0:
        return math.nan
    return float(np.mean(nearest_lags))


def lag_distribution_ms(spike_times_1_ms: np.ndarray, spike_times_2_ms: np.ndarray) -> np.ndarray:
    """Return signed nearest-spike lags for plotting."""
    return _nearest_lags(spike_times_1_ms, spike_times_2_ms)


def membrane_correlation(trace_a: np.ndarray, trace_b: np.ndarray, start_index: int = 0) -> float:
    """Pearson correlation between two voltage traces after a transient."""
    if trace_a.size == 0 or trace_b.size == 0:
        return math.nan
    a = trace_a[start_index:]
    b = trace_b[start_index:]
    if a.size < 2 or b.size < 2:
        return math.nan
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a == 0.0 or std_b == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def dominant_frequency(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    min_frequency_hz: float = 0.0,
) -> tuple[float, float]:
    """Return dominant frequency and power above a minimum frequency."""
    mask = frequencies_hz >= min_frequency_hz
    if not np.any(mask):
        return math.nan, math.nan
    sub_freq = frequencies_hz[mask]
    sub_power = power[mask]
    peak_index = int(np.argmax(sub_power))
    return float(sub_freq[peak_index]), float(sub_power[peak_index])


def welch_psd(
    signal: np.ndarray,
    fs_hz: float,
    segment_length_s: float,
    overlap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the PSD using a NumPy implementation of Welch's method."""
    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")
    nperseg = max(int(segment_length_s * fs_hz), 8)
    noverlap = max(int(overlap_s * fs_hz), 0)
    if noverlap >= nperseg:
        raise ValueError("Overlap must be smaller than segment length.")
    step = nperseg - noverlap
    if signal.size < nperseg:
        nperseg = signal.size
        step = nperseg
    if nperseg < 8:
        raise ValueError("Signal is too short for PSD estimation.")

    window = np.hanning(nperseg)
    window_power = np.sum(window ** 2)
    segments = []
    for start in range(0, signal.size - nperseg + 1, step):
        segment = signal[start : start + nperseg]
        segment = segment - np.mean(segment)
        spectrum = np.fft.rfft(segment * window)
        periodogram = (np.abs(spectrum) ** 2) / (fs_hz * window_power)
        if periodogram.size > 2:
            periodogram[1:-1] *= 2.0
        segments.append(periodogram)

    power = np.mean(np.vstack(segments), axis=0)
    frequencies_hz = np.fft.rfftfreq(nperseg, d=1.0 / fs_hz)
    return frequencies_hz, power


def summary_rows(metric_map: dict[str, float | int | str], section: str) -> list[dict[str, str | float | int]]:
    """Convert a metric dictionary into CSV-friendly summary rows."""
    return [
        {"section": section, "metric": metric, "value": value}
        for metric, value in metric_map.items()
    ]


def min_positive_current(rows: Iterable[dict[str, float]], rate_key: str, current_key: str) -> float:
    """Return the smallest current with a strictly positive rate."""
    candidates = [row[current_key] for row in rows if float(row[rate_key]) > 0.0]
    return float(min(candidates)) if candidates else math.nan
