"""Reproducible neural-mass spectral analysis."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_dynamics.io import FIGURES_ROOT, PROJECT_ROOT, metrics_path, write_manifest, write_rows_csv
from neural_dynamics.metrics import summary_rows
from neural_dynamics.neural_mass import NeuralMassParams, simulate_neural_mass
from neural_dynamics.plotting import apply_style, save_figure


def run_experiment() -> dict[str, object]:
    apply_style()

    base_params = NeuralMassParams()
    base_result = simulate_neural_mass(base_params)
    coupling_values = [110.0, 122.5, 135.0, 147.5, 160.0]
    sweep_rows: list[dict[str, float]] = []
    output_files: list[Path] = []

    for coupling_ep in coupling_values:
        params = replace(base_params, coupling_ep=float(coupling_ep))
        result = simulate_neural_mass(params)
        sweep_rows.append(
            {
                "coupling_ep": float(coupling_ep),
                "peak_frequency_hz": result["peak_frequency_hz"],
                "peak_power": result["peak_power"],
                "rms_amplitude": result["rms_amplitude"],
            }
        )

    fig_dir = FIGURES_ROOT / "neural_mass"

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    window_start_s = 1.0
    window_stop_s = 3.0
    mask = (base_result["steady_time_s"] >= window_start_s) & (base_result["steady_time_s"] <= window_stop_s)
    axes[0].plot(
        base_result["steady_time_s"][mask],
        base_result["steady_signal_centered"][mask],
        color="black",
        linewidth=1.1,
    )
    axes[0].set_title("EEG-like signal after transient removal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    freq_mask = base_result["frequencies_hz"] >= base_params.min_peak_frequency_hz
    axes[1].plot(
        base_result["frequencies_hz"][freq_mask],
        base_result["power"][freq_mask],
        color="tab:blue",
        linewidth=1.3,
    )
    axes[1].axvline(
        base_result["peak_frequency_hz"],
        color="gray",
        linestyle="--",
        label=f"Peak = {base_result['peak_frequency_hz']:.2f} Hz",
    )
    axes[1].set_title("Power spectral density")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].legend()
    signal_path = fig_dir / "neural_mass_signal_psd.png"
    save_figure(fig, signal_path)
    output_files.append(signal_path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(
        [row["coupling_ep"] for row in sweep_rows],
        [row["peak_frequency_hz"] for row in sweep_rows],
        "o-",
        color="tab:blue",
    )
    axes[0].set_ylabel("Peak frequency (Hz)")
    axes[0].set_title("Oscillation sensitivity to excitatory coupling")
    axes[1].plot(
        [row["coupling_ep"] for row in sweep_rows],
        [row["peak_power"] for row in sweep_rows],
        "s-",
        color="black",
        label="Peak PSD",
    )
    axes[1].plot(
        [row["coupling_ep"] for row in sweep_rows],
        [row["rms_amplitude"] for row in sweep_rows],
        "^-",
        color="tab:orange",
        label="RMS amplitude",
    )
    axes[1].set_xlabel("Excitatory coupling Wep")
    axes[1].set_ylabel("Metric value")
    axes[1].legend()
    sweep_path = fig_dir / "neural_mass_parameter_sweep.png"
    save_figure(fig, sweep_path)
    output_files.append(sweep_path)

    strongest_peak = max(sweep_rows, key=lambda row: row["peak_power"])
    summary = {
        "base_peak_frequency_hz": base_result["peak_frequency_hz"],
        "base_peak_power": base_result["peak_power"],
        "base_rms_amplitude": base_result["rms_amplitude"],
        "max_peak_power_in_sweep": strongest_peak["peak_power"],
        "coupling_ep_at_max_peak_power": strongest_peak["coupling_ep"],
        "peak_frequency_at_max_peak_power_hz": strongest_peak["peak_frequency_hz"],
    }

    sweep_csv = metrics_path("neural_mass_spectral_metrics.csv")
    write_rows_csv(sweep_csv, sweep_rows)
    summary_csv = metrics_path("neural_mass_summary.csv")
    write_rows_csv(summary_csv, summary_rows(summary, section="neural_mass"))
    output_files.extend([sweep_csv, summary_csv])
    write_manifest("neural_mass", output_files)

    return {
        "name": "neural_mass",
        "summary": summary,
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in output_files],
    }


if __name__ == "__main__":
    run_experiment()
