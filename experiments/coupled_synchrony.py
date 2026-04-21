"""Reproducible synchrony analysis for coupled neurons."""

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

from neural_dynamics.coupled_neurons import CoupledNeuronParams, simulate_coupled_neurons
from neural_dynamics.io import FIGURES_ROOT, PROJECT_ROOT, metrics_path, write_manifest, write_rows_csv
from neural_dynamics.metrics import summary_rows
from neural_dynamics.plotting import apply_style, save_figure


def run_experiment() -> dict[str, object]:
    apply_style()

    base_params = CoupledNeuronParams()
    base_result = simulate_coupled_neurons(base_params)
    strength_ratios = np.arange(0.5, 3.25, 0.5)
    sweep_rows: list[dict[str, float]] = []
    output_files: list[Path] = []

    for ratio in strength_ratios:
        params = replace(base_params, synapse_12_strength_ratio=float(ratio))
        result = simulate_coupled_neurons(params)
        sweep_rows.append(
            {
                "synapse_12_strength_ratio": float(ratio),
                "rate_1_hz": result["rate_1_hz"],
                "rate_2_hz": result["rate_2_hz"],
                "steady_rate_1_hz": result["steady_rate_1_hz"],
                "steady_rate_2_hz": result["steady_rate_2_hz"],
                "coincidence_fraction": result["coincidence_fraction"],
                "mean_nearest_lag_ms": result["mean_nearest_lag_ms"],
                "mean_signed_lag_ms": result["mean_signed_lag_ms"],
                "membrane_correlation": result["membrane_correlation"],
            }
        )

    fig_dir = FIGURES_ROOT / "coupled"

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(base_result["time_ms"], base_result["voltage_1_mv"], color="black", label="Neuron 1")
    axes[0].plot(base_result["time_ms"], base_result["threshold_1_mv"], color="tab:red", linestyle="--", label="Threshold 1")
    axes[0].plot(base_result["time_ms"], base_result["voltage_2_mv"], color="tab:blue", alpha=0.8, label="Neuron 2")
    axes[0].plot(base_result["time_ms"], base_result["threshold_2_mv"], color="tab:orange", linestyle="--", label="Threshold 2")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title("Coupled-neuron membrane dynamics")
    axes[0].legend(ncol=2)
    axes[1].eventplot(
        [base_result["spike_times_1_ms"], base_result["spike_times_2_ms"]],
        colors=["black", "tab:blue"],
        lineoffsets=[1, 0],
        linelengths=0.8,
    )
    axes[1].set_yticks([1, 0], ["Neuron 1", "Neuron 2"])
    axes[1].set_title("Spike raster")
    axes[2].plot(base_result["time_ms"], base_result["synapse_12_state"], color="tab:purple", label="1 -> 2")
    axes[2].plot(base_result["time_ms"], base_result["synapse_21_state"], color="tab:green", label="2 -> 1")
    axes[2].set_ylabel("Synaptic state")
    axes[2].set_title("Synaptic opening variables")
    axes[2].legend()
    axes[3].hist(base_result["lag_distribution_ms"], bins=20, color="gray", edgecolor="white")
    axes[3].set_xlabel("Signed nearest-spike lag (ms)")
    axes[3].set_ylabel("Count")
    axes[3].set_title("Leader-lag distribution")
    fig.tight_layout()
    dynamics_path = fig_dir / "coupled_neuron_dynamics.png"
    save_figure(fig, dynamics_path)
    output_files.append(dynamics_path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(
        [row["synapse_12_strength_ratio"] for row in sweep_rows],
        [row["coincidence_fraction"] for row in sweep_rows],
        "o-",
        color="tab:blue",
    )
    axes[0].set_ylabel("Coincidence fraction")
    axes[0].set_title("Synchrony sensitivity to excitatory strength")
    axes[1].plot(
        [row["synapse_12_strength_ratio"] for row in sweep_rows],
        [row["mean_signed_lag_ms"] for row in sweep_rows],
        "s-",
        color="tab:orange",
        label="Signed lag",
    )
    axes[1].plot(
        [row["synapse_12_strength_ratio"] for row in sweep_rows],
        [row["membrane_correlation"] for row in sweep_rows],
        "^-",
        color="black",
        label="Voltage correlation",
    )
    axes[1].set_xlabel("Excitatory synapse strength ratio")
    axes[1].set_ylabel("Metric value")
    axes[1].legend()
    fig.tight_layout()
    sweep_path = fig_dir / "coupled_synchrony_sweep.png"
    save_figure(fig, sweep_path)
    output_files.append(sweep_path)

    best_sync = max(sweep_rows, key=lambda row: row["coincidence_fraction"])
    summary = {
        "base_rate_1_hz": base_result["rate_1_hz"],
        "base_rate_2_hz": base_result["rate_2_hz"],
        "base_coincidence_fraction": base_result["coincidence_fraction"],
        "base_mean_signed_lag_ms": base_result["mean_signed_lag_ms"],
        "base_membrane_correlation": base_result["membrane_correlation"],
        "max_coincidence_fraction_in_sweep": best_sync["coincidence_fraction"],
        "strength_ratio_at_max_coincidence": best_sync["synapse_12_strength_ratio"],
    }
    summary["coincidence_fraction_gain_ratio"] = (
        summary["max_coincidence_fraction_in_sweep"] / summary["base_coincidence_fraction"]
        if summary["base_coincidence_fraction"] > 0.0
        else float("nan")
    )

    sweep_csv = metrics_path("coupled_synchrony_metrics.csv")
    write_rows_csv(sweep_csv, sweep_rows)
    summary_csv = metrics_path("coupled_synchrony_summary.csv")
    write_rows_csv(summary_csv, summary_rows(summary, section="coupled"))
    output_files.extend([sweep_csv, summary_csv])
    write_manifest("coupled", output_files)

    return {
        "name": "coupled",
        "summary": summary,
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in output_files],
    }


if __name__ == "__main__":
    run_experiment()
