"""Reproducible adaptation analysis for the adaptive neuron model."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_dynamics.adaptive_neuron import AdaptiveNeuronParams, simulate_adaptive_neuron
from neural_dynamics.io import FIGURES_ROOT, PROJECT_ROOT, metrics_path, write_manifest, write_rows_csv
from neural_dynamics.metrics import summary_rows
from neural_dynamics.plotting import apply_style, save_figure


def run_experiment() -> dict[str, object]:
    apply_style()

    params = AdaptiveNeuronParams()
    currents_na = np.arange(0.0, 10.5, 0.5)
    example_current_na = 4.0
    output_files: list[Path] = []

    no_adaptation = simulate_adaptive_neuron(example_current_na, params, with_adaptation=False)
    with_adaptation = simulate_adaptive_neuron(example_current_na, params, with_adaptation=True)

    fi_rows: list[dict[str, float | str]] = []
    for current_na in currents_na:
        for label, adaptation_flag in [("without_adaptation", False), ("with_adaptation", True)]:
            result = simulate_adaptive_neuron(current_na, params, with_adaptation=adaptation_flag)
            fi_rows.append(
                {
                    "condition": label,
                    "current_na": float(current_na),
                    "spike_count": result["spike_count"],
                    "mean_rate_hz": result["mean_rate_hz"],
                    "steady_rate_hz": result["steady_rate_hz"],
                    "adaptation_index": result["adaptation_index"],
                    "max_adaptation_state": result["max_adaptation_state"],
                }
            )

    fig_dir = FIGURES_ROOT / "adaptation"

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(no_adaptation["time_ms"], no_adaptation["voltage_mv"], color="black", label="Without adaptation")
    axes[0].plot(with_adaptation["time_ms"], with_adaptation["voltage_mv"], color="tab:blue", label="With adaptation")
    axes[0].plot(with_adaptation["time_ms"], with_adaptation["threshold_mv"], color="tab:red", linestyle="--", label="Threshold")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title(f"Adaptive neuron response at {example_current_na:.1f} nA")
    axes[0].legend(ncol=2)
    axes[1].plot(with_adaptation["time_ms"], with_adaptation["adaptation_state"], color="tab:green")
    axes[1].set_ylabel("Adaptation state")
    axes[1].set_title("Spike-triggered adaptation build-up")
    axes[2].eventplot(
        [no_adaptation["spike_times_ms"], with_adaptation["spike_times_ms"]],
        colors=["black", "tab:blue"],
        lineoffsets=[1, 0],
        linelengths=0.8,
    )
    axes[2].set_yticks([1, 0], ["No adapt", "Adapt"])
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("Spike train comparison")
    fig.tight_layout()
    dynamics_path = fig_dir / "adaptation_dynamics.png"
    save_figure(fig, dynamics_path)
    output_files.append(dynamics_path)

    no_adaptation_curve = [row for row in fi_rows if row["condition"] == "without_adaptation"]
    with_adaptation_curve = [row for row in fi_rows if row["condition"] == "with_adaptation"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [row["current_na"] for row in no_adaptation_curve],
        [row["steady_rate_hz"] for row in no_adaptation_curve],
        "o-",
        color="black",
        label="Without adaptation",
    )
    ax.plot(
        [row["current_na"] for row in with_adaptation_curve],
        [row["steady_rate_hz"] for row in with_adaptation_curve],
        "s-",
        color="tab:blue",
        label="With adaptation",
    )
    ax.set_xlabel("Input current (nA)")
    ax.set_ylabel("Steady-state firing rate (Hz)")
    ax.set_title("Adaptation shifts the f-I curve")
    ax.legend()
    fig.tight_layout()
    fi_path = fig_dir / "adaptation_fi_curve.png"
    save_figure(fig, fi_path)
    output_files.append(fi_path)

    summary = {
        "steady_rate_without_adaptation_at_4na_hz": no_adaptation["steady_rate_hz"],
        "steady_rate_with_adaptation_at_4na_hz": with_adaptation["steady_rate_hz"],
        "steady_rate_reduction_at_4na_hz": no_adaptation["steady_rate_hz"] - with_adaptation["steady_rate_hz"],
        "adaptation_index_at_4na": with_adaptation["adaptation_index"],
        "max_adaptation_state_at_4na": with_adaptation["max_adaptation_state"],
    }
    summary["steady_rate_reduction_at_4na_percent"] = (
        100.0
        * summary["steady_rate_reduction_at_4na_hz"]
        / summary["steady_rate_without_adaptation_at_4na_hz"]
    )

    fi_csv = metrics_path("adaptation_fi_curve.csv")
    write_rows_csv(fi_csv, fi_rows)
    summary_csv = metrics_path("adaptation_summary.csv")
    write_rows_csv(summary_csv, summary_rows(summary, section="adaptation"))
    output_files.extend([fi_csv, summary_csv])
    write_manifest("adaptation", output_files)

    return {
        "name": "adaptation",
        "summary": summary,
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in output_files],
    }


if __name__ == "__main__":
    run_experiment()
