"""Reproducible single-neuron current-frequency analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_dynamics.io import FIGURES_ROOT, PROJECT_ROOT, metrics_path, write_manifest, write_rows_csv
from neural_dynamics.metrics import min_positive_current, summary_rows
from neural_dynamics.plotting import apply_style, save_figure
from neural_dynamics.single_neuron import (
    DynamicThresholdParams,
    FixedThresholdParams,
    rectified_sinusoid_current,
    simulate_dynamic_threshold_neuron,
    simulate_fixed_threshold_neuron,
)


def run_experiment() -> dict[str, object]:
    apply_style()

    fixed_params = FixedThresholdParams()
    dynamic_params = DynamicThresholdParams()
    currents_na = np.arange(0.0, 11.0, 0.5)
    output_files: list[Path] = []

    time_ms = np.arange(0.0, fixed_params.duration_ms + fixed_params.dt_ms, fixed_params.dt_ms)
    dynamic_example = simulate_dynamic_threshold_neuron(4.0, dynamic_params)
    fixed_example = simulate_fixed_threshold_neuron(
        rectified_sinusoid_current(time_ms, max_current_na=4.0),
        fixed_params,
    )

    fi_rows: list[dict[str, float | str]] = []
    for current_na in currents_na:
        fixed_result = simulate_fixed_threshold_neuron(current_na, fixed_params)
        dynamic_result = simulate_dynamic_threshold_neuron(current_na, dynamic_params)
        fi_rows.append(
            {
                "model": "fixed_threshold",
                "current_na": float(current_na),
                "spike_count": fixed_result["spike_count"],
                "mean_rate_hz": fixed_result["mean_rate_hz"],
                "steady_rate_hz": fixed_result["steady_rate_hz"],
                "first_spike_latency_ms": fixed_result["first_spike_latency_ms"],
            }
        )
        fi_rows.append(
            {
                "model": "dynamic_threshold",
                "current_na": float(current_na),
                "spike_count": dynamic_result["spike_count"],
                "mean_rate_hz": dynamic_result["mean_rate_hz"],
                "steady_rate_hz": dynamic_result["steady_rate_hz"],
                "first_spike_latency_ms": dynamic_result["first_spike_latency_ms"],
            }
        )

    fig_dir = FIGURES_ROOT / "single_neuron"

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(fixed_example["time_ms"], fixed_example["input_current_na"], color="black")
    axes[0].set_ylabel("Current (nA)")
    axes[0].set_title("Rectified sinusoidal drive")
    axes[1].plot(fixed_example["time_ms"], fixed_example["voltage_mv"], color="black", label="Fixed threshold")
    axes[1].plot(
        dynamic_example["time_ms"],
        dynamic_example["threshold_mv"],
        color="tab:red",
        linestyle="--",
        label="Dynamic threshold",
    )
    axes[1].plot(dynamic_example["time_ms"], dynamic_example["voltage_mv"], color="tab:blue", alpha=0.75, label="Dynamic voltage")
    axes[1].axhline(fixed_params.threshold_mv, color="gray", linestyle=":", label="Fixed threshold")
    axes[1].set_ylabel("Voltage (mV)")
    axes[1].legend(ncol=2)
    axes[2].eventplot(
        [fixed_example["spike_times_ms"], dynamic_example["spike_times_ms"]],
        colors=["black", "tab:blue"],
        lineoffsets=[1, 0],
        linelengths=0.8,
    )
    axes[2].set_yticks([1, 0], ["Fixed", "Dynamic"])
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("Spike timing comparison")
    fig.tight_layout()
    dynamics_path = fig_dir / "single_neuron_dynamics.png"
    save_figure(fig, dynamics_path)
    output_files.append(dynamics_path)

    fixed_curve = [row for row in fi_rows if row["model"] == "fixed_threshold"]
    dynamic_curve = [row for row in fi_rows if row["model"] == "dynamic_threshold"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [row["current_na"] for row in fixed_curve],
        [row["steady_rate_hz"] for row in fixed_curve],
        "o-",
        color="black",
        label="Fixed threshold",
    )
    ax.plot(
        [row["current_na"] for row in dynamic_curve],
        [row["steady_rate_hz"] for row in dynamic_curve],
        "s-",
        color="tab:blue",
        label="Dynamic threshold",
    )
    ax.set_xlabel("Input current (nA)")
    ax.set_ylabel("Steady-state firing rate (Hz)")
    ax.set_title("Single-neuron current-frequency curves")
    ax.legend()
    fig.tight_layout()
    fi_path = fig_dir / "single_neuron_fi_curve.png"
    save_figure(fig, fi_path)
    output_files.append(fi_path)

    summary = {
        "fixed_rheobase_current_na": min_positive_current(fixed_curve, "mean_rate_hz", "current_na"),
        "dynamic_rheobase_current_na": min_positive_current(dynamic_curve, "mean_rate_hz", "current_na"),
        "fixed_steady_rate_at_4na_hz": next(
            row["steady_rate_hz"] for row in fixed_curve if row["current_na"] == 4.0
        ),
        "dynamic_steady_rate_at_4na_hz": next(
            row["steady_rate_hz"] for row in dynamic_curve if row["current_na"] == 4.0
        ),
    }
    summary["steady_rate_difference_at_4na_hz"] = (
        summary["fixed_steady_rate_at_4na_hz"] - summary["dynamic_steady_rate_at_4na_hz"]
    )

    fi_csv = metrics_path("single_neuron_fi_curve.csv")
    write_rows_csv(fi_csv, fi_rows)
    summary_csv = metrics_path("single_neuron_summary.csv")
    write_rows_csv(summary_csv, summary_rows(summary, section="single_neuron"))
    output_files.extend([fi_csv, summary_csv])
    write_manifest("single_neuron", output_files)

    return {
        "name": "single_neuron",
        "summary": summary,
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in output_files],
    }


if __name__ == "__main__":
    run_experiment()
