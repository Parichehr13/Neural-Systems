"""Run the full neural dynamics analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptation_analysis import run_experiment as run_adaptation_analysis
from coupled_synchrony import run_experiment as run_coupled_synchrony
from neural_mass_spectral_analysis import run_experiment as run_neural_mass_analysis
from single_neuron_fi import run_experiment as run_single_neuron_analysis

from neural_dynamics.io import PROJECT_ROOT, metrics_path, write_rows_csv


def run_all() -> list[dict[str, object]]:
    experiment_outputs = [
        run_single_neuron_analysis(),
        run_adaptation_analysis(),
        run_coupled_synchrony(),
        run_neural_mass_analysis(),
    ]

    project_rows: list[dict[str, object]] = []
    for experiment in experiment_outputs:
        for metric, value in experiment["summary"].items():
            project_rows.append(
                {
                    "section": experiment["name"],
                    "metric": metric,
                    "value": value,
                }
            )
    write_rows_csv(metrics_path("project_summary.csv"), project_rows)
    return experiment_outputs


if __name__ == "__main__":
    run_all()
