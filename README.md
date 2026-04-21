# Modeling Neural Dynamics from Single-Neuron Spiking to Population-Level EEG-like Activity

This repository is a reproducible computational neuroscience project that studies neural dynamics across scales using reduced models of increasing complexity. The main workflow follows a progression from single-neuron excitability, to spike-frequency adaptation, to coupled-neuron synchrony, and finally to population-level EEG-like oscillations in a neural mass model.

The project refactors four original course modules into one coherent analysis pipeline:

- single-neuron integrate-and-fire excitability
- adaptive spiking dynamics with refractory threshold and slow conductance adaptation
- synchrony and leader-lag structure in a coupled two-neuron circuit
- spectral analysis of EEG-like activity in a Jansen-Rit neural mass model

## Scientific Motivation

Reduced neural models are useful when the goal is to interpret mechanisms rather than reproduce every biophysical detail. Here, the project asks a simple cross-scale question:

How do compact dynamical models explain the transition from input-driven single-neuron spiking to coordinated population rhythms?

The answer is developed in four steps:

1. A fixed-threshold and dynamic-threshold integrate-and-fire model quantify how input current maps to firing rate.
2. An adaptive neuron model shows how slow spike-triggered conductance reduces sustained excitability.
3. A conductance-coupled two-neuron system measures synchrony, lag structure, and correlation under asymmetric excitation and inhibition.
4. A Jansen-Rit neural mass model produces EEG-like activity whose dominant frequency and spectral power can be tracked under parameter changes.

## Key Results

All values below are generated from the current code in `results/metrics/project_summary.csv`.

| Analysis | Computed result | Source |
| --- | --- | --- |
| Single-neuron excitability | Fixed-threshold rheobase: `2.0 nA`; dynamic-threshold rheobase: `1.5 nA` | `results/metrics/single_neuron_summary.csv` |
| Threshold dynamics | At `4.0 nA`, fixed-threshold steady rate: `70.67 Hz`; dynamic-threshold steady rate: `53.48 Hz` | `results/metrics/single_neuron_summary.csv` |
| Adaptation | At `4.0 nA`, adaptation lowers steady-state rate from `44.44 Hz` to `10.95 Hz` | `results/metrics/adaptation_summary.csv` |
| Adaptation strength | Adaptation index at `4.0 nA`: `2.64`; max adaptation state: `0.476` | `results/metrics/adaptation_summary.csv` |
| Coupled synchrony | Base coincidence fraction: `0.286`; strongest sweep value: `0.571` at synapse strength ratio `3.0` | `results/metrics/coupled_synchrony_summary.csv` |
| Coupled timing | Base signed nearest-spike lag: `0.481 ms`; membrane correlation: `0.032` | `results/metrics/coupled_synchrony_summary.csv` |
| Neural mass spectrum | Base dominant frequency: `3.0 Hz`; base peak power: `1.449` | `results/metrics/neural_mass_summary.csv` |
| Neural mass sensitivity | Peak power rises to `16.838` at `Wep = 160.0`, with dominant frequency `3.5 Hz` | `results/metrics/neural_mass_summary.csv` |

## Example Figures

The experiment pipeline saves all figures automatically under `figures/`.

- Single-neuron dynamics and f-I curve: [figures/single_neuron](figures/single_neuron)
- Adaptation dynamics and adapted f-I curve: [figures/adaptation](figures/adaptation)
- Coupled-neuron dynamics and synchrony sweep: [figures/coupled](figures/coupled)
- Neural-mass signal, PSD, and parameter sweep: [figures/neural_mass](figures/neural_mass)

## Repository Structure

```text
README.md
pyproject.toml
src/
  neural_dynamics/
experiments/
results/
  metrics/
  manifests/
figures/
  single_neuron/
  adaptation/
  coupled/
  neural_mass/
tests/
additional_modules/
```

## Installation

Create a virtual environment, activate it, and install the project:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## How To Run

Run the full pipeline:

```bash
python experiments/run_all.py
```

Run individual analyses:

```bash
python experiments/single_neuron_fi.py
python experiments/adaptation_analysis.py
python experiments/coupled_synchrony.py
python experiments/neural_mass_spectral_analysis.py
```

Run tests:

```bash
pytest -q
```

## Outputs

Each experiment writes:

- figures to `figures/<experiment>/`
- CSV metrics to `results/metrics/`
- file manifests to `results/manifests/`

The most useful summary file for CV writing and reporting is:

- [results/metrics/project_summary.csv](results/metrics/project_summary.csv)

## Main Methods

### 1. Single-Neuron Excitability

The project compares fixed-threshold and dynamic-threshold integrate-and-fire neurons using exact exponential voltage updates and current sweeps. The main quantitative output is the firing-rate versus current relationship.

### 2. Adaptation Analysis

The adaptive neuron adds a slow spike-triggered conductance with a hyperpolarized reversal potential. The analysis measures how adaptation changes steady-state firing rate, ISIs, and adaptation build-up under sustained current.

### 3. Coupled-Neuron Synchrony

Two neurons interact through asymmetric conductance-based synapses. The analysis measures spike coincidence, signed lag, firing-rate changes, and membrane correlation while sweeping excitatory coupling strength.

### 4. Neural-Mass Spectral Analysis

A Jansen-Rit neural mass model is simulated under reproducible noisy drive. After transient removal, a Welch-style PSD is computed to estimate the dominant oscillation frequency and spectral peak power.

## Limitations

- These are reduced models intended for interpretability, not detailed biophysical reconstruction.
- The coupled-neuron analysis uses a two-neuron motif, so conclusions about network synchrony should stay local and mechanistic.
- The neural mass regime used here produces low-frequency EEG-like rhythms in the current parameter range; it is not tuned to match a specific experimental dataset.
- Parameter sweeps are intentionally lightweight and meant to show sensitivity, not exhaustive model fitting.

## Future Work

- Add larger parameter sweeps with command-line configuration files.
- Extend the coupled-neuron study to small recurrent networks with richer synchrony measures.
- Compare neural-mass outputs across parameter regimes associated with alpha-like and beta-like activity.
- Link spike-based and neural-mass analyses more directly through population statistics or coarse-graining experiments.

## Secondary Material

Additional coursework modules are preserved under [additional_modules](additional_modules) but are not part of the main project framing.
