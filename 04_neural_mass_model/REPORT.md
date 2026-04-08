# Neural Mass Model Report

## Overview
This module simulates the Jansen-De Ridt neural mass model with three interacting populations:

- pyramidal population,
- excitatory interneuron population,
- inhibitory interneuron population.

The goal is to generate an EEG-like output and analyze both its time-domain dynamics and frequency-domain behavior.

## Model Setup
The simulation uses:

- connectivity terms `Wep`, `Wpe`, `Wip`, `Wpi`,
- excitatory/inhibitory gains `Ae`, `Ai`,
- synaptic rates `ae`, `ai`,
- sigmoidal firing response with parameters `kr`, `v0`, `rmax`,
- stochastic external drive (Gaussian white noise).

State variables are evolved with Euler integration at:

- `dt = 1e-4 s`
- `tmax = 20 s`

## Output Signal
The EEG-like signal is built as:

`eeg = Wpe * ye - Wpi * yi`

An initial transient of 2 seconds is removed, then the steady-state signal is centered before spectral analysis.

## Spectral Analysis
Power spectral density is estimated with Welch method:

- sampling frequency: `fs = 1/dt`,
- segment length: `4 s`,
- overlap: `2 s`.

Frequencies below 3 Hz are excluded for peak detection.

From the latest run:

- `Wep = 135.0`
- Peak frequency = `3.00 Hz`
- Peak power = `1.4490e+00`

## Result Figure
The updated code now exports a single professional figure with two panels:

1. EEG signal after transient (time domain),
2. PSD with dominant peak marker.

![Neural Mass Model - Signal and PSD](figures/neural_mass_model_fig_001_signal_and_psd.png)

## Interpretation
- The model produces structured oscillatory activity under noisy drive.
- The PSD reveals a dominant low-frequency component for this parameter set.
- The time and spectral views together provide a compact validation of model behavior.

## Conclusion
The neural mass simulation reproduces EEG-like dynamics and provides a clear workflow from state-space simulation to spectral characterization.

## Reproducibility
Run:

```bash
python 04_neural_mass_model/neural_mass_model.py
```

Generated outputs:

- `04_neural_mass_model/figures/neural_mass_model_fig_001_signal_and_psd.png`
- `04_neural_mass_model/figures/neural_mass_model_manifest.json`
