# Integrate-and-Fire Model Report

## Overview
This module implements two related single-neuron spiking models:

- Stage 1: leaky integrate-and-fire with fixed threshold.
- Stage 2: leaky integrate-and-fire with dynamic threshold (relative refractoriness).

The report focuses on membrane dynamics, spike timing, and current-frequency behavior for both stages.

## Objective
The goal is to analyze how threshold dynamics change neuron excitability and spike timing.

The study includes:

- time-domain response under variable/constant current,
- spike-train generation,
- current-frequency (f-I) curves for fixed-threshold and dynamic-threshold neurons.

## Model Equations
Membrane equation:

`tau_m * dV/dt = -(V - E0) + r * I`

Equivalent steady-state voltage:

`Vinf = E0 + r * I`

Stage 1 rule (fixed threshold):

- if `V >= Vth`, register spike and reset `V <- E0`.

Stage 2 threshold dynamics:

`tau_t * dVt/dt = -(Vt - VtL)`

Spike/reset in Stage 2:

- if `V >= Vt`, reset `V <- E0` and raise threshold `Vt <- VtH`.

## Parameters
- `E0 = -65 mV`
- `r = 10 MOhm`
- `tau_m = 30 ms`
- Stage 1 threshold: `Vth = -50 mV`
- Stage 2 thresholds: `VtL = -55 mV`, `VtH = 0 mV`
- Stage 2 threshold time constant: `tau_t = 10 ms`
- `dt = 0.05 ms`
- `tend = 300 ms`

## Results

### Figure 1 - Stage 1 Dynamics (Input, Voltage, Spikes)
This figure shows rectified-sinusoid input current, membrane response, and output spikes for the fixed-threshold model.

![Stage 1 Dynamics](figures/integrate_and_fire_fig_001_stage1_dynamics.png)

### Figure 2 - Stage 1 f-I Curve
Firing rate increases with current for the fixed-threshold model.

![Stage 1 f-I Curve](figures/integrate_and_fire_fig_002_stage1_fi_curve.png)

### Figure 3 - Stage 2 Dynamics (Voltage, Dynamic Threshold, Spikes)
The threshold is elevated after each spike and relaxes back, producing relative refractoriness.

![Stage 2 Dynamics](figures/integrate_and_fire_fig_003_stage2_dynamics.png)

### Figure 4 - Stage 2 f-I Curve
The dynamic threshold keeps the global monotonic f-I trend while modulating timing and effective excitability.

![Stage 2 f-I Curve](figures/integrate_and_fire_fig_004_stage2_fi_curve.png)

## Interpretation
- Stage 1 gives the baseline integrate-and-fire response with fixed spike condition.
- Stage 2 introduces temporal self-regulation through dynamic threshold elevation.
- Dynamic threshold reduces immediate re-firing after spikes and produces more realistic spike timing patterns.
- Both stages preserve increasing firing rate with increasing input current.

## Conclusion
The model progression from fixed threshold to dynamic threshold demonstrates how a small state extension improves temporal realism without sacrificing interpretability.

## Reproducibility
Run:

```powershell
python 01_integrate_and_fire/integrate_and_fire.py
```

Generated files are stored in:

- `01_integrate_and_fire/figures/integrate_and_fire_fig_001_stage1_dynamics.png`
- `01_integrate_and_fire/figures/integrate_and_fire_fig_002_stage1_fi_curve.png`
- `01_integrate_and_fire/figures/integrate_and_fire_fig_003_stage2_dynamics.png`
- `01_integrate_and_fire/figures/integrate_and_fire_fig_004_stage2_fi_curve.png`
- `01_integrate_and_fire/figures/integrate_and_fire_manifest.json`
