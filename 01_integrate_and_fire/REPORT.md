# Integrate-and-Fire Model Report

## Scope
This module implements and analyzes two related single-neuron spiking models based on the provided assignment specification:

- Stage 1: leaky integrate-and-fire (LIF) with fixed threshold.
- Stage 2: LIF with dynamic threshold to model relative refractoriness.

Reference document used for this module:

- `reference/integrate_and_fire_model_spec.pdf`

## Modeling Assumptions
- Single-compartment membrane model.
- No explicit synaptic conductance dynamics in this module.
- Spike event is represented by threshold crossing plus instantaneous reset.
- Time integration performed in discrete steps (`dt = 0.01 ms`).

## Governing Equations
Membrane dynamics:

```text
tau_m * dV/dt = -(V - E0) + r * I
```

Equivalent steady-state form used in code:

```text
Vinf = E0 + r * I
```

Stage 1 (fixed threshold):

```text
if V > Vt: spike, then V <- E0
```

Stage 2 (dynamic threshold):

```text
tau_t * dVt/dt = -(Vt - VtL)
if V > Vt: spike, then V <- E0 and Vt <- VtH
```

## Numerical Method
- Forward Euler update is used in part of Stage 1 for variable-current simulation.
- Closed-form step update (`exp(-dt/tau)`) is used for constant-current trial sweeps.
- Firing frequency is estimated from the last inter-spike interval when at least two spikes occur.

## Parameters
Core parameters used in simulations:

- `E0 = -65 mV`
- `r = 10 Mohm`
- `tau_m = 30 ms`
- Stage 1 threshold: `Vt = -55 mV`
- Stage 2 thresholds: `VtL = -55 mV`, `VtH = 0 mV`
- Stage 2 threshold time constant: `tau_t = 10 ms`

## Results
### Stage 1 (Fixed Threshold)
Input current (rectified sinusoid):

![Stage 1 - Input Current](figures/fig_001.png)

Membrane potential and spike reset behavior:

![Stage 1 - Membrane Potential](figures/fig_002.png)

Spike train:

![Stage 1 - Spike Train](figures/fig_003.png)

Current-frequency relationship:

![Stage 1 - I-F Curve](figures/fig_026.png)

Interpretation:
- The model reproduces canonical LIF behavior.
- Increasing constant current increases firing rate monotonically.

### Stage 2 (Dynamic Threshold / Relative Refractory Effect)
Example response with one constant current:

![Stage 2 - Voltage and Dynamic Threshold](figures/fig_027.png)

Current-frequency relationship:

![Stage 2 - I-F Curve](figures/fig_050.png)

Interpretation:
- After each spike, threshold elevation temporarily reduces excitability.
- Spike timing becomes more regulated than in the fixed-threshold case.
- The global I-F trend remains increasing, with refractory modulation of timing.

## Comparison Summary
- Stage 1 is the baseline excitability model.
- Stage 2 adds a biologically meaningful refractory mechanism with minimal added complexity.
- Together, these two stages establish a clear progression from basic LIF dynamics to enhanced temporal realism.

## Reproducibility
Run:

```powershell
python 01_integrate_and_fire/integrate_and_fire.py
```

Generated figures are saved under:

- `01_integrate_and_fire/figures/`

## Limitations and Next Step
- Frequency is currently estimated from the final inter-spike interval only.
- Potential extension: use mean rate over a stable time window for more robust rate estimates.
