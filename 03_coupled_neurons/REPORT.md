# Coupled Neurons Report

## Overview

This module studies two coupled integrate-and-fire neurons connected through conductance-based synapses. Each neuron receives a constant input current and interacts with the other neuron through one synapse. The purpose of the simulation is to understand how synaptic coupling affects firing rate, spike timing, and synchrony.

The model includes a dynamic threshold to represent refractoriness and does not include adaptation. This follows the coupled integrate-and-fire specification with synaptic conductances and variable-threshold dynamics.

---

## Objective

The goal of this module is to analyze the response of two interacting spiking neurons under constant stimulation.

The simulation is used to evaluate:

- membrane-potential dynamics in both neurons,
- threshold recovery after each spike,
- synaptic interaction through conductance variables,
- synchrony between spike trains.

In practice, the main question is whether the two neurons tend to fire together, remain phase-shifted, or show an asymmetric relationship due to the chosen synaptic signs.

---

## Model Description

Below threshold, each neuron behaves as a leaky RC membrane. The synaptic conductance modifies the effective membrane dynamics by changing both total conductance and effective equilibrium potential.

For each neuron, the simulation computes:

- total conductance: `geq = g + gsyn`
- effective equilibrium potential: `Eeq = (g*E0 + gsyn*Esyn) / geq`
- equivalent resistance: `req = 1 / geq`
- effective time constant: `tau = C * req`
- asymptotic membrane voltage: `Vinf = Eeq + req * I`

The membrane potential is then updated with the exact exponential step:

`V[k+1] = (V[k] - Vinf) * exp(-dt/tau) + Vinf`

This is consistent with the integrate-and-fire formulation, where membrane voltage follows first-order subthreshold dynamics and spikes are represented by reset events rather than fully modeled action potentials.

---

## Refractory Mechanism

The refractory period is modeled through a dynamic threshold.

Instead of a fixed firing threshold, the threshold is treated as a variable that:

- jumps to a high value immediately after a spike,
- decays exponentially back to a lower resting threshold.

This provides a simple representation of relative refractoriness: after firing, the neuron is temporarily harder to excite.

Threshold recovery in code:

`Vt[k+1] = (Vt[k] - Vt_low) * exp(-dt/taut) + Vt_low`

When a spike occurs, threshold is set to `Vt_high`.

---

## Synaptic Coupling

The two neurons are connected through conductance-based synapses. When one neuron fires, it increases the synaptic opening variable of the outgoing synapse. Between spikes, this synaptic variable decays exponentially.

Synaptic opening variables follow:

`P[k+1] = P[k] * exp(-dt/taus)`

and on spikes they are incremented by:

`P <- P + dP * (1 - P)`

This captures the idea that presynaptic spikes open postsynaptic channels and synaptic conductance then decays over time.

In this simulation, coupling is mixed:

- synapse `1 -> 2` is excitatory with `E_12 = 0 mV`
- synapse `2 -> 1` is inhibitory with `E_21 = -70 mV`

This makes interaction asymmetric and produces non-identical timing behavior.

---

## Parameters Used

Main simulation parameters:

- `E0 = -65 mV`
- `r = 10 MOhm`
- `taum = 30 ms`
- `taut = 5 ms`
- `Vt_low = -55 mV`
- `Vt_high = 50 mV`
- `dt = 0.01 ms`
- `tmax = 150 ms`
- `I1 = I2 = 2.5 nA`
- `tau_12 = tau_21 = 10 ms`
- `dP_12 = dP_21 = 0.20`
- `r * gsmax = 2` for both synapses

These values are consistent with typical recommendations for synaptic strength, synaptic time constant, and constant-current stimulation.

---

## Results

### Figure 1 - Membrane Potentials and Dynamic Thresholds

This figure shows membrane potentials and thresholds of both neurons over time.

It helps inspect spike timing, threshold recovery after each spike, and differences between neuron responses due to asymmetric coupling.

![Figure 1](figures/coupled_neurons_fig_001_membrane_and_thresholds.png)

### Figure 2 - Zoomed Timing Comparison

This figure overlays both membrane traces and thresholds on a shorter time window.

It makes it easier to inspect timing offsets and phase relationships between spikes.

![Figure 2](figures/coupled_neurons_fig_002_zoomed_timing_comparison.png)

### Figure 3 - Spike Raster

The spike raster gives the clearest synchrony view.

If spikes align strongly, synchrony is high. If one neuron consistently leads/lags, coupling is asymmetric in timing.

![Figure 3](figures/coupled_neurons_fig_003_spike_raster.png)

### Figure 4 - Synaptic Opening Variables

This figure shows the time evolution of synaptic state variables for both directed synapses.

These traces indicate when interaction is strongest and how long synaptic influence persists.

![Figure 4](figures/coupled_neurons_fig_004_synaptic_opening_variables.png)

### Figure 5 - Nearest Spike-Lag Histogram

This histogram summarizes synchrony quantitatively through nearest spike-lag distribution.

A distribution concentrated near zero indicates tighter synchrony; broader or shifted distributions indicate weaker synchrony or systematic lag.

![Figure 5](figures/coupled_neurons_fig_005_nearest_spike_lag_histogram.png)

---

## Interpretation

The simulation shows that conductance coupling plus dynamic threshold is sufficient to produce nontrivial timing relationships between two neurons.

Main effects:

- dynamic threshold prevents immediate re-firing and creates relative refractoriness,
- excitatory coupling tends to promote postsynaptic firing,
- inhibitory coupling tends to delay or suppress firing in the opposite direction,
- mixed coupling produces asymmetric timing instead of perfect symmetry.

Synchrony is therefore controlled not only by input current, but also by synaptic sign, conductance strength, and synaptic decay time.

---

## Conclusion

This module demonstrates that a compact conductance-based integrate-and-fire model can reproduce meaningful interactions between coupled neurons.

Even with only two neurons, the model captures:

- recurrent interaction through synaptic conductance,
- refractoriness through dynamic threshold,
- direction-dependent effects of excitation and inhibition,
- measurable synchrony and lag structure between spike trains.

Overall, this provides an effective framework for studying how synaptic coupling shapes spike timing in small neural systems.

---

## Reproducibility

Run:

```bash
python 03_coupled_neurons/coupled_neurons.py
```
