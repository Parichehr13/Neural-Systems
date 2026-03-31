# Adaptive Neuron Report

## Objective
This module extends the basic integrate-and-fire neuron by adding two biologically meaningful recovery mechanisms:

- a dynamic threshold `Vt` (relative refractory behavior),
- an adaptation conductance `ga` driven by a slow state variable `Pa`.

The goal is to evaluate how adaptation changes temporal spiking behavior and the current-frequency (I-F) relationship.

## Modeling Intuition
The core idea is that each spike should make the neuron temporarily harder to fire again.

- Dynamic threshold: immediately increases after a spike, then relaxes.
- Adaptation conductance: increases after spikes and pulls membrane dynamics toward a more hyperpolarized equilibrium (`Ea`), reducing excitability.

Together, these terms reduce firing rate for sustained input and produce spike-frequency adaptation.

## Governing Equations
At each time step, the model computes:

```text
ga   = gamax * Pa
geq  = g + ga
E0tot = (g*E0 + ga*Ea) / geq
rtot = 1 / geq
Vinf = E0tot + rtot*I
tau_eff = C * rtot
```

State updates:

```text
V[k+1]  = (V[k]  - Vinf) * exp(-dt/tau_eff) + Vinf
Vt[k+1] = (Vt[k] - Vtl)  * exp(-dt/taut)    + Vtl
Pa[k+1] =  Pa[k]         * exp(-dt/taua)
```

Spike/reset rule:

```text
if V[k+1] > Vt[k+1]:
    V[k+1]  = E0
    Vt[k+1] = Vth
    Pa[k+1] = Pa[k+1] + dPa*(1 - Pa[k+1])
```

## Parameter Set (Code Values)
- `E0 = -65 mV` (resting/reset potential)
- `Ea = -90 mV` (adaptation reversal potential)
- `taum = 30 ms`
- `taut = 10 ms` (threshold relaxation)
- `taua = 1000 ms` (slow adaptation decay)
- `r = 10 Mohm`, `g = 1/r`, `C = taum/r`
- `Vtl = -55 mV`, `Vth = 50 mV`
- `gamax = 2/r`
- `dPa = 0.1`
- simulation step `dt = 0.01 ms`

## Numerical Method
- Time-discrete simulation with analytical exponential step updates for stable first-order state evolution.
- Single-current simulation (`I = 4 nA`) for dynamics visualization.
- I-F sweep over `I = 0:0.5:10.5 nA`.
- Frequency estimate is computed from the last inter-spike interval when at least two spikes occur.

## Results
Combined dynamics (`V`, `Vt`, `Pa`, spikes):

![Adaptive Neuron - Dynamics](figures/exercise03_fig_001.png)

Current-frequency curve:

![Adaptive Neuron - I-F](figures/exercise03_fig_002.png)

## Interpretation
- `Pa` gradually accumulates during repetitive firing and decays slowly between spikes.
- As `Pa` rises, `ga` increases, reducing effective membrane resistance (`rtot`) and changing equilibrium (`E0tot`) toward `Ea`.
- This lowers effective excitability under sustained current, so firing rate is reduced relative to a non-adapting neuron.
- The I-F curve remains monotonic, but adaptation shifts spiking behavior toward more physiologically plausible rate control.

## Limitations
- Frequency is estimated from only the final inter-spike interval (not full-window average rate).
- No noise or synaptic input variability is modeled.
- No direct side-by-side plotted comparison with non-adaptive model in this report.

## Reproducibility
Run:

```powershell
python 02_adaptive_neuron/adaptive_neuron.py
```

Figures are stored in:

- `02_adaptive_neuron/figures/`
