# Exercise 03 Report

## Objective
Integrate-and-fire neuron with:
- variable threshold (relative refractory effect),
- adaptation conductance.

## Model Used in Code
- `ga = gamax * Pa[k]`
- `geq = g + ga`
- `E0tot = (g*E0 + ga*Ea) / geq`
- `rtot = 1 / geq`
- `Vinf = E0tot + rtot*I`
- `tau = C * rtot`
- `V[k+1] = (V[k] - Vinf) * exp(-dt/tau) + Vinf`
- `Vt[k+1] = (Vt[k] - Vtl) * exp(-dt/taut) + Vtl`
- `Pa[k+1] = Pa[k] * exp(-dt/taua)`
- If spike (`V[k+1] > Vt[k+1]`): reset `V`, raise `Vt`, update `Pa`.

## Results
Dynamics with adaptation (V, Vt, Pa, spikes):

![Exercise 03 - Dynamics](figures/exercise03_fig_001.png)

Current-frequency relationship:

![Exercise 03 - I-F](figures/exercise03_fig_002.png)

## Notes
- Figures generated automatically by running `exercise03.py` headlessly.
