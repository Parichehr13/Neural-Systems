# Exercise 05 Report

## Objective
Simulate a neural mass model (Jansen and De Rit) and analyze:
- EEG-like output in time,
- power spectral density.

## Model Used in Code
State variables for pyramidal, excitatory interneuron, inhibitory interneuron populations:
- `dyp = zp`
- `dzp = Ae*ae*rp - 2*ae*zp - ae*ae*yp`
- `dye = ze`
- `dze = Ae*ae*(re + n/Wpe) - 2*ae*ze - ae*ae*ye`
- `dyi = zi`
- `dzi = Ai*ai*ri - 2*ai*zi - ai*ai*yi`
- Sigmoid: `r = rmax / (1 + exp(-kr*(v - v0)))`
- Euler update for all states with step `dt`.

EEG proxy:
- `eeg = Wpe*ye - Wpi*yi`
- PSD with Welch method.

## Results
EEG segment after transient:

![Exercise 05 - EEG](figures/exercise05_fig_001.png)

Power spectral density:

![Exercise 05 - PSD](figures/exercise05_fig_002.png)
