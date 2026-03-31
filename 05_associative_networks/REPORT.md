# Associative Networks Report

## Pattern Set A (Hetero-Associative Network, 10x3)

### Model Used in Code
- Training (Hebbian linear map): `W = Y * X^T`
- Linear output: `Yout = W * X`
- Sigmoid output test: `1 / (1 + exp(-k*(Yout - 0.5)))` with different `k`.

### Results
Bar-plot comparisons for three input patterns:

![Set A - Pattern 1](figures/exercise06_fig_001.png)
![Set A - Pattern 2](figures/exercise06_fig_002.png)
![Set A - Pattern 3](figures/exercise06_fig_003.png)

## Pattern Set B (Hetero-Associative Network, 16x4 with 4x4 bars)

### Model Used in Code
- Four 4x4 bar-like patterns converted to 16-element vectors.
- Training: `W = Y * X^T` with 4 outputs.
- Noisy inputs + sigmoid readout (`k=20`).
- Reconstructed image from weighted clean templates.

### Results
Pattern visualization, noisy inputs, and reconstructions:

![Set B - Inputs](figures/exercise07_fig_001.png)
![Set B - Noisy Inputs](figures/exercise07_fig_002.png)
![Set B - Recon 1](figures/exercise07_fig_003.png)
![Set B - Recon 2](figures/exercise07_fig_004.png)
![Set B - Recon 3](figures/exercise07_fig_005.png)
![Set B - Recon 4](figures/exercise07_fig_006.png)

