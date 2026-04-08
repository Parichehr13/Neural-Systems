# Associative Networks Report

## Pattern Set A (Hetero-Associative Network, 10x3)

### Model Used in Code
- Training (Hebbian linear map): `W = Y * X^T`
- Linear output: `Yout = W * X`
- Sigmoid output test: `1 / (1 + exp(-k*(Yout - 0.5)))` with different `k`.

### Results
Bar-plot comparisons for three input patterns:

![Set A - Pattern 1](figures/associative_networks_fig_001_setA_pattern1.png)
![Set A - Pattern 2](figures/associative_networks_fig_002_setA_pattern2.png)
![Set A - Pattern 3](figures/associative_networks_fig_003_setA_pattern3.png)

## Pattern Set B (Hetero-Associative Network, 16x4 with 4x4 bars)

### Model Used in Code
- Four 4x4 bar-like patterns converted to 16-element vectors.
- Training: `W = Y * X^T` with 4 outputs.
- Noisy inputs + sigmoid readout (`k=20`).
- Reconstructed image from weighted clean templates.

### Results
Pattern visualization, noisy inputs, and reconstructions:

![Set B - Inputs](figures/associative_networks_fig_004_setB_inputs.png)
![Set B - Noisy Inputs](figures/associative_networks_fig_005_setB_noisy_inputs.png)
![Set B - Recon 1](figures/associative_networks_fig_006_setB_recon1.png)
![Set B - Recon 2](figures/associative_networks_fig_007_setB_recon2.png)
![Set B - Recon 3](figures/associative_networks_fig_008_setB_recon3.png)
![Set B - Recon 4](figures/associative_networks_fig_009_setB_recon4.png)


