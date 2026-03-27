# Exercise 08 Report

## Objective
Hopfield network for image memory and recovery from corrupted patterns.

## Model Used in Code
- Images converted to binary then to {-1, +1}.
- Downsampled to 64x64.
- Pattern vectors: `Y1, Y2, Y3`.
- Hebbian memory:
  - `W = Y1*Y1^T + Y2*Y2^T + Y3*Y3^T`
- Asynchronous update:
  - Find unstable neurons: `idx where Y*(W*Y) < 0`
  - Randomly pick one unstable neuron and flip sign.
  - Repeat until no unstable neurons remain.

## Results
Stored images (after preprocessing):

![Exercise 08 - Stored Patterns](figures/exercise08_fig_001.png)

Initial corrupted image:

![Exercise 08 - Initial Corruption](figures/exercise08_fig_002.png)

Recovery trajectory snapshots:

![Exercise 08 - Recovery 1](figures/exercise08_fig_003.png)
![Exercise 08 - Recovery 2](figures/exercise08_fig_004.png)
![Exercise 08 - Recovery 3](figures/exercise08_fig_005.png)
![Exercise 08 - Recovery 4](figures/exercise08_fig_006.png)
![Exercise 08 - Recovery 5](figures/exercise08_fig_007.png)
![Exercise 08 - Recovery 6](figures/exercise08_fig_008.png)
![Exercise 08 - Recovery 7](figures/exercise08_fig_009.png)
![Exercise 08 - Recovery Final](figures/exercise08_fig_010.png)
