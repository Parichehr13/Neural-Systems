# Hopfield Extensions Report

## Objective
Test three Hopfield-network extensions:
1. Dilution (damaged synapses),
2. Low M/N ratio with small symbolic patterns,
3. Sparse coding with modified Hebbian rule and threshold.

## Model Used in Code
### 1) Dilution
- Start from Hopfield training on image patterns.
- Damage mask: `mask_damage = (rand > dilution)`
- Effective weights: `W = W * mask_damage`
- Asynchronous flips until stability.

### 2) Low M/N
- Four 6x6 symbolic patterns stored.
- Corrupted pattern iteratively recovered by asynchronous updates.

### 3) Sparse coding
- Binary sparse patterns (0/1), activity level `a`.
- Learning rule:
  - `W += (Y-a)*(Y-a)^T`
- Switching condition:
  - `(Y-0.5) * (W*Y - teta) < 0`
- Update: `Y = 1 - Y` for selected neurons.

## Results
The script exports a chronological sequence of snapshots (`exercise09_fig_001` to `exercise09_fig_039`) across all three extensions.

Figure references:
- **Fig 1-19**: Dilution experiment (image-pattern recovery with damaged synapses).
- **Fig 20-31**: Low M/N experiment (6x6 symbolic patterns, showing recovery and occasional spurious states).
- **Fig 32-39**: Sparse-coding experiment (0/1 neurons, modified Hebbian learning, thresholded updates).

Selected examples:

![Hopfield Extensions - Fig 1](figures/exercise09_fig_001.png)
Fig 1: training images used for the dilution test.

![Hopfield Extensions - Fig 20](figures/exercise09_fig_020.png)
Fig 20: one low M/N recovery snapshot showing a partially structured intermediate state.

![Hopfield Extensions - Fig 30](figures/exercise09_fig_030.png)
Fig 30: low M/N run converging toward an attractor with some distortion/spurious content.

![Hopfield Extensions - Fig 39](figures/exercise09_fig_039.png)
Fig 39 (last image): final stable output of the **sparse-pattern** experiment.  
It starts from a perturbed version of pattern `X1` (about 15% random flips), then asynchronous updates with
`(Y-0.5)(WY-theta)<0`, `a=0.02`, and `theta=8` progressively remove noise.  
This frame is the convergence endpoint, where the network cleanly reconstructs the stored `L` shape.

## What Is The Last Image?
The last image (`exercise09_fig_039.png`) belongs to the **sparse patterns** section of the code (Part 3), not to dilution or low M/N tests.

How to interpret it:
- White pixels are neurons at state `1` (active), black pixels are neurons at state `0` (inactive).
- The target memory is pattern `X1`, which is an `L`-shaped sparse pattern in a 30x30 grid.
- The run starts from a perturbed version of `X1` (`perc = 0.15`, about 15% random flips).
- The network then performs asynchronous single-neuron updates using:
  - learning: `W = sum((Yk-a)(Yk-a)^T)` with `a = 0.02`,
  - switching test: `(Y - 0.5) * (WY - teta) < 0`,
  - update: `Y_i = 1 - Y_i` for one selected unstable neuron at a time.

Why this figure is the endpoint:
- In the loop, `L` is the number of unstable neurons.
- The script keeps updating until `L = 0` (no neuron violates the stability condition).
- A frame is plotted whenever `L % 25 == 0`; therefore the final frame can occur exactly at convergence (`L = 0`).
- So the last image is the **final stable attractor state** reached by the sparse Hopfield dynamics.

Practical reading:
- The clear recovered `L` indicates successful associative recall from a noisy input.
- Any small isolated white pixels are residual errors/spurious activations and depend on the threshold `teta` and noise level.

