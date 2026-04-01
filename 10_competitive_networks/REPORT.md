# Competitive Neural Network with Lateral Inhibition

## Overview
This project investigates the behavior of a **competitive neural network** arranged as a one-dimensional chain of neurons.  
The main goal is to study how **lateral inhibition** modifies the network response over time and how this mechanism can enhance contrast and improve the separation between nearby input regions.

Unlike a simple feedforward architecture, a competitive network includes **recurrent lateral interactions** between neurons.  
Because of these interactions, neurons do not respond independently: the activation of one unit can suppress the activity of neighboring units, leading to a more selective and structured final response.

---

## Theoretical Background
Competitive networks belong to the family of **self-organized neural systems**, where structure emerges directly from input statistics rather than from an explicit teacher signal.  
A key biological inspiration is the phenomenon of **lateral inhibition**, observed in sensory systems such as the visual system, where neighboring receptors interact to sharpen spatial contrast.

In this framework:

- each neuron receives an external input;
- each neuron also interacts with the others through lateral connections;
- the network evolves dynamically until it reaches a stable response pattern.

Depending on the structure of lateral connectivity, different behaviors can emerge:

- **global competition**, which can produce a winner-takes-all response;
- **local competition**, which emphasizes local differences and enhances contours;
- **Mexican-hat interactions**, where short-range excitation and long-range inhibition support clustered activation and topological organization.

In this model, the implementation focuses on the **local inhibitory case**, which is especially relevant for studying **contrast enhancement**.

---

## Network Model
The network contains `N = 180` neurons arranged along a 1D chain.  
Each neuron receives:

1. an external input profile `I_j`,
2. recurrent lateral input from the other neurons.

The temporal evolution is modeled through a **first-order dynamical system** with sigmoidal activation:

`tau * dy_j(t)/dt = -y_j(t) + S(sum_{k=1..N}(l_jk * y_k(t)) + i_j)`

where:

- `y_j(t)` is the activity of neuron `j`,
- `i_j` is the external input,
- `l_jk` is the lateral interaction from neuron `k` to neuron `j`,
- `S(.)` is a sigmoid nonlinearity,
- `tau` is the time constant.

### Lateral Interaction Matrix
The implemented lateral matrix follows a **distance-dependent inhibitory law**:

- **self-excitation** on the diagonal:
  `L[i,i] = Lex0`

- **Gaussian inhibition** for different neurons:
  `L[i,j] = -Lin0 * exp(-(d(i,j)^2) / (2 * sigma_in^2)),  i != j`

where `d(i,j)` is the neuron-to-neuron distance.  
A **circular distance** option is used so that the chain behaves as a ring and edge effects are reduced.

### Activation Function
The neuron nonlinearity is a sigmoid:

`S(x) = 1 / (1 + exp(-k * (x - x0)))`

This keeps the response bounded and introduces a soft thresholding effect.

### Numerical Integration
The differential equation is solved with an **Euler discretization**:

`x(:,k+1) = x(:,k) + dt * (1/tau) * (-x(:,k) + S(I + L*x(:,k) - threshold))`

This allows the progressive observation of network activity from the initial state to its steady response.

---

## Simulation Objective
The purpose of the simulation is not only to obtain the final output, but also to analyze **how competition reshapes the activity profile over time**.

More specifically, the model aims to show that:

- lateral inhibition suppresses broadly distributed activity,
- neighboring neurons compete with each other,
- the response becomes more selective than the original feedforward activation,
- contours and transitions in the input profile become more evident.

This is the computational basis of **contrast enhancement**.

---

## Results
### 1. Temporal Evolution of Network Activity

![Competitive Network - Evolution](figures/competitive_networks_fig_001.png)

The figure shows the progressive evolution of neural activity during the simulation.

At the beginning, the response is relatively smooth and still resembles the input distribution.  
As time evolves, recurrent inhibitory interactions suppress surrounding activity while the most stimulated regions remain active.  
This leads to a sharpening of the response profile.

The main qualitative effects are:

- reduction of average activity level,
- suppression of weak neighboring responses,
- increased separation between active and inactive regions,
- enhancement of sharp transitions in the signal.

This behavior is coherent with the expected effect of **local competition**: the network does not simply reproduce the input, but transforms it into a more selective representation.

---

### 2. Comparison Between Feedforward and Competitive Responses

![Competitive Network - Output Comparison](figures/competitive_networks_fig_002.png)

The second figure compares:

- the original input profile,
- the output of a purely feedforward response,
- the final output of the competitive network.

The comparison highlights that the competitive network produces a response that is more structured and localized than the feedforward one.  
While the feedforward output mainly reflects input amplitude, the competitive output emphasizes regions where local contrast is strongest.

In other words, the lateral inhibitory mechanism improves **resolution** by making nearby peaks more distinguishable and by reducing diffuse activation.

---

## Interpretation
The essential idea behind this model is that neurons should not respond in isolation.  
When one neuron becomes active, it inhibits its neighbors; as a result, only the most strongly supported regions preserve a high response.

This mechanism enables the network to:

- reduce redundancy,
- produce sparse or semi-sparse responses,
- enhance local differences,
- approximate biologically plausible sensory preprocessing.

In sensory systems, this type of processing is useful because the nervous system is often more interested in **changes, boundaries, and contrasts** than in absolute uniform intensity values.

---

## Relation to Broader Competitive-Network Theory
This model represents one specific case of a broader class of competitive systems.

- If inhibition is global and strong, the system can evolve toward a **winner-takes-all** regime, where only one neuron remains strongly active.
- If inhibition decreases with distance, as in this implementation, the result is **local competition** and **contrast enhancement**.
- If short-range excitation and long-range inhibition are both present, the network exhibits a **Mexican-hat** organization, which can support activity clusters and, in higher-dimensional grids, the formation of **topological maps**.

Therefore, this model is a useful intermediate step between simple recurrent inhibition and more advanced self-organizing neural systems.

---

## Conclusion
This simulation demonstrates that adding lateral inhibition to a neural network fundamentally changes its computational behavior.

Competitive interactions transform the network from a passive input-response system into a dynamical processor able to:

- sharpen activity profiles,
- suppress weaker neighboring responses,
- enhance contrast,
- improve spatial selectivity.

The results confirm that even a relatively simple first-order recurrent model can reproduce an important computational principle observed in biological sensory systems.

---

## Key Takeaway
A competitive network with local lateral inhibition does not merely pass information forward: it **reorganizes** the response so that relevant spatial differences become more visible.  
This is why lateral inhibition is a fundamental mechanism in both computational neuroscience and biological sensory processing.
