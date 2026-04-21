import os
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================
SAVE_FIGURES = True
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

if SAVE_FIGURES:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) ACTIVATION FUNCTION
# ============================================================
def sigmoid(u, slope=0.6):
    """
    Sigmoidal activation function with maximum activity equal to 1.
    """
    u = np.clip(u, -100, 100)  # numerical stability
    return 1.0 / (1.0 + np.exp(-slope * u))


# ============================================================
# 2) LATERAL CONNECTIVITY MATRIX
# ============================================================
def build_lateral_matrix(N, lex0=5, lin0=2, sigma_in=24, structure="linear"):
    """
    Build the lateral connectivity matrix L.

    Diagonal terms:
        self-excitation = lex0

    Off-diagonal terms:
        inhibition decreases with distance according to a Gaussian law

    structure:
        - 'linear'   : standard 1D chain
        - 'circular' : optional, avoids edge effects
    """
    idx = np.arange(N)
    D = np.abs(idx[:, None] - idx[None, :])

    if structure == "circular":
        D = np.minimum(D, N - D)
    elif structure != "linear":
        raise ValueError("structure must be 'linear' or 'circular'")

    L = -lin0 * np.exp(-(D ** 2) / (2 * sigma_in ** 2))
    np.fill_diagonal(L, lex0)

    return L


# ============================================================
# 3) INPUT STIMULI
# ============================================================
def rectangular_stimulus(N, baseline=5, high=15, width=None):
    """
    Rectangular stimulus for contrast enhancement.
    """
    if width is None:
        width = N // 8

    I = baseline * np.ones(N)
    center = N // 2
    start = center - width // 2
    stop = start + width
    I[start:stop] = high
    return I


def double_gaussian_stimulus(N, baseline=5, amp=10, c1=100, c2=120, sigma=7):
    """
    Two nearby Gaussian stimuli immersed in a high background.
    """
    n = np.arange(N)
    I = (
        baseline
        + amp * np.exp(-((n - c1) ** 2) / (2 * sigma ** 2))
        + amp * np.exp(-((n - c2) ** 2) / (2 * sigma ** 2))
    )
    return I


# ============================================================
# 4) NETWORK SIMULATION
# ============================================================
def simulate_competitive_network(I, L, threshold=6, tau=3, dt=0.1, n_steps=1000, slope=0.6):
    """
    Simulate the competitive network using Euler's method.

    Dynamics:
        dy/dt = (1/tau) * [ -y + sigmoid(I + L@y - threshold) ]
    """
    N = len(I)
    y = np.zeros((N, n_steps))

    for t in range(n_steps - 1):
        u = I + L @ y[:, t] - threshold
        y[:, t + 1] = y[:, t] + (dt / tau) * (-y[:, t] + sigmoid(u, slope=slope))

    return y


# ============================================================
# 5) UTILITIES
# ============================================================
def save_current_figure(filename):
    if SAVE_FIGURES:
        plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")


# ============================================================
# 6) PLOTTING FUNCTIONS
# ============================================================
def plot_lateral_profile(L, neuron_idx=None, title="Lateral connectivity profile"):
    """
    Plot the lateral synaptic profile entering one neuron.
    """
    N = L.shape[0]
    if neuron_idx is None:
        neuron_idx = N // 2

    x = np.arange(N)
    profile = L[neuron_idx, :]

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, profile, color="teal", linewidth=2.5, label="Lateral synapses")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.scatter(
        neuron_idx,
        profile[neuron_idx],
        color="gold",
        edgecolor="black",
        s=120,
        zorder=3,
        label="Self-excitation",
    )

    plt.fill_between(x, profile, 0, where=(profile < 0), color="skyblue", alpha=0.35)
    plt.fill_between(x, profile, 0, where=(profile > 0), color="orange", alpha=0.35)

    plt.title(title)
    plt.xlabel("Presynaptic neuron index")
    plt.ylabel("Synaptic weight")
    plt.xlim(0, N - 1)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()


def plot_snapshots(I, y, title, snapshot_steps=(10, 30, 70, 120, 300, 999)):
    """
    Plot input and selected activity snapshots in separate subplots.
    This avoids mixing input scale (5-15) with activity scale (0-1).
    """
    valid_steps = [s for s in snapshot_steps if s < y.shape[1]]
    colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(valid_steps)))

    plt.figure(figsize=(11, 7))

    # Input
    plt.subplot(2, 1, 1)
    plt.plot(I, color="black", linewidth=2.2)
    plt.title(title)
    plt.ylabel("Input")
    plt.xlim(0, len(I) - 1)
    plt.grid(True, alpha=0.2)

    # Activity snapshots
    plt.subplot(2, 1, 2)
    for step, c in zip(valid_steps, colors):
        plt.plot(y[:, step], color=c, linewidth=2, label=f"Step {step}")
    plt.xlabel("Neuron index")
    plt.ylabel("Activity")
    plt.xlim(0, len(I) - 1)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=9, ncol=2)

    plt.tight_layout()


def plot_final_comparison(I, y_final, y_ff, title):
    """
    Compare input, feedforward-only response, and competitive final response.
    """
    plt.figure(figsize=(11, 7))

    plt.subplot(2, 1, 1)
    plt.plot(I, color="forestgreen", linewidth=2.5)
    plt.title(title)
    plt.ylabel("Input")
    plt.xlim(0, len(I) - 1)
    plt.grid(True, alpha=0.2)

    plt.subplot(2, 1, 2)
    plt.plot(y_ff, color="royalblue", linewidth=2.5, linestyle="--", label="Feedforward only")
    plt.plot(y_final, color="crimson", linewidth=2.5, label="Competitive final output")
    plt.xlabel("Neuron index")
    plt.ylabel("Activity")
    plt.xlim(0, len(I) - 1)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.2)
    plt.legend()

    plt.tight_layout()


def parameter_exploration(I, N, threshold, tau, dt, n_steps, slope, structure="linear"):
    """
    Explore how the final competitive output changes by varying:
    - self-excitation lex0
    - inhibition strength lin0
    - inhibitory spread sigma_in
    """
    settings = [
        (3, 2, 24),
        (5, 2, 24),
        (7, 2, 24),
        (5, 1, 24),
        (5, 3, 24),
        (5, 2, 12),
        (5, 2, 36),
    ]

    y_ff = sigmoid(I - threshold, slope=slope)

    plt.figure(figsize=(12, 12))

    for i, (lex0_i, lin0_i, sigma_in_i) in enumerate(settings, start=1):
        L_i = build_lateral_matrix(
            N=N,
            lex0=lex0_i,
            lin0=lin0_i,
            sigma_in=sigma_in_i,
            structure=structure,
        )

        y_i = simulate_competitive_network(
            I=I,
            L=L_i,
            threshold=threshold,
            tau=tau,
            dt=dt,
            n_steps=n_steps,
            slope=slope,
        )

        ax = plt.subplot(len(settings), 1, i)
        ax.plot(y_ff, color="royalblue", linestyle="--", linewidth=2, label="Feedforward only")
        ax.plot(y_i[:, -1], color="crimson", linewidth=2, label="Competitive final")
        ax.set_xlim(0, N - 1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Act.")
        ax.set_title(f"lex0={lex0_i}, lin0={lin0_i}, sigma_in={sigma_in_i}")
        ax.grid(True, alpha=0.2)

        if i == 1:
            ax.legend(fontsize=8)

    plt.xlabel("Neuron index")
    plt.tight_layout()


# ============================================================
# 7) MAIN PARAMETERS
# ============================================================
N = 180
dt = 0.1
tau = 3
threshold = 6
slope = 0.6
n_steps = 1000

lex0 = 5
lin0 = 2
sigma_in = 24

# Main mandatory case
structure = "linear"

# Optional extension
# structure = "circular"


# ============================================================
# 8) BUILD LATERAL MATRIX
# ============================================================
L = build_lateral_matrix(
    N=N,
    lex0=lex0,
    lin0=lin0,
    sigma_in=sigma_in,
    structure=structure,
)


# ============================================================
# 9) FIGURE 1: LATERAL CONNECTIVITY PROFILE
# ============================================================
plot_lateral_profile(
    L,
    neuron_idx=N // 2,
    title=f"Lateral connectivity profile ({structure} structure)",
)
save_current_figure("lateral_connectivity.png")
plt.close()


# ============================================================
# 10) CASE A: RECTANGULAR STIMULUS (CONTRAST ENHANCEMENT)
# ============================================================
I_rect = rectangular_stimulus(N, baseline=5, high=15, width=N // 8)

y_rect = simulate_competitive_network(
    I=I_rect,
    L=L,
    threshold=threshold,
    tau=tau,
    dt=dt,
    n_steps=n_steps,
    slope=slope,
)

# Feedforward-only output
y_ff_rect = sigmoid(I_rect - threshold, slope=slope)

plot_snapshots(
    I_rect,
    y_rect,
    title="Rectangular stimulus: temporal evolution of competitive network",
    snapshot_steps=(10, 30, 70, 120, 300, 999),
)
save_current_figure("rectangular_evolution.png")
plt.close()

plot_final_comparison(
    I_rect,
    y_rect[:, -1],
    y_ff_rect,
    title="Rectangular stimulus: feedforward vs competitive response",
)
save_current_figure("rectangular_comparison.png")
plt.close()


# ============================================================
# 11) CASE B: TWO NEARBY GAUSSIAN STIMULI (IMPROVED RESOLUTION)
# ============================================================
I_gauss = double_gaussian_stimulus(
    N=N,
    baseline=5,
    amp=10,
    c1=100,
    c2=120,
    sigma=7,
)

y_gauss = simulate_competitive_network(
    I=I_gauss,
    L=L,
    threshold=threshold,
    tau=tau,
    dt=dt,
    n_steps=n_steps,
    slope=slope,
)

# Feedforward-only output
y_ff_gauss = sigmoid(I_gauss - threshold, slope=slope)

plot_snapshots(
    I_gauss,
    y_gauss,
    title="Two nearby Gaussian stimuli: temporal evolution of competitive network",
    snapshot_steps=(10, 30, 70, 120, 300, 999),
)
save_current_figure("gaussian_evolution.png")
plt.close()

plot_final_comparison(
    I_gauss,
    y_gauss[:, -1],
    y_ff_gauss,
    title="Two nearby Gaussian stimuli: feedforward vs competitive response",
)
save_current_figure("gaussian_comparison.png")
plt.close()


# ============================================================
# 12) PARAMETER EXPLORATION
# ============================================================
parameter_exploration(
    I=I_rect,
    N=N,
    threshold=threshold,
    tau=tau,
    dt=dt,
    n_steps=n_steps,
    slope=slope,
    structure=structure,
)
save_current_figure("parameter_exploration.png")
plt.close()


# ============================================================
# 13) OPTIONAL: RUN AGAIN WITH CIRCULAR CONNECTIVITY
# ============================================================
run_optional_circular = False

if run_optional_circular:
    structure_circ = "circular"

    L_circ = build_lateral_matrix(
        N=N,
        lex0=lex0,
        lin0=lin0,
        sigma_in=sigma_in,
        structure=structure_circ,
    )

    plot_lateral_profile(
        L_circ,
        neuron_idx=N // 2,
        title="Lateral connectivity profile (circular structure)",
    )
    save_current_figure("lateral_connectivity_circular.png")
    plt.close()

if SAVE_FIGURES:
    manifest = [
        "lateral_connectivity.png",
        "rectangular_evolution.png",
        "rectangular_comparison.png",
        "gaussian_evolution.png",
        "gaussian_comparison.png",
        "parameter_exploration.png",
    ]
    with open(FIG_DIR / "competitive_networks_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
