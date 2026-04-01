#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1) Activation function
# ============================================================
def sigmoid(u, slope=0.6):
    """
    Sigmoidal excitation function with maximum activity = 1.
    """
    u = np.clip(u, -100, 100)  # numerical stability
    return 1.0 / (1.0 + np.exp(-slope * u))


# ============================================================
# 2) Lateral synapse matrix
# ============================================================
def build_lateral_matrix(N, lex0=5, lin0=2, sigma_in=24, structure="circular"):
    """
    Build lateral connectivity matrix L.

    Diagonal terms: self-excitation = lex0
    Off-diagonal terms: inhibition decreases with distance
                        through a Gaussian law.

    structure:
        - 'linear'   -> simple distance |i-j|
        - 'circular' -> circular distance to avoid edge effects
    """
    idx = np.arange(N)
    D = np.abs(idx[:, None] - idx[None, :])

    if structure == "circular":
        D = np.minimum(D, N - D)
    elif structure != "linear":
        raise ValueError("structure must be 'linear' or 'circular'")

    L = -lin0 * np.exp(-(D**2) / (2 * sigma_in**2))
    np.fill_diagonal(L, lex0)

    return L


# ============================================================
# 3) Input stimuli
# ============================================================
def rectangular_stimulus(N, baseline=5, high=15, width=None):
    """
    Rectangular stimulus for contrast enhancement.
    Input varies approximately between 5 and 15.
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
    Useful to test improved resolution.
    """
    n = np.arange(N)
    I = (
        baseline
        + amp * np.exp(-((n - c1) ** 2) / (2 * sigma**2))
        + amp * np.exp(-((n - c2) ** 2) / (2 * sigma**2))
    )
    return I


# ============================================================
# 4) Network simulation
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
# 5) Plot helpers
# ============================================================
def plot_snapshots(I, y, title, snapshot_steps=(10, 30, 70, 120, 300, 999)):
    """
    Plot selected time snapshots of network evolution.
    """
    fig = plt.figure(figsize=(11, 6))
    plt.plot(I, color="tab:blue", linewidth=2.5, label="Input")

    for step in snapshot_steps:
        if step < y.shape[1]:
            plt.plot(y[:, step], linewidth=1.5, label=f"Output step {step}")

    plt.title(title)
    plt.xlabel("Neuron index")
    plt.ylabel("Activity")
    plt.xlim(0, len(I) - 1)
    plt.ylim(0, max(1.05, np.max(I) + 0.5))
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    return fig


def plot_final_comparison(I, y_final, y_ff, title):
    """
    Compare:
    - input
    - feedforward-only output
    - competitive-network final output
    """
    fig = plt.figure(figsize=(11, 7))

    plt.subplot(2, 1, 1)
    plt.plot(I, color="tab:green", linewidth=2)
    plt.title(title)
    plt.ylabel("Input")
    plt.xlim(0, len(I) - 1)

    plt.subplot(2, 1, 2)
    plt.plot(y_ff, color="tab:blue", linewidth=2, label="Feedforward only")
    plt.plot(y_final, color="tab:red", linewidth=2, label="Competitive final output")
    plt.xlabel("Neuron index")
    plt.ylabel("Activity")
    plt.xlim(0, len(I) - 1)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_lateral_profile(L, neuron_idx=None, title="Lateral synapses profile"):
    """
    Plot incoming lateral synapses for one neuron.
    Useful to visualize the Gaussian inhibition + self-excitation.
    """
    N = L.shape[0]
    if neuron_idx is None:
        neuron_idx = N // 2

    fig = plt.figure(figsize=(10, 4))
    plt.plot(L[neuron_idx, :], linewidth=2)
    plt.title(f"{title} (neuron {neuron_idx})")
    plt.xlabel("Presynaptic neuron index")
    plt.ylabel("Synaptic weight")
    plt.xlim(0, N - 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def parameter_exploration(I, N, threshold=6, tau=3, dt=0.1, n_steps=1000, slope=0.6, structure="circular"):
    """
    Show how the final competitive output changes by varying lex0 and sigma_in.
    """
    settings = [
        (3, 2, 24),
        (5, 2, 24),
        (7, 2, 24),
        (5, 2, 12),
        (5, 2, 36),
    ]

    fig = plt.figure(figsize=(12, 8))

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

        plt.subplot(len(settings), 1, i)
        plt.plot(sigmoid(I - threshold, slope=slope), "b", linewidth=1.8, label="Feedforward only")
        plt.plot(y_i[:, -1], "r", linewidth=1.8, label="Competitive final")
        plt.xlim(0, N - 1)
        plt.ylim(0, 1.05)
        plt.ylabel("Act.")
        plt.title(f"lex0={lex0_i}, lin0={lin0_i}, sigma_in={sigma_in_i}")
        if i == 1:
            plt.legend(fontsize=8)

    plt.xlabel("Neuron index")
    plt.tight_layout()
    return fig


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    # ============================================================
    # 6) Main parameters
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

    # Optional: choose 'linear' or 'circular'
    structure = "circular"

    # ============================================================
    # 7) Build lateral matrix
    # ============================================================
    L = build_lateral_matrix(
        N=N,
        lex0=lex0,
        lin0=lin0,
        sigma_in=sigma_in,
        structure=structure,
    )

    figures_dir = Path(__file__).resolve().parent.parent / "figures"

    fig_paths = [
        figures_dir / "competitive_networks_fig_001.png",
        figures_dir / "competitive_networks_fig_002.png",
        figures_dir / "competitive_networks_fig_003.png",
        figures_dir / "competitive_networks_fig_004.png",
        figures_dir / "competitive_networks_fig_005.png",
        figures_dir / "competitive_networks_fig_006.png",
    ]

    # Visualize the lateral synapse profile
    fig = plot_lateral_profile(L, neuron_idx=N // 2, title=f"Lateral connectivity ({structure} structure)")
    save_fig(fig, fig_paths[0])

    # ============================================================
    # 8) CASE A: Rectangular stimulus (contrast enhancement)
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

    # Feedforward-only output (without competition)
    y_ff_rect = sigmoid(I_rect - threshold, slope=slope)

    # Plot evolution and final comparison
    fig = plot_snapshots(
        I_rect,
        y_rect,
        title="Rectangular stimulus: temporal evolution of competitive network",
    )
    save_fig(fig, fig_paths[1])

    fig = plot_final_comparison(
        I_rect,
        y_rect[:, -1],
        y_ff_rect,
        title="Rectangular stimulus: feedforward vs competitive response",
    )
    save_fig(fig, fig_paths[2])

    # ============================================================
    # 9) CASE B: Two nearby Gaussian stimuli (improved resolution)
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

    # Feedforward-only output (without competition)
    y_ff_gauss = sigmoid(I_gauss - threshold, slope=slope)

    # Plot evolution and final comparison
    fig = plot_snapshots(
        I_gauss,
        y_gauss,
        title="Two nearby Gaussian stimuli: temporal evolution of competitive network",
    )
    save_fig(fig, fig_paths[3])

    fig = plot_final_comparison(
        I_gauss,
        y_gauss[:, -1],
        y_ff_gauss,
        title="Two nearby Gaussian stimuli: feedforward vs competitive response",
    )
    save_fig(fig, fig_paths[4])

    # ============================================================
    # 10) Optional: parameter exploration
    # ============================================================
    fig = parameter_exploration(
        I_rect,
        N,
        threshold=threshold,
        tau=tau,
        dt=dt,
        n_steps=n_steps,
        slope=slope,
        structure=structure,
    )
    save_fig(fig, fig_paths[5])

    manifest = [p.name for p in fig_paths]
    (figures_dir / "competitive_networks_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
