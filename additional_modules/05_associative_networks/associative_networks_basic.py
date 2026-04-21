#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Hetero-associative network (10 inputs -> 3 outputs)
# ============================================================
# Task summary:
# - Build 3 binary input patterns in {-1,+1}, length 10
# - Normalize patterns and arrange as columns of X
# - Train with Hebbian hetero-associative rule: W = Y X^T
# - Corrupt one pattern (or all patterns) with Gaussian noise and renormalize
# - Compute outputs with:
#   1) linear neurons
#   2) sigmoid neurons in [0,1], centered at 0.5, for multiple slopes k


FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


def sigmoid(u, k):
    """Sigmoid in [0,1], centered at 0.5 as requested."""
    return 1.0 / (1.0 + np.exp(-k * (u - 0.5)))


def normalize_columns(x):
    """Normalize each column to unit norm."""
    x = x.copy().astype(float)
    for j in range(x.shape[1]):
        nrm = np.linalg.norm(x[:, j])
        if nrm > 0:
            x[:, j] /= nrm
    return x


def plot_pattern_outputs(
    pattern_idx,
    y_target,
    y_lin_clean,
    y_lin_noisy,
    y_sig10_clean,
    y_sig10_noisy,
    y_sig20_clean,
    y_sig20_noisy,
):
    """Bar comparison for one input pattern across three output neurons."""
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(3)  # 3 output neurons
    width = 0.11

    ax.bar(x - 3.0 * width, y_target, width=width, label="Target", color="#111111")
    ax.bar(x - 2.0 * width, y_lin_clean, width=width, label="Linear clean", color="#1f77b4")
    ax.bar(x - 1.0 * width, y_lin_noisy, width=width, label="Linear noisy", color="#4f9dd9")
    ax.bar(x + 0.0 * width, y_sig10_clean, width=width, label="Sigmoid k=10 clean", color="#ff7f0e")
    ax.bar(x + 1.0 * width, y_sig10_noisy, width=width, label="Sigmoid k=10 noisy", color="#ffb066")
    ax.bar(x + 2.0 * width, y_sig20_clean, width=width, label="Sigmoid k=20 clean", color="#2ca02c")
    ax.bar(x + 3.0 * width, y_sig20_noisy, width=width, label="Sigmoid k=20 noisy", color="#6ecf6e")

    ax.set_xticks(x)
    ax.set_xticklabels(["Neuron 1", "Neuron 2", "Neuron 3"])
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel("Output")
    ax.set_title(f"Set A - Pattern {pattern_idx + 1}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def main():
    # --------------------------------------------------------
    # 1) Build binary patterns and normalize
    # --------------------------------------------------------
    x_clean = 2 * np.round(RNG.random((10, 3))) - 1  # {-1,+1}
    x_clean = normalize_columns(x_clean)

    # Desired one-hot outputs
    y_target_all = np.eye(3)

    # --------------------------------------------------------
    # 2) Hebbian hetero-associative training
    # --------------------------------------------------------
    # W = sum_p y_p x_p^T = Y X^T
    w = y_target_all @ x_clean.T

    # --------------------------------------------------------
    # 3) Clean/noisy test patterns
    # --------------------------------------------------------
    sigma = 0.2
    x_noisy = x_clean + sigma * RNG.standard_normal((10, 3))
    x_noisy = normalize_columns(x_noisy)

    # Linear outputs
    y_clean = w @ x_clean
    y_noisy = w @ x_noisy

    # Sigmoid outputs for two slopes
    y_sig10_clean = sigmoid(y_clean, k=10)
    y_sig10_noisy = sigmoid(y_noisy, k=10)
    y_sig20_clean = sigmoid(y_clean, k=20)
    y_sig20_noisy = sigmoid(y_noisy, k=20)

    print("Clean linear outputs:")
    print(y_clean)
    print("\nNoisy linear outputs:")
    print(y_noisy)

    # --------------------------------------------------------
    # 4) Plot one figure per pattern (3 figures)
    # --------------------------------------------------------
    output_files = []
    for p in range(3):
        fig = plot_pattern_outputs(
            pattern_idx=p,
            y_target=y_target_all[:, p],
            y_lin_clean=y_clean[:, p],
            y_lin_noisy=y_noisy[:, p],
            y_sig10_clean=y_sig10_clean[:, p],
            y_sig10_noisy=y_sig10_noisy[:, p],
            y_sig20_clean=y_sig20_clean[:, p],
            y_sig20_noisy=y_sig20_noisy[:, p],
        )
        fname = f"associative_networks_fig_{p + 1:03d}_setA_pattern{p + 1}.png"
        fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_files.append(fname)

    manifest_path = FIG_DIR / "associative_networks_setA_manifest.json"
    manifest_path.write_text(json.dumps(output_files, indent=2), encoding="utf-8")

    print("\nSaved figures:")
    for f in output_files:
        print(f"- {f}")


if __name__ == "__main__":
    main()
