#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# HETERO-ASSOCIATIVE NETWORK (16 inputs, 4 outputs)
# ============================================================

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def mat_to_vec(M):
    return M.reshape(-1)


def vec_to_mat(v, n=4):
    return v.reshape(n, n)


def normalize(v):
    nrm = np.linalg.norm(v)
    return v if nrm == 0 else v / nrm


def sigmoid(u, k):
    return 1.0 / (1.0 + np.exp(-k * u))


def add_noise_and_normalize(v, sigma, rng):
    v_noisy = v + sigma * rng.standard_normal(v.shape)
    return normalize(v_noisy)


def reconstruct_from_output(y, patterns_raw):
    """Reconstruct pattern from output activations and stored templates."""
    P = np.column_stack([mat_to_vec(p) for p in patterns_raw])  # (16, 4)
    v_rec = P @ y
    return vec_to_mat(v_rec, 4)


def save_fig(fig, filename, written):
    fig.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    written.append(filename)
    plt.close(fig)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
rng = np.random.default_rng(42)
written_files = []


# ------------------------------------------------------------
# 1. Build 4 bars on a 4x4 grid
# ------------------------------------------------------------
I1 = -np.ones((4, 4))                  # horizontal
I1[1, :] = 1

I2 = -np.ones((4, 4))                  # vertical
I2[:, 1] = 1

I3 = -np.ones((4, 4))                  # main diagonal
np.fill_diagonal(I3, 1)

I4 = -np.ones((4, 4))                  # anti-diagonal
I4[np.arange(4), np.arange(3, -1, -1)] = 1

patterns = [I1, I2, I3, I4]
pattern_names = ["Horizontal", "Vertical", "Main diagonal", "Anti-diagonal"]

fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(patterns[i], cmap="gray", vmin=-1, vmax=1)
    ax.set_title(pattern_names[i], fontsize=10)
    ax.axis("off")
fig.suptitle("Set B - Clean input bars", fontsize=13)
fig.tight_layout()
save_fig(fig, "associative_networks_fig_004_setB_inputs.png", written_files)


# ------------------------------------------------------------
# 2. Convert bars to normalized vectors and train with Hebb
# ------------------------------------------------------------
X = np.column_stack([normalize(mat_to_vec(p)) for p in patterns])  # (16, 4)
Y = np.eye(4)                                                       # one-hot outputs
W = Y @ X.T                                                         # Hebbian hetero-associative map


# ------------------------------------------------------------
# 3. One noisy realization for visualization
# ------------------------------------------------------------
sigma_demo = 0.35
k_demo = 10

X_noisy = np.column_stack([add_noise_and_normalize(X[:, i], sigma_demo, rng) for i in range(4)])

fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(vec_to_mat(X_noisy[:, i], 4), cmap="gray")
    ax.set_title(pattern_names[i], fontsize=10)
    ax.axis("off")
fig.suptitle(f"Set B - Noisy inputs (sigma={sigma_demo:.2f})", fontsize=13)
fig.tight_layout()
save_fig(fig, "associative_networks_fig_005_setB_noisy_inputs.png", written_files)


# ------------------------------------------------------------
# 4. Reconstruction demo for each noisy bar
# ------------------------------------------------------------
for i in range(4):
    y_linear = W @ X_noisy[:, i]
    y_sigmoid = sigmoid(y_linear, k_demo)
    I_recon = reconstruct_from_output(y_sigmoid, patterns)

    print("#" * 12 + f" Pattern {i + 1}")
    print("Linear output:", np.round(y_linear, 3))
    print(f"Sigmoid output (k={k_demo}):", np.round(y_sigmoid, 3))
    print(f"Predicted class: neuron {np.argmax(y_sigmoid) + 1}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    axes[0].imshow(vec_to_mat(X_noisy[:, i], 4), cmap="gray")
    axes[0].set_title("Noisy input")
    axes[0].axis("off")

    axes[1].bar(np.arange(1, 5), y_sigmoid, color="black")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(np.arange(1, 5))
    axes[1].set_xlabel("Output neuron")
    axes[1].set_ylabel("Activation")
    axes[1].set_title(f"Sigmoid output (k={k_demo})")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].imshow(I_recon, cmap="gray")
    axes[2].set_title("Reconstructed bar")
    axes[2].axis("off")

    fig.tight_layout()
    save_fig(fig, f"associative_networks_fig_00{6 + i}_setB_recon{i + 1}.png", written_files)


# ------------------------------------------------------------
# 5. Recognition performance vs noise and sigmoid slope
# ------------------------------------------------------------
sigma_values = np.linspace(0.0, 1.0, 11)
k_values = [1, 2, 5, 10, 20, 40]
n_trials = 200

accuracy = np.zeros((len(k_values), len(sigma_values)))

for ik, k in enumerate(k_values):
    for isg, sigma in enumerate(sigma_values):
        correct = 0
        total = 0

        for label in range(4):
            x_ref = X[:, label]
            for _ in range(n_trials):
                x_noisy = add_noise_and_normalize(x_ref, sigma, rng)
                y = sigmoid(W @ x_noisy, k)
                pred = np.argmax(y)
                if pred == label:
                    correct += 1
                total += 1

        accuracy[ik, isg] = correct / total

fig = plt.figure(figsize=(9, 5))
im = plt.imshow(
    accuracy,
    aspect="auto",
    origin="lower",
    cmap="gray",
    extent=[sigma_values[0], sigma_values[-1], 0, len(k_values) - 1],
)
plt.yticks(np.arange(len(k_values)), labels=k_values)
plt.xlabel("Noise standard deviation")
plt.ylabel("Sigmoid slope k")
plt.title("Set B - Recognition accuracy vs noise and sigmoid slope")
plt.colorbar(im, label="Accuracy")
plt.tight_layout()
save_fig(fig, "associative_networks_fig_010_setB_accuracy_heatmap.png", written_files)

print("\nRecognition accuracy table:")
for ik, k in enumerate(k_values):
    print(f"k = {k:>2}: {np.round(accuracy[ik], 3)}")


# ------------------------------------------------------------
# 6. Manifest
# ------------------------------------------------------------
manifest_path = FIG_DIR / "associative_networks_setB_manifest.json"
manifest_path.write_text(json.dumps(written_files, indent=2), encoding="utf-8")

print("\nSaved Set B figures:")
for f in written_files:
    print(f"- {f}")
