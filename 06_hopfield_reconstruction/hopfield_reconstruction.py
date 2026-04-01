#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================
def im2bw(I, th_value=128):
    """
    Convert uint8 grayscale image to binary image {0,1}.
    """
    return (I >= th_value).astype(np.int8)


def bw_to_pm1(Ibw):
    """
    Convert binary image {0,1} to Hopfield format {-1,+1}.
    """
    return (2 * Ibw - 1).astype(np.int8)


def reduce_to_64(I):
    """
    Reduce 128x128 image to 64x64 by selecting even rows/columns.
    """
    return I[::2, ::2]


def from_mtx_to_array(I):
    """
    Convert LxL matrix to vector of length L^2.
    """
    return I.reshape(-1)


def from_array_to_mtx(V):
    """
    Convert vector of length L^2 to LxL matrix.
    """
    L = int(np.sqrt(V.size))
    return V.reshape(L, L)


def corrupt_pattern(V, n_flip, rng=None):
    """
    Flip exactly n_flip randomly chosen neurons.
    """
    if rng is None:
        rng = np.random.default_rng()

    Vc = V.copy()
    idx = rng.choice(V.size, size=n_flip, replace=False)
    Vc[idx] *= -1
    return Vc, idx


def hebbian_weights(patterns, normalize=True):
    """
    Build Hopfield weight matrix using Hebb rule.
    patterns: list of vectors in {-1,+1}
    """
    N = patterns[0].size
    W = np.zeros((N, N), dtype=np.float32)

    for p in patterns:
        W += np.outer(p, p)

    np.fill_diagonal(W, 0)

    if normalize:
        W /= N

    return W


def hopfield_energy(W, y):
    """
    Hopfield energy function.
    """
    return -0.5 * y @ W @ y


def asynchronous_update(W, y0, max_steps=50000, rng=None, plot_every=None, title_prefix=""):
    """
    Asynchronous Hopfield update:
    - compute unstable neurons
    - randomly choose one
    - update it
    """
    if rng is None:
        rng = np.random.default_rng()

    y = y0.copy()
    energy_trace = [hopfield_energy(W, y)]
    states = [y.copy()]
    step = 0

    while step < max_steps:
        h = W @ y
        unstable = np.where(y * h < 0)[0]

        if unstable.size == 0:
            break

        i = rng.choice(unstable)
        y[i] = 1 if h[i] >= 0 else -1

        step += 1
        energy_trace.append(hopfield_energy(W, y))

        if plot_every is not None and step % plot_every == 0:
            states.append(y.copy())
            print(f"{title_prefix} step {step:5d} | unstable neurons: {unstable.size}")

    return y, np.array(energy_trace), states, step


def show_images(images, titles, cmap="gray", figsize=(12, 4)):
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, len(images), i)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def pattern_overlap(a, b):
    """
    Scalar product / overlap between two stored patterns.
    """
    return np.dot(a, b)


# ============================================================
# LOAD AND PREPROCESS IMAGES
# ============================================================
data = loadmat("imdemos.mat")

img_names = ["saturn", "vertigo", "coins"]
images_128 = [data[name] for name in img_names]

# Visualize original images
show_images(images_128, [f"{n} (128x128)" for n in img_names])

# Threshold to binary {0,1}, then scale to {-1,+1}, then reduce to 64x64
images_bin = [im2bw(img) for img in images_128]
images_pm1 = [bw_to_pm1(img) for img in images_bin]
images_64 = [reduce_to_64(img) for img in images_pm1]

# Visualize reduced Hopfield patterns
show_images(images_64, [f"{n} (64x64, +-1)" for n in img_names])

# Convert to vectors
patterns = [from_mtx_to_array(img) for img in images_64]
Y1, Y2, Y3 = patterns


# ============================================================
# CHECK CORRELATIONS BETWEEN STORED PATTERNS
# ============================================================
print("Pattern overlaps:")
print("saturn  vs vertigo:", pattern_overlap(Y1, Y2))
print("saturn  vs coins  :", pattern_overlap(Y1, Y3))
print("vertigo vs coins  :", pattern_overlap(Y2, Y3))


# ============================================================
# PART 1 - SINGLE IMAGE STORAGE
# ============================================================
rng = np.random.default_rng(42)

target_single = Y3   # coins
W_single = hebbian_weights([target_single], normalize=True)

n_flip = int(0.02 * target_single.size)   # 2% corruption
Y0_single, flipped_idx = corrupt_pattern(target_single, n_flip=n_flip, rng=rng)

print(f"\nSingle-image experiment: flipped {n_flip} neurons.")

# Visualize clean and corrupted
show_images(
    [from_array_to_mtx(target_single), from_array_to_mtx(Y0_single)],
    ["Original pattern", "Corrupted pattern"]
)

Yrec_single, E_single, states_single, steps_single = asynchronous_update(
    W_single,
    Y0_single,
    max_steps=50000,
    rng=rng,
    plot_every=50,
    title_prefix="[single]"
)

print(f"Single-image recovery finished in {steps_single} asynchronous updates.")

show_images(
    [from_array_to_mtx(target_single),
     from_array_to_mtx(Y0_single),
     from_array_to_mtx(Yrec_single)],
    ["Original", "Corrupted", "Recovered"]
)

plt.figure(figsize=(8, 4))
plt.plot(E_single)
plt.title("Hopfield energy during recovery (single image)")
plt.xlabel("Asynchronous update step")
plt.ylabel("Energy")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PART 2 - MULTI-IMAGE STORAGE
# ============================================================
W_multi = hebbian_weights([Y1, Y2, Y3], normalize=True)

target_multi = Y3   # corrupted coins again
Y0_multi, flipped_idx_multi = corrupt_pattern(target_multi, n_flip=n_flip, rng=rng)

print(f"\nMulti-image experiment: flipped {n_flip} neurons.")

show_images(
    [from_array_to_mtx(target_multi), from_array_to_mtx(Y0_multi)],
    ["Stored image", "Corrupted image"]
)

Yrec_multi, E_multi, states_multi, steps_multi = asynchronous_update(
    W_multi,
    Y0_multi,
    max_steps=50000,
    rng=rng,
    plot_every=50,
    title_prefix="[multi]"
)

print(f"Multi-image recovery finished in {steps_multi} asynchronous updates.")

show_images(
    [from_array_to_mtx(target_multi),
     from_array_to_mtx(Y0_multi),
     from_array_to_mtx(Yrec_multi)],
    ["Original", "Corrupted", "Recovered"]
)

plt.figure(figsize=(8, 4))
plt.plot(E_multi)
plt.title("Hopfield energy during recovery (multiple images)")
plt.xlabel("Asynchronous update step")
plt.ylabel("Energy")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# OPTIONAL - CHECK WHICH STORED IMAGE THE FINAL STATE RESEMBLES MOST
# ============================================================
final_overlaps = [pattern_overlap(Yrec_multi, p) for p in [Y1, Y2, Y3]]
print("\nFinal overlaps with stored patterns:")
for name, ov in zip(img_names, final_overlaps):
    print(f"{name:8s}: {ov}")
