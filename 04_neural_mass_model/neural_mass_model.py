#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path
import json

# ============================================================
# NEURAL MASS MODEL - JANSEN & DE RIT
# ============================================================

# -----------------------------
# Parameters
# -----------------------------
Wep = 135.0
Wpe = 0.8 * Wep
Wip = 0.25 * Wep
Wpi = 0.25 * Wep

Ae = 3.25
Ai = 22.0

ae = 100.0
ai = 50.0

kr = 0.56
v0 = 6.0
rmax = 5.0

dt = 1e-4
tmax = 20.0
t = np.arange(0, tmax + dt, dt)
N = len(t)

# reproducible white noise
rng = np.random.default_rng(42)
noise = rng.normal(loc=160.0, scale=200.0, size=N - 1)

PROJECT_DIR = Path(__file__).resolve().parent
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = []

# -----------------------------
# State variables
# -----------------------------
yp = np.zeros(N)
zp = np.zeros(N)

ye = np.zeros(N)
ze = np.zeros(N)

yi = np.zeros(N)
zi = np.zeros(N)

# -----------------------------
# Simulation
# -----------------------------
for k in range(N - 1):
    n = noise[k]

    vp = Wpe * ye[k] - Wpi * yi[k]
    ve = Wep * yp[k]
    vi = Wip * yp[k]

    rp = rmax / (1 + np.exp(-kr * (vp - v0)))
    re = rmax / (1 + np.exp(-kr * (ve - v0)))
    ri = rmax / (1 + np.exp(-kr * (vi - v0)))

    dyp = zp[k]
    dzp = Ae * ae * rp - 2 * ae * zp[k] - (ae ** 2) * yp[k]

    dye = ze[k]
    dze = Ae * ae * (re + n / Wep) - 2 * ae * ze[k] - (ae ** 2) * ye[k]

    dyi = zi[k]
    dzi = Ai * ai * ri - 2 * ai * zi[k] - (ai ** 2) * yi[k]

    yp[k + 1] = yp[k] + dyp * dt
    zp[k + 1] = zp[k] + dzp * dt

    ye[k + 1] = ye[k] + dye * dt
    ze[k + 1] = ze[k] + dze * dt

    yi[k + 1] = yi[k] + dyi * dt
    zi[k + 1] = zi[k] + dzi * dt

# -----------------------------
# EEG-like signal
# -----------------------------
eeg = Wpe * ye - Wpi * yi

# remove transient
transient = 2.0
start_idx = int(transient / dt)

t_ss = t[start_idx:]
eeg_ss = eeg[start_idx:]

# center signal for prettier plot and cleaner PSD
eeg_ss_centered = eeg_ss - np.mean(eeg_ss)

# -----------------------------
# PSD
# -----------------------------
fs = 1 / dt
f, Peeg = welch(
    eeg_ss_centered,
    fs=fs,
    nperseg=int(4 * fs),
    noverlap=int(2 * fs)
)

# ignore very low frequencies
mask = f >= 3.0

# dominant peak
peak_freq = f[mask][np.argmax(Peeg[mask])]
peak_power = Peeg[mask][np.argmax(Peeg[mask])]

# -----------------------------
# Better looking plots
# -----------------------------

# select a nice time window after transient
plot_start = 1.0   # seconds after steady-state begins
plot_duration = 2.0

i1 = int(plot_start / dt)
i2 = int((plot_start + plot_duration) / dt)

fig, ax = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

# EEG plot
ax[0].plot(
    t_ss[i1:i2],
    eeg_ss_centered[i1:i2],
    color='black',
    linewidth=1.2
)
ax[0].set_title('EEG signal after transient', fontsize=14, fontweight='bold')
ax[0].set_xlabel('Time (s)', fontsize=11)
ax[0].set_ylabel('EEG amplitude', fontsize=11)
ax[0].grid(True, alpha=0.3)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

# PSD plot
ax[1].plot(
    f[mask],
    Peeg[mask],
    color='black',
    linewidth=1.4
)
ax[1].axvline(
    peak_freq,
    color='gray',
    linestyle='--',
    linewidth=1.2,
    label=f'Peak = {peak_freq:.2f} Hz'
)
ax[1].set_title('Power spectral density', fontsize=14, fontweight='bold')
ax[1].set_xlabel('Frequency (Hz)', fontsize=11)
ax[1].set_ylabel('PSD', fontsize=11)
ax[1].set_xlim(3, 40)
ax[1].grid(True, alpha=0.3)
ax[1].legend(frameon=False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

fig_name = "neural_mass_model_fig_001_signal_and_psd.png"
fig.savefig(FIG_DIR / fig_name, dpi=300, bbox_inches="tight")
MANIFEST.append(fig_name)

plt.show()

# -----------------------------
# Final info
# -----------------------------
print('==============================')
print('Neural Mass Model')
print('==============================')
print(f'Wep = {Wep}')
print(f'Peak frequency = {peak_freq:.2f} Hz')
print(f'Peak power = {peak_power:.4e}')
print('==============================')

manifest_path = FIG_DIR / "neural_mass_model_manifest.json"
manifest_path.write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")
