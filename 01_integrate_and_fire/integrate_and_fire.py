#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import json


# ============================================================
# PARAMETERS
# ============================================================

@dataclass
class Stage1Params:
    E0: float = -65.0      # mV, resting/reset potential
    Vth: float = -50.0     # mV, constant threshold
    tau_m: float = 30.0    # ms
    r: float = 10.0        # MOhm
    dt: float = 0.05       # ms
    tend: float = 300.0    # ms


@dataclass
class Stage2Params:
    E0: float = -65.0      # mV, resting/reset potential
    VtL: float = -55.0     # mV, long-term threshold
    VtH: float = 0.0       # mV, threshold after spike
    tau_m: float = 30.0    # ms
    tau_t: float = 10.0    # ms
    r: float = 10.0        # MOhm
    dt: float = 0.05       # ms
    tend: float = 300.0    # ms


SAVE_FIGURES = True
PROJECT_DIR = Path(__file__).resolve().parent
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIGS_WRITTEN = []
FIG_COUNTER = 0


def save_current_figure(label):
    global FIG_COUNTER
    if not SAVE_FIGURES:
        return
    FIG_COUNTER += 1
    fname = f"integrate_and_fire_fig_{FIG_COUNTER:03d}_{label}.png"
    plt.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight")
    FIGS_WRITTEN.append(fname)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_time_vector(tend, dt):
    return np.arange(0.0, tend + dt, dt)


def rectified_sinusoid_current(t, Imax=4.0):
    """
    Example variable current for Stage 1.
    """
    return np.abs(Imax * np.sin(np.pi * t / t[-1]))


def compute_rate_from_spikes(spike_times_ms):
    """
    Estimate firing rate from the last interspike interval.
    Returns 0 if fewer than 2 spikes are present.
    """
    if len(spike_times_ms) < 2:
        return 0.0
    isi = spike_times_ms[-1] - spike_times_ms[-2]  # ms
    return 1000.0 / isi


def plot_spike_train(ax, spike_times, t_end, title="Spikes"):
    ax.vlines(spike_times, 0, 1, color='k', linewidth=1.2)
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


# ============================================================
# STAGE 1: INTEGRATE-AND-FIRE WITH CONSTANT THRESHOLD
# ============================================================

def simulate_stage1(I, p: Stage1Params):
    """
    Simulate Stage 1 integrate-and-fire neuron.
    I can be:
    - scalar: constant current
    - array of same length as time vector: time-varying current
    """
    t = create_time_vector(p.tend, p.dt)
    n = len(t)

    if np.isscalar(I):
        I_vec = np.full(n, I, dtype=float)
    else:
        I_vec = np.asarray(I, dtype=float)
        if len(I_vec) != n:
            raise ValueError("Input current vector must have same length as time vector.")

    V = np.zeros(n)
    V[0] = p.E0

    spike_idx = []

    for k in range(n - 1):
        V_inf = p.E0 + p.r * I_vec[k]

        # exact discrete-time solution for constant current during dt
        V[k + 1] = (V[k] - V_inf) * np.exp(-p.dt / p.tau_m) + V_inf

        if V[k + 1] >= p.Vth:
            V[k + 1] = p.E0
            spike_idx.append(k + 1)

    spike_times = t[spike_idx]

    return {
        "t": t,
        "I": I_vec,
        "V": V,
        "spike_idx": spike_idx,
        "spike_times": spike_times,
        "rate_hz": compute_rate_from_spikes(spike_times)
    }


def plot_stage1_variable_current(sim, p: Stage1Params):
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    ax[0].plot(sim["t"], sim["I"], 'k')
    ax[0].set_ylabel("I (nA)")
    ax[0].set_title("Input current")
    ax[0].grid(alpha=0.3)

    ax[1].plot(sim["t"], sim["V"], 'k', label="V")
    ax[1].axhline(p.Vth, color='r', linestyle='--', label="Vth")
    ax[1].set_ylabel("V (mV)")
    ax[1].set_title("Membrane potential")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plot_spike_train(ax[2], sim["spike_times"], sim["t"][-1], title="Output spikes")

    fig.suptitle("Stage 1 - Integrate and fire with constant threshold", fontsize=13)
    fig.tight_layout()
    save_current_figure("stage1_dynamics")
    plt.show()


def stage1_fI_curve(currents, p: Stage1Params):
    rates = np.zeros(len(currents))

    for i, I in enumerate(currents):
        sim = simulate_stage1(I, p)
        rates[i] = sim["rate_hz"]

    return rates


def plot_fI_curve(currents, rates, title, fig_label):
    plt.figure(figsize=(9, 6))
    plt.plot(currents, rates, 'o-k', linewidth=1.5, markersize=5)
    plt.xlabel("Input current (nA)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_current_figure(fig_label)
    plt.show()


# ============================================================
# STAGE 2: INTEGRATE-AND-FIRE WITH VARIABLE THRESHOLD
# ============================================================

def simulate_stage2(I, p: Stage2Params):
    """
    Simulate Stage 2 integrate-and-fire neuron with variable threshold.
    I is assumed constant.
    """
    t = create_time_vector(p.tend, p.dt)
    n = len(t)

    V = np.zeros(n)
    Vt = np.zeros(n)

    V[0] = p.E0
    Vt[0] = p.VtL

    spike_idx = []

    for k in range(n - 1):
        V_inf = p.E0 + p.r * I

        V[k + 1] = (V[k] - V_inf) * np.exp(-p.dt / p.tau_m) + V_inf
        Vt[k + 1] = (Vt[k] - p.VtL) * np.exp(-p.dt / p.tau_t) + p.VtL

        if V[k + 1] >= Vt[k + 1]:
            V[k + 1] = p.E0
            Vt[k + 1] = p.VtH
            spike_idx.append(k + 1)

    spike_times = t[spike_idx]

    return {
        "t": t,
        "V": V,
        "Vt": Vt,
        "spike_idx": spike_idx,
        "spike_times": spike_times,
        "rate_hz": compute_rate_from_spikes(spike_times)
    }


def plot_stage2_single_current(sim, I):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax[0].plot(sim["t"], sim["V"], 'k', label="V")
    ax[0].plot(sim["t"], sim["Vt"], 'r', label="Vt")
    ax[0].set_ylabel("mV")
    ax[0].set_title(f"Membrane potential and dynamic threshold (I = {I:.2f} nA)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    plot_spike_train(ax[1], sim["spike_times"], sim["t"][-1], title="Output spikes")

    fig.suptitle("Stage 2 - Integrate and fire with variable threshold", fontsize=13)
    fig.tight_layout()
    save_current_figure("stage2_dynamics")
    plt.show()


def stage2_fI_curve(currents, p: Stage2Params):
    rates = np.zeros(len(currents))

    for i, I in enumerate(currents):
        sim = simulate_stage2(I, p)
        rates[i] = sim["rate_hz"]

    return rates


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------
    p1 = Stage1Params(
        E0=-65.0,
        Vth=-50.0,
        tau_m=30.0,
        r=10.0,
        dt=0.05,
        tend=300.0
    )

    t1 = create_time_vector(p1.tend, p1.dt)
    I_var = rectified_sinusoid_current(t1, Imax=4.0)

    sim1_var = simulate_stage1(I_var, p1)
    plot_stage1_variable_current(sim1_var, p1)

    currents = np.arange(0.0, 11.0, 0.5)
    rates1 = stage1_fI_curve(currents, p1)
    plot_fI_curve(currents, rates1, "Stage 1 - Current-frequency curve", "stage1_fi_curve")

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------
    p2 = Stage2Params(
        E0=-65.0,
        VtL=-55.0,
        VtH=0.0,
        tau_m=30.0,
        tau_t=10.0,
        r=10.0,
        dt=0.05,
        tend=300.0
    )

    I_test = 4.0
    sim2_single = simulate_stage2(I_test, p2)
    plot_stage2_single_current(sim2_single, I_test)

    rates2 = stage2_fI_curve(currents, p2)
    plot_fI_curve(currents, rates2, "Stage 2 - Current-frequency curve", "stage2_fi_curve")

    if SAVE_FIGURES:
        manifest_path = FIG_DIR / "integrate_and_fire_manifest.json"
        manifest_path.write_text(json.dumps(FIGS_WRITTEN, indent=2), encoding="utf-8")
        print(f"Saved {len(FIGS_WRITTEN)} figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
