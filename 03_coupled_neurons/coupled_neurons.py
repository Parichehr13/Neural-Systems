#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


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
    fname = f"coupled_neurons_fig_{FIG_COUNTER:03d}_{label}.png"
    plt.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight")
    FIGS_WRITTEN.append(fname)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def exact_membrane_update(V, Vinf, tau, dt):
    """Exact update of first-order membrane dynamics over one time step."""
    return (V - Vinf) * np.exp(-dt / tau) + Vinf


def exp_decay(x, tau, dt):
    """Exponential decay over one time step."""
    return x * np.exp(-dt / tau)


def compute_rate(spike_times_ms, tmax_ms, transient_ms=50.0):
    """Mean firing rate after removing an initial transient."""
    valid = spike_times_ms[spike_times_ms >= transient_ms]
    duration_s = (tmax_ms - transient_ms) / 1000.0
    if duration_s <= 0:
        return np.nan
    return len(valid) / duration_s


def compute_last_isi_rate(spike_times_ms):
    """Instantaneous rate from the last inter-spike interval."""
    if len(spike_times_ms) < 2:
        return np.nan
    isi = spike_times_ms[-1] - spike_times_ms[-2]
    if isi <= 0:
        return np.nan
    return 1000.0 / isi


def compute_synchrony(spike_times_1, spike_times_2, tolerance_ms=2.0):
    """
    Simple synchrony summary:
    - coincidence fraction: fraction of spikes in neuron 1
      that have a spike in neuron 2 within tolerance
    - mean nearest lag
    """
    if len(spike_times_1) == 0 or len(spike_times_2) == 0:
        return {
            "coincidence_fraction": np.nan,
            "mean_nearest_lag_ms": np.nan,
            "all_nearest_lags_ms": np.array([])
        }

    nearest_lags = []
    for t1 in spike_times_1:
        nearest_lags.append(np.min(np.abs(spike_times_2 - t1)))

    nearest_lags = np.array(nearest_lags)

    return {
        "coincidence_fraction": np.mean(nearest_lags <= tolerance_ms),
        "mean_nearest_lag_ms": np.mean(nearest_lags),
        "all_nearest_lags_ms": nearest_lags
    }


# ============================================================
# MAIN SIMULATION
# ============================================================

def simulate_coupled_neurons(params):
    """
    Two integrate-and-fire neurons coupled by conductance-based synapses.
    No adaptation. Variable threshold included.
    """

    # -----------------------------
    # Unpack parameters
    # -----------------------------
    E0 = params["E0"]             # resting / reset potential
    taum = params["taum"]         # membrane time constant
    taut = params["taut"]         # threshold recovery time constant
    r = params["r"]               # membrane resistance
    g = 1.0 / r
    C = taum / r
    dt = params["dt"]
    tmax = params["tmax"]

    Vt_low = params["Vt_low"]
    Vt_high = params["Vt_high"]

    I1 = params["I1"]
    I2 = params["I2"]

    # Synapse 1 -> 2
    tau_12 = params["tau_12"]
    dP_12 = params["dP_12"]
    E_12 = params["E_12"]
    gsmax_12 = params["gsmax_12"]

    # Synapse 2 -> 1
    tau_21 = params["tau_21"]
    dP_21 = params["dP_21"]
    E_21 = params["E_21"]
    gsmax_21 = params["gsmax_21"]

    # -----------------------------
    # Time axis and state variables
    # -----------------------------
    t = np.arange(0.0, tmax + dt, dt)
    L = len(t)

    V1 = np.zeros(L)
    V2 = np.zeros(L)
    Vt1 = np.zeros(L)
    Vt2 = np.zeros(L)

    # P12 = synaptic opening produced by neuron 1 and acting on neuron 2
    # P21 = synaptic opening produced by neuron 2 and acting on neuron 1
    P12 = np.zeros(L)
    P21 = np.zeros(L)

    V1[0] = E0
    V2[0] = E0
    Vt1[0] = Vt_low
    Vt2[0] = Vt_low

    spike_idx_1 = []
    spike_idx_2 = []

    # -----------------------------
    # Simulation loop
    # -----------------------------
    for k in range(L - 1):

        # Incoming conductances
        gsyn_to_1 = gsmax_21 * P21[k]   # synapse from neuron 2 to neuron 1
        gsyn_to_2 = gsmax_12 * P12[k]   # synapse from neuron 1 to neuron 2

        # Equivalent parameters for neuron 1
        geq1 = g + gsyn_to_1
        Eeq1 = (g * E0 + gsyn_to_1 * E_21) / geq1
        req1 = 1.0 / geq1
        tau1 = C * req1
        Vinf1 = Eeq1 + req1 * I1

        # Equivalent parameters for neuron 2
        geq2 = g + gsyn_to_2
        Eeq2 = (g * E0 + gsyn_to_2 * E_12) / geq2
        req2 = 1.0 / geq2
        tau2 = C * req2
        Vinf2 = Eeq2 + req2 * I2

        # Decay threshold and synaptic variables
        Vt1_next = exact_membrane_update(Vt1[k], Vt_low, taut, dt)
        Vt2_next = exact_membrane_update(Vt2[k], Vt_low, taut, dt)

        P12_next = exp_decay(P12[k], tau_12, dt)
        P21_next = exp_decay(P21[k], tau_21, dt)

        # Update membrane potentials
        V1_next = exact_membrane_update(V1[k], Vinf1, tau1, dt)
        V2_next = exact_membrane_update(V2[k], Vinf2, tau2, dt)

        # --- Spike check neuron 1
        spiked_1 = V1_next >= Vt1_next
        if spiked_1:
            V1_next = E0
            Vt1_next = Vt_high
            P12_next = P12_next + dP_12 * (1.0 - P12_next)
            spike_idx_1.append(k + 1)

        # --- Spike check neuron 2
        spiked_2 = V2_next >= Vt2_next
        if spiked_2:
            V2_next = E0
            Vt2_next = Vt_high
            P21_next = P21_next + dP_21 * (1.0 - P21_next)
            spike_idx_2.append(k + 1)

        # Store next state
        V1[k + 1] = V1_next
        V2[k + 1] = V2_next
        Vt1[k + 1] = Vt1_next
        Vt2[k + 1] = Vt2_next
        P12[k + 1] = P12_next
        P21[k + 1] = P21_next

    # -----------------------------
    # Spike trains and summaries
    # -----------------------------
    spike_train_1 = np.zeros(L)
    spike_train_2 = np.zeros(L)
    spike_train_1[spike_idx_1] = 1
    spike_train_2[spike_idx_2] = 1

    spike_times_1 = t[spike_idx_1]
    spike_times_2 = t[spike_idx_2]

    results = {
        "t": t,
        "V1": V1,
        "V2": V2,
        "Vt1": Vt1,
        "Vt2": Vt2,
        "P12": P12,
        "P21": P21,
        "spike_train_1": spike_train_1,
        "spike_train_2": spike_train_2,
        "spike_idx_1": np.array(spike_idx_1),
        "spike_idx_2": np.array(spike_idx_2),
        "spike_times_1": spike_times_1,
        "spike_times_2": spike_times_2,
        "rate1_mean_hz": compute_rate(spike_times_1, tmax),
        "rate2_mean_hz": compute_rate(spike_times_2, tmax),
        "rate1_lastisi_hz": compute_last_isi_rate(spike_times_1),
        "rate2_lastisi_hz": compute_last_isi_rate(spike_times_2),
        "sync": compute_synchrony(spike_times_1, spike_times_2, tolerance_ms=2.0)
    }

    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_results(res, params):
    t = res["t"]

    # -----------------------------
    # Figure 1: membrane potentials + thresholds
    # -----------------------------
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax[0].plot(t, res["V1"], color="black", linewidth=1.2, label="V1")
    ax[0].plot(t, res["Vt1"], color="red", linestyle="--", linewidth=1.2, label="Vt1")
    ax[0].set_title("Neuron 1")
    ax[0].set_ylabel("mV")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, res["V2"], color="blue", linewidth=1.2, label="V2")
    ax[1].plot(t, res["Vt2"], color="orange", linestyle="--", linewidth=1.2, label="Vt2")
    ax[1].set_title("Neuron 2")
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("mV")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.suptitle("Membrane potentials and dynamic thresholds")
    plt.tight_layout()
    save_current_figure("membrane_and_thresholds")
    plt.show()

    # -----------------------------
    # Figure 2: zoomed comparison
    # -----------------------------
    zoom_tmax = min(120, params["tmax"])
    zoom_mask = t <= zoom_tmax

    plt.figure(figsize=(11, 5))
    plt.plot(t[zoom_mask], res["V1"][zoom_mask], color="black", linewidth=1.2, label="V1")
    plt.plot(t[zoom_mask], res["V2"][zoom_mask], color="blue", linewidth=1.2, label="V2")
    plt.plot(t[zoom_mask], res["Vt1"][zoom_mask], color="red", linestyle="--", linewidth=1.0, label="Vt1")
    plt.plot(t[zoom_mask], res["Vt2"][zoom_mask], color="orange", linestyle="--", linewidth=1.0, label="Vt2")
    plt.xlabel("time (ms)")
    plt.ylabel("mV")
    plt.title("Zoomed timing comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("zoomed_timing_comparison")
    plt.show()

    # -----------------------------
    # Figure 3: spike raster
    # -----------------------------
    plt.figure(figsize=(11, 3.5))
    plt.eventplot(
        [res["spike_times_1"], res["spike_times_2"]],
        colors=["black", "blue"],
        lineoffsets=[1, 0],
        linelengths=0.8
    )
    plt.yticks([1, 0], ["Neuron 1", "Neuron 2"])
    plt.xlabel("time (ms)")
    plt.title("Spike raster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_current_figure("spike_raster")
    plt.show()

    # -----------------------------
    # Figure 4: synaptic variables
    # -----------------------------
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax[0].plot(t, res["P12"], color="purple", linewidth=1.3)
    ax[0].set_ylabel("P12")
    ax[0].set_title("Synapse 1 -> 2")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, res["P21"], color="green", linewidth=1.3)
    ax[1].set_ylabel("P21")
    ax[1].set_xlabel("time (ms)")
    ax[1].set_title("Synapse 2 -> 1")
    ax[1].grid(True, alpha=0.3)

    fig.suptitle("Synaptic opening variables")
    plt.tight_layout()
    save_current_figure("synaptic_opening_variables")
    plt.show()

    # -----------------------------
    # Figure 5: nearest spike lag histogram
    # -----------------------------
    lags = res["sync"]["all_nearest_lags_ms"]
    if len(lags) > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(lags, bins=15)
        plt.xlabel("nearest spike lag (ms)")
        plt.ylabel("count")
        plt.title("Synchrony: nearest spike lag distribution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_current_figure("nearest_spike_lag_histogram")
        plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    params = {
        # Membrane and threshold
        "E0": -65.0,          # mV
        "taum": 30.0,         # ms
        "taut": 5.0,          # ms
        "r": 10.0,            # MOhm
        "dt": 0.01,           # ms
        "tmax": 150.0,        # ms
        "Vt_low": -55.0,      # mV
        "Vt_high": 50.0,      # mV

        # Constant current
        # r * I = 10 * 2.5 = 25 mV, as recommended
        "I1": 2.5,            # nA
        "I2": 2.5,            # nA

        # Synapse 1 -> 2
        # change E_12 to 0.0 for excitatory or -70.0 for inhibitory
        "tau_12": 10.0,       # ms
        "dP_12": 0.20,        # 0.03 - 0.6
        "E_12": 0.0,          # mV (excitatory)
        "gsmax_12": 2.0 / 10.0,   # r * gsmax = 2

        # Synapse 2 -> 1
        "tau_21": 10.0,       # ms
        "dP_21": 0.20,        # 0.03 - 0.6
        "E_21": -70.0,        # mV (inhibitory)
        "gsmax_21": 2.0 / 10.0,   # r * gsmax = 2
    }

    res = simulate_coupled_neurons(params)

    print("=== Coupled integrate-and-fire neurons ===")
    print(f"Neuron 1 mean rate: {res['rate1_mean_hz']:.2f} Hz")
    print(f"Neuron 2 mean rate: {res['rate2_mean_hz']:.2f} Hz")
    print(f"Neuron 1 last-ISI rate: {res['rate1_lastisi_hz']:.2f} Hz")
    print(f"Neuron 2 last-ISI rate: {res['rate2_lastisi_hz']:.2f} Hz")
    print(f"Synchrony coincidence fraction (+/-2 ms): {res['sync']['coincidence_fraction']:.3f}")
    print(f"Mean nearest spike lag: {res['sync']['mean_nearest_lag_ms']:.3f} ms")

    plot_results(res, params)

    if SAVE_FIGURES:
        manifest_path = FIG_DIR / "coupled_neurons_manifest.json"
        manifest_path.write_text(json.dumps(FIGS_WRITTEN, indent=2), encoding="utf-8")
        print(f"Saved {len(FIGS_WRITTEN)} figures to: {FIG_DIR}")
