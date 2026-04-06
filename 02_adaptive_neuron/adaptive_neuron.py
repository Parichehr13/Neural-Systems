#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class IFAdaptParams:
    # Membrane / leak
    E0: float = -65.0      # mV, resting/reset potential
    r: float = 10.0        # MOhm
    taum: float = 30.0     # ms

    # Dynamic threshold (refractory effect)
    Vtl: float = -55.0     # mV, long-term threshold
    Vth: float = 50.0      # mV, threshold immediately after a spike
    taut: float = 10.0     # ms

    # Adaptation
    Ea: float = -90.0      # mV
    taua: float = 700.0    # ms (required range: 300-1000 ms)
    dPa: float = 0.10      # adaptation increment at spike (0.03-0.2)
    rgamax: float = 2.0    # dimensionless, r * ga,max (suggested 1-5)

    # Simulation
    dt: float = 0.05       # ms
    tend: float = 800.0    # ms

    @property
    def C(self) -> float:
        return self.taum / self.r   # nF (consistent with ms / MOhm)

    @property
    def gL(self) -> float:
        return 1.0 / self.r         # uS

    @property
    def gamax(self) -> float:
        return self.rgamax / self.r # uS


def simulate_if_adaptation(I_nA: float,
                           p: IFAdaptParams,
                           with_adaptation: bool = True):
    """
    Simulate an integrate-and-fire neuron with:
    - dynamic threshold (refractory effect)
    - optional spike-triggered adaptation conductance
    """
    t = np.arange(0.0, p.tend + p.dt, p.dt)
    n = len(t)

    V = np.empty(n)
    Vt = np.empty(n)
    Pa = np.empty(n)
    spikes = np.zeros(n, dtype=int)

    V[0] = p.E0
    Vt[0] = p.Vtl
    Pa[0] = 0.0

    spike_idx = []

    for k in range(n - 1):
        # Adaptation conductance
        ga = p.gamax * Pa[k] if with_adaptation else 0.0

        # Equivalent membrane parameters
        geq = p.gL + ga
        Eeq = (p.gL * p.E0 + ga * p.Ea) / geq
        req = 1.0 / geq
        Vinf = Eeq + req * I_nA
        tau_eff = p.C * req

        # Exact exponential update
        V[k + 1] = Vinf + (V[k] - Vinf) * np.exp(-p.dt / tau_eff)
        Vt[k + 1] = p.Vtl + (Vt[k] - p.Vtl) * np.exp(-p.dt / p.taut)
        Pa[k + 1] = Pa[k] * np.exp(-p.dt / p.taua)

        # Spike event
        if V[k + 1] >= Vt[k + 1]:
            spikes[k + 1] = 1
            spike_idx.append(k + 1)

            V[k + 1] = p.E0
            Vt[k + 1] = p.Vth

            if with_adaptation:
                Pa[k + 1] = Pa[k + 1] + p.dPa * (1.0 - Pa[k + 1])

    spike_times = t[spike_idx]
    isi = np.diff(spike_times) if len(spike_times) >= 2 else np.array([])

    result = {
        "t": t,
        "V": V,
        "Vt": Vt,
        "Pa": Pa,
        "spikes": spikes,
        "spike_times": spike_times,
        "isi_ms": isi,
        "n_spikes": len(spike_idx),
        "mean_rate_hz": 1000.0 / np.mean(isi) if len(isi) > 0 else 0.0,
        "steady_rate_hz": (
            1000.0 / np.mean(isi[-3:]) if len(isi) >= 3
            else (1000.0 / isi[-1] if len(isi) > 0 else 0.0)
        ),
    }
    return result


def firing_rate_curve(currents_nA, p: IFAdaptParams, with_adaptation: bool = True):
    """
    Compute f-I curve using the steady-state firing rate
    estimated from the last few ISIs.
    """
    rates = np.zeros_like(currents_nA, dtype=float)

    for i, I in enumerate(currents_nA):
        sim = simulate_if_adaptation(I, p, with_adaptation=with_adaptation)
        rates[i] = sim["steady_rate_hz"]

    return rates


def plot_comparison(sim_no_adapt, sim_adapt, I_nA):
    t = sim_adapt["t"]

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Membrane potential
    ax[0].plot(t, sim_no_adapt["V"], label="V without adaptation", linewidth=1.5)
    ax[0].plot(t, sim_adapt["V"], label="V with adaptation", linewidth=1.5)
    ax[0].plot(t, sim_adapt["Vt"], "--", label="Threshold Vt", linewidth=1.2)
    ax[0].set_ylabel("mV")
    ax[0].set_title(f"Response to constant current I = {I_nA:.2f} nA")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Adaptation variable
    ax[1].plot(t, sim_no_adapt["Pa"], label="Pa without adaptation", linewidth=1.5)
    ax[1].plot(t, sim_adapt["Pa"], label="Pa with adaptation", linewidth=1.5)
    ax[1].set_ylabel("Pa")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # Spike trains
    ax[2].vlines(sim_no_adapt["spike_times"], 0.00, 1.00, label="No adaptation", linewidth=1.2)
    ax[2].vlines(sim_adapt["spike_times"], 0.00, 0.75, label="With adaptation", linewidth=1.2)
    ax[2].set_ylabel("Spikes")
    ax[2].set_xlabel("Time (ms)")
    ax[2].set_ylim(0, 1.1)
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    return fig, ax


def main():
    # Parameters
    p = IFAdaptParams(
        taua=700.0,   # in the requested range
        Ea=-90.0,
        dPa=0.10,
        rgamax=2.0,
        dt=0.05,
        tend=800.0
    )

    # ------------------------------------------------------------
    # 1) Constant current response: WITHOUT vs WITH adaptation
    # ------------------------------------------------------------
    I_test = 4.0  # nA

    sim_no_adapt = simulate_if_adaptation(I_test, p, with_adaptation=False)
    sim_adapt = simulate_if_adaptation(I_test, p, with_adaptation=True)

    print("=== SINGLE-CURRENT COMPARISON ===")
    print(f"I = {I_test:.2f} nA")
    print(f"Without adaptation -> spikes: {sim_no_adapt['n_spikes']}, "
          f"steady-state rate: {sim_no_adapt['steady_rate_hz']:.2f} Hz")
    print(f"With adaptation    -> spikes: {sim_adapt['n_spikes']}, "
          f"steady-state rate: {sim_adapt['steady_rate_hz']:.2f} Hz")

    plot_comparison(sim_no_adapt, sim_adapt, I_test)

    # ------------------------------------------------------------
    # 2) Optional current-discharge rate curve (f-I curve)
    # ------------------------------------------------------------
    currents = np.arange(0.0, 10.5, 0.5)

    rates_no_adapt = firing_rate_curve(currents, p, with_adaptation=False)
    rates_adapt = firing_rate_curve(currents, p, with_adaptation=True)

    plt.figure(figsize=(8, 5))
    plt.plot(currents, rates_no_adapt, "o-", label="Without adaptation", linewidth=1.5)
    plt.plot(currents, rates_adapt, "s-", label="With adaptation", linewidth=1.5)
    plt.xlabel("Input current (nA)")
    plt.ylabel("Steady-state firing rate (Hz)")
    plt.title("Current-discharge rate (f-I curve)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()