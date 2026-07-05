"""Streamlit dashboard for the 3x MPPT simulation engine.

Launch with:
    streamlit run dashboard.py

The dashboard offers four tabs: a live simulation with real-time plots and parameter tuning,
a results analysis view, a cell model explorer, and a parameter sensitivity study. It uses the
dark theme configured in .streamlit/config.toml and the colour scheme shared with the plotting
module.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cell_model
import irradiance_profiles as profiles
from algorithms.ags_mppt import AdaptiveGradientScaledMPPT
from algorithms.perturb_observe import PerturbAndObserve
from simulation import SimulationEngine, SimulationResults
from visualisation import COLOUR_AGS, COLOUR_IRRADIANCE, COLOUR_PO, COLOUR_TRUE

st.set_page_config(page_title="3x MPPT Engine", layout="wide")


def build_profile(name, duration, time_step, seed, irradiance):
    """Return a (time_array, irradiance_array) tuple for the selected profile."""
    if name == "office":
        return profiles.generate_indoor_irradiance_profile(duration, time_step, seed)
    if name == "warehouse":
        return profiles.generate_warehouse_profile(duration, time_step, seed)
    if name == "retail":
        return profiles.generate_retail_profile(duration, time_step, seed)
    if name == "constant":
        return profiles.generate_constant_low_light(irradiance, duration, time_step, seed)
    return profiles.generate_stress_test_profile(duration, time_step, seed)


def ags_from_sidebar(params):
    """Construct an AGS controller from the sidebar parameter dictionary."""
    return AdaptiveGradientScaledMPPT(**params)


def sidebar_controls():
    """Render the sidebar and return the chosen settings."""
    st.sidebar.header("Simulation settings")
    profile = st.sidebar.selectbox(
        "Irradiance profile", ["office", "warehouse", "retail", "constant", "stress"]
    )
    duration = st.sidebar.slider("Duration (hours)", 0.5, 24.0, 4.0, 0.5)
    time_step = st.sidebar.select_slider("Time step (seconds)", [1.0, 2.0, 5.0], value=2.0)
    irradiance = st.sidebar.slider("Constant irradiance (W/m2)", 5.0, 200.0, 30.0, 5.0)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    st.sidebar.header("AGS-MPPT parameters")
    params = {
        "base_step": st.sidebar.slider("Base step (V)", 0.002, 0.02, 0.008, 0.001),
        "min_multiplier": st.sidebar.slider("Minimum multiplier", 0.05, 1.0, 0.1, 0.05),
        "max_multiplier": st.sidebar.slider("Maximum multiplier", 1.0, 6.0, 3.0, 0.5),
        "gradient_high_threshold": st.sidebar.slider("Gradient high threshold", 0.02, 0.5, 0.1, 0.01),
        "gradient_low_threshold": st.sidebar.slider("Gradient low threshold", 0.001, 0.05, 0.01, 0.001),
        "max_step": st.sidebar.slider("Maximum step (V)", 0.005, 0.05, 0.02, 0.005),
        "low_light_threshold": st.sidebar.slider("Low-light threshold (W/m2)", 0.0, 30.0, 8.0, 1.0),
        "hold_voltage": st.sidebar.slider("Hold voltage (V)", 0.1, 0.5, 0.26, 0.02),
    }
    return {
        "profile": profile,
        "duration": duration,
        "time_step": time_step,
        "irradiance": irradiance,
        "seed": int(seed),
        "ags_params": params,
    }


def run_live(settings, power_area, cumulative_area, step_area, readout_area, progress_bar):
    """Run the simulation, updating the live plots and readouts as it progresses."""
    time_array, irradiance_array = build_profile(
        settings["profile"], settings["duration"], settings["time_step"],
        settings["seed"], settings["irradiance"],
    )
    po = PerturbAndObserve()
    ags = ags_from_sidebar(settings["ags_params"])
    po.reset()
    ags.reset()

    num_steps = len(time_array)
    hours = time_array / 3600.0
    po_power = np.zeros(num_steps)
    ags_power = np.zeros(num_steps)
    true_power = np.zeros(num_steps)
    mpp_cache: dict[float, float] = {}

    updates = 60
    update_every = max(num_steps // updates, 1)

    for i in range(num_steps):
        G = float(irradiance_array[i])
        _, p_po = po.step(G, cell_model)
        v_ags, p_ags = ags.step(G, cell_model)
        po_power[i] = max(p_po, 0.0)
        ags_power[i] = max(p_ags, 0.0)
        key = round(G, 1)
        if key not in mpp_cache:
            mpp_cache[key] = cell_model.find_true_mpp(key)[2]
        true_power[i] = mpp_cache[key]

        if i % update_every == 0 or i == num_steps - 1:
            upto = i + 1
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hours[:upto], po_power[:upto] * 1000.0, color=COLOUR_PO, lw=0.8, label="P&O")
            ax.plot(hours[:upto], ags_power[:upto] * 1000.0, color=COLOUR_AGS, lw=0.8, label="AGS-MPPT")
            ax.plot(hours[:upto], true_power[:upto] * 1000.0, color=COLOUR_TRUE, lw=0.8, ls="--", label="True MPP")
            ax.set_xlabel("Time (hours)"); ax.set_ylabel("Power (mW)"); ax.legend(loc="upper right")
            power_area.pyplot(fig); plt.close(fig)

            dt = settings["time_step"]
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(hours[:upto], np.cumsum(po_power[:upto]) * dt / 3600.0 * 1000.0, color=COLOUR_PO, label="P&O")
            ax2.plot(hours[:upto], np.cumsum(ags_power[:upto]) * dt / 3600.0 * 1000.0, color=COLOUR_AGS, label="AGS-MPPT")
            ax2.set_xlabel("Time (hours)"); ax2.set_ylabel("Cumulative (mWh)"); ax2.legend(loc="upper left")
            cumulative_area.pyplot(fig2); plt.close(fig2)

            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ss = np.array(ags.step_size_history) * 1000.0
            ax3.plot(hours[:len(ss)], ss, color=COLOUR_AGS, lw=0.6)
            ax3.set_xlabel("Time (hours)"); ax3.set_ylabel("AGS step (mV)")
            step_area.pyplot(fig3); plt.close(fig3)

            readout_area.markdown(
                "**Irradiance:** %.1f W/m2  |  **P&O:** V=%.3f V, P=%.3f mW  |  "
                "**AGS:** V=%.3f V, P=%.3f mW  |  **True MPP:** %.3f mW"
                % (G, po.voltage, po_power[i] * 1000.0, v_ags, ags_power[i] * 1000.0, true_power[i] * 1000.0)
            )
            progress_bar.progress(upto / num_steps)

    # Build the results directly from the data the live loop already collected, rather
    # than running the whole simulation a second time.
    dt = settings["time_step"]
    po_eff = np.where(true_power > 1e-12, np.clip(po_power / np.maximum(true_power, 1e-12), 0.0, 1.0), np.nan)
    ags_eff = np.where(true_power > 1e-12, np.clip(ags_power / np.maximum(true_power, 1e-12), 0.0, 1.0), np.nan)
    po_energy = float(np.sum(po_power) * dt / 3600.0)
    ags_energy = float(np.sum(ags_power) * dt / 3600.0)
    results = SimulationResults(
        time_seconds=np.asarray(time_array, dtype=float),
        irradiance=np.asarray(irradiance_array, dtype=float),
        po_voltage=np.array(po.voltage_history),
        po_power=po_power,
        po_efficiency=po_eff,
        ags_voltage=np.array(ags.voltage_history),
        ags_power=ags_power,
        ags_efficiency=ags_eff,
        ags_step_size=np.array(ags.step_size_history),
        ags_mode=list(ags.mode_history),
        true_mpp_power=true_power,
        po_total_energy_wh=po_energy,
        ags_total_energy_wh=ags_energy,
        true_total_energy_wh=float(np.sum(true_power) * dt / 3600.0),
        improvement_percent=100.0 * (ags_energy - po_energy) / po_energy if po_energy > 1e-12 else 0.0,
        time_step_seconds=dt,
    )
    st.session_state["results"] = results
    st.session_state["ran"] = True


def tab_live(settings):
    """Render the live simulation tab."""
    st.subheader("Live simulation")
    st.caption("Adjust the AGS-MPPT parameters in the sidebar, then run the simulation.")
    if st.button("Run simulation", type="primary"):
        readout_area = st.empty()
        progress_bar = st.progress(0.0)
        power_area = st.empty()
        col1, col2 = st.columns(2)
        cumulative_area = col1.empty()
        step_area = col2.empty()
        run_live(settings, power_area, cumulative_area, step_area, readout_area, progress_bar)
        st.success("Simulation complete. See the Results Analysis tab for the full breakdown.")


def tab_results():
    """Render the results analysis tab."""
    st.subheader("Results analysis")
    if not st.session_state.get("ran"):
        st.info("Run a simulation on the Live Simulation tab first.")
        return
    results = st.session_state["results"]

    save_path = os.path.join("outputs", "plots", "dashboard")
    from visualisation import Visualiser
    Visualiser().generate_all_plots(results, cell_model=cell_model, save_path=save_path)
    st.text(results.summary())

    order = [
        "power_comparison", "cumulative_energy", "efficiency_comparison",
        "band_analysis", "step_size_adaptation", "voltage_tracking", "iv_curves",
    ]
    columns = st.columns(2)
    for index, name in enumerate(order):
        path = os.path.join(save_path, "%s.png" % name)
        if os.path.exists(path):
            columns[index % 2].image(path, caption=name.replace("_", " "))

    st.subheader("Band analysis")
    st.dataframe(results.compute_band_analysis())

    csv_bytes = results_to_csv_bytes(results)
    st.download_button("Download results CSV", csv_bytes, "mppt_results.csv", "text/csv")


def results_to_csv_bytes(results):
    """Return the results time series as CSV bytes for download."""
    frame = pd.DataFrame(
        {
            "time_seconds": results.time_seconds,
            "irradiance_wm2": results.irradiance,
            "po_power_w": results.po_power,
            "ags_power_w": results.ags_power,
        }
    )
    if results.true_mpp_power is not None:
        frame["true_mpp_power_w"] = results.true_mpp_power
    return frame.to_csv(index=False).encode("utf-8")


def tab_cell_model():
    """Render the cell model explorer tab."""
    st.subheader("Cell model explorer")
    st.caption("Change the cell parameters and see how the I-V and P-V curves respond.")
    col = st.columns(5)
    i_ph_ref = col[0].slider("I_ph_ref (A)", 0.02, 0.5, 0.1, 0.01)
    i_0 = col[1].select_slider("I_0 (A)", [1e-11, 1e-10, 1e-9, 1e-8], value=1e-10)
    r_s = col[2].slider("R_s (ohms)", 0.1, 2.0, 0.5, 0.1)
    r_sh = col[3].slider("R_sh (ohms)", 100.0, 1000.0, 500.0, 50.0)
    n = col[4].slider("Ideality n", 1.0, 2.0, 1.3, 0.1)

    saved = (cell_model.I_PH_REF, cell_model.I_0, cell_model.R_S, cell_model.R_SH, cell_model.N_IDEALITY)
    cell_model.I_PH_REF, cell_model.I_0, cell_model.R_S, cell_model.R_SH, cell_model.N_IDEALITY = (
        i_ph_ref, i_0, r_s, r_sh, n
    )
    try:
        levels = [1000, 200, 100, 50, 20]
        colours = plt.cm.viridis(np.linspace(0.0, 0.9, len(levels)))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for colour, G in zip(colours, levels):
            V, I, P = cell_model.generate_iv_curve(G, num_points=200)
            ax1.plot(V, I * 1000.0, color=colour, label="%d W/m2" % G)
            ax2.plot(V, P * 1000.0, color=colour, label="%d W/m2" % G)
            v_mpp, _, p_mpp = cell_model.find_true_mpp(G)
            ax2.plot(v_mpp, p_mpp * 1000.0, "o", color=colour)
            ax2.annotate("%.2f V" % v_mpp, (v_mpp, p_mpp * 1000.0), fontsize=7)
        ax1.set_xlabel("Voltage (V)"); ax1.set_ylabel("Current (mA)"); ax1.set_title("I-V"); ax1.legend()
        ax2.set_xlabel("Voltage (V)"); ax2.set_ylabel("Power (mW)"); ax2.set_title("P-V (MPP marked)"); ax2.legend()
        st.pyplot(fig); plt.close(fig)
    finally:
        cell_model.I_PH_REF, cell_model.I_0, cell_model.R_S, cell_model.R_SH, cell_model.N_IDEALITY = saved


def tab_sensitivity(settings):
    """Render the parameter sensitivity tab."""
    st.subheader("Parameter sensitivity")
    st.caption("Sweep two AGS parameters and view the improvement over P&O as a heatmap.")
    options = ["base_step", "max_multiplier", "gradient_high_threshold", "max_step", "hold_voltage"]
    col1, col2 = st.columns(2)
    param_x = col1.selectbox("Parameter X", options, index=1)
    param_y = col2.selectbox("Parameter Y", options, index=0)
    ranges = {
        "base_step": np.linspace(0.004, 0.016, 5),
        "max_multiplier": np.linspace(1.5, 5.0, 5),
        "gradient_high_threshold": np.linspace(0.04, 0.2, 5),
        "max_step": np.linspace(0.01, 0.04, 5),
        "hold_voltage": np.linspace(0.2, 0.4, 5),
    }
    if st.button("Run sensitivity sweep"):
        duration = min(settings["duration"], 3.0)
        time_array, irradiance_array = build_profile(
            settings["profile"], duration, settings["time_step"], settings["seed"], settings["irradiance"]
        )
        po = PerturbAndObserve()
        engine = SimulationEngine(cell_model, po, AdaptiveGradientScaledMPPT())
        po_result = engine.run(time_array, irradiance_array, compute_true_mpp=False, verbose=False)
        po_energy = po_result.po_total_energy_wh

        xs, ys = ranges[param_x], ranges[param_y]
        grid = np.zeros((len(ys), len(xs)))
        progress = st.progress(0.0)
        total = len(xs) * len(ys)
        done = 0
        for yi, yv in enumerate(ys):
            for xi, xv in enumerate(xs):
                base = dict(settings["ags_params"])
                base[param_x] = float(xv)
                base[param_y] = float(yv)
                ags = AdaptiveGradientScaledMPPT(**base)
                eng = SimulationEngine(cell_model, PerturbAndObserve(), ags)
                res = eng.run(time_array, irradiance_array, compute_true_mpp=False, verbose=False)
                grid[yi, xi] = res.improvement_percent
                done += 1
                progress.progress(done / total)

        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(["%.3g" % v for v in xs], rotation=45)
        ax.set_yticks(range(len(ys))); ax.set_yticklabels(["%.3g" % v for v in ys])
        ax.set_xlabel(param_x); ax.set_ylabel(param_y)
        ax.set_title("Improvement over P&O (%)")
        for yi in range(len(ys)):
            for xi in range(len(xs)):
                ax.text(xi, yi, "%.1f" % grid[yi, xi], ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, label="Improvement (%)")
        st.pyplot(fig); plt.close(fig)


def main():
    """Assemble the dashboard."""
    st.title("3x MPPT Engine")
    st.markdown("#### Low-Light Photovoltaic Energy Harvesting Simulation")
    settings = sidebar_controls()
    live, results, cell, sensitivity = st.tabs(
        ["Live Simulation", "Results Analysis", "Cell Model Explorer", "Parameter Sensitivity"]
    )
    with live:
        tab_live(settings)
    with results:
        tab_results()
    with cell:
        tab_cell_model()
    with sensitivity:
        tab_sensitivity(settings)


main()
