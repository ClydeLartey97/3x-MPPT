"""Publication-quality plotting for the 3x MPPT simulation.

The Visualiser turns simulation results into a consistent set of figures. Every figure is
rendered at 300 dots per inch, uses a shared colour scheme, is labelled with units, and is
saved to disk as both PNG and SVG. Long time series are gently downsampled for display so
that the vector output remains a sensible size without changing the visible shape.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation import SimulationResults

# Shared colour scheme.
COLOUR_PO = "#E63946"
COLOUR_AGS = "#2A9D8F"
COLOUR_TRUE = "#264653"
COLOUR_IRRADIANCE = "#E9C46A"

_MAX_DISPLAY_POINTS = 4000


def _downsample(*arrays: np.ndarray, limit: int = _MAX_DISPLAY_POINTS):
    """Return the arrays strided down to at most the display limit of points."""
    length = len(arrays[0])
    stride = max(length // limit, 1)
    return [array[::stride] for array in arrays]


class Visualiser:
    """Generate and save all plots from a set of simulation results.

    Parameters:
        style: Optional matplotlib style name. The default clean grid style suits reports.
    """

    def __init__(self, style: str = "seaborn-v0_8-whitegrid") -> None:
        if style in plt.style.available:
            plt.style.use(style)

    def _save(self, fig, name: str, save_path: str) -> None:
        """Save a figure to disk as both PNG and SVG at 300 dots per inch."""
        os.makedirs(save_path, exist_ok=True)
        for extension in ("png", "svg"):
            fig.savefig(
                os.path.join(save_path, "%s.%s" % (name, extension)),
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)

    def plot_power_comparison(
        self, results: SimulationResults, save_path: str = "outputs/plots/"
    ) -> None:
        """Plot both controllers' extracted power over time with the true maximum and light."""
        hours, po, ags = _downsample(
            results.time_seconds / 3600.0, results.po_power * 1000.0, results.ags_power * 1000.0
        )
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(hours, po, color=COLOUR_PO, lw=1.0, label="P&O (baseline)")
        ax.plot(hours, ags, color=COLOUR_AGS, lw=1.0, label="AGS-MPPT (3x)")
        if results.true_mpp_power is not None:
            (true_mpp,) = _downsample(results.true_mpp_power * 1000.0)
            ax.plot(hours, true_mpp, color=COLOUR_TRUE, lw=1.0, ls="--", label="True MPP")
        ax.fill_between(
            hours, po, ags, where=(ags >= po), color=COLOUR_AGS, alpha=0.15,
            label="AGS advantage",
        )
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Power (mW)")
        ax.set_title("Extracted power: AGS-MPPT versus P&O")

        (irradiance,) = _downsample(results.irradiance)
        ax2 = ax.twinx()
        ax2.plot(hours, irradiance, color=COLOUR_IRRADIANCE, lw=0.7, alpha=0.6)
        ax2.set_ylabel("Irradiance (W/m2)", color=COLOUR_IRRADIANCE)
        ax2.tick_params(axis="y", labelcolor=COLOUR_IRRADIANCE)
        ax.legend(loc="upper right")
        self._save(fig, "power_comparison", save_path)

    def plot_efficiency_comparison(
        self, results: SimulationResults, save_path: str = "outputs/plots/"
    ) -> None:
        """Plot tracking efficiency, the share of the true maximum power captured, over time."""
        hours, po, ags = _downsample(
            results.time_seconds / 3600.0,
            results.po_efficiency * 100.0,
            results.ags_efficiency * 100.0,
        )
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(hours, po, color=COLOUR_PO, lw=1.0, label="P&O (baseline)")
        ax.plot(hours, ags, color=COLOUR_AGS, lw=1.0, label="AGS-MPPT (3x)")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Tracking efficiency (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Tracking efficiency versus the true maximum power point")
        ax.legend(loc="lower right")
        self._save(fig, "efficiency_comparison", save_path)

    def plot_band_analysis(
        self, results: SimulationResults, save_path: str = "outputs/plots/"
    ) -> None:
        """Plot energy harvested by each controller within each irradiance band."""
        frame = results.compute_band_analysis()
        frame = frame[frame["time_in_band_seconds"] > 0.0]
        labels = [label.split(" (")[0] for label in frame["band_label"]]
        positions = np.arange(len(labels))
        width = 0.38

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(positions - width / 2, frame["po_energy_wh"] * 1000.0, width,
               color=COLOUR_PO, label="P&O (baseline)")
        ax.bar(positions + width / 2, frame["ags_energy_wh"] * 1000.0, width,
               color=COLOUR_AGS, label="AGS-MPPT (3x)")
        for pos, improvement, ags_energy in zip(
            positions, frame["improvement_percent"], frame["ags_energy_wh"] * 1000.0
        ):
            if not np.isnan(improvement):
                ax.annotate(
                    "%+.1f%%" % improvement, (pos + width / 2, ags_energy),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8,
                    color=COLOUR_AGS,
                )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Irradiance band (W/m2)")
        ax.set_ylabel("Energy harvested (mWh)")
        ax.set_title("Energy harvested by irradiance band")
        ax.legend()
        self._save(fig, "band_analysis", save_path)

    def plot_step_size_adaptation(
        self, results: SimulationResults, save_path: str = "outputs/plots/"
    ) -> None:
        """Plot the AGS step size over time, coloured by mode, with the irradiance profile."""
        hours, step, irradiance = _downsample(
            results.time_seconds / 3600.0,
            results.ags_step_size * 1000.0,
            results.irradiance,
        )
        modes = np.array(results.ags_mode)[:: max(len(results.ags_mode) // _MAX_DISPLAY_POINTS, 1)]
        modes = modes[: len(hours)]
        mode_colours = {
            "normal": COLOUR_AGS,
            "reacquisition": COLOUR_PO,
            "damping": "#8338EC",
            "hold": "#6C757D",
        }

        fig, ax = plt.subplots(figsize=(13, 6))
        for mode, colour in mode_colours.items():
            mask = modes == mode
            if np.any(mask):
                ax.scatter(hours[mask], step[mask], s=3, color=colour, label=mode)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("AGS step size (mV)")
        ax.set_title("Adaptive step size and tracking mode over time")

        ax2 = ax.twinx()
        ax2.plot(hours, irradiance, color=COLOUR_IRRADIANCE, lw=0.7, alpha=0.5)
        ax2.set_ylabel("Irradiance (W/m2)", color=COLOUR_IRRADIANCE)
        ax2.tick_params(axis="y", labelcolor=COLOUR_IRRADIANCE)
        ax.legend(loc="upper right", markerscale=3)
        self._save(fig, "step_size_adaptation", save_path)

    def plot_voltage_tracking(
        self,
        results: SimulationResults,
        save_path: str = "outputs/plots/",
        time_window: tuple[float, float] | None = None,
    ) -> None:
        """Plot both controllers' voltage decisions over a window around a transition."""
        time_seconds = results.time_seconds
        if time_window is None:
            centre = self._most_interesting_time(results)
            time_window = (max(centre - 300.0, 0.0), centre + 300.0)
        mask = (time_seconds >= time_window[0]) & (time_seconds <= time_window[1])

        minutes = (time_seconds[mask] - time_window[0]) / 60.0
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(minutes, results.po_voltage[mask], color=COLOUR_PO, lw=1.0, label="P&O")
        ax.plot(minutes, results.ags_voltage[mask], color=COLOUR_AGS, lw=1.0, label="AGS-MPPT")
        ax.set_xlabel("Time within window (minutes)")
        ax.set_ylabel("Operating voltage (V)")
        ax.set_title("Operating voltage around an irradiance transition")

        ax2 = ax.twinx()
        ax2.plot(minutes, results.irradiance[mask], color=COLOUR_IRRADIANCE, lw=0.8, alpha=0.6)
        ax2.set_ylabel("Irradiance (W/m2)", color=COLOUR_IRRADIANCE)
        ax2.tick_params(axis="y", labelcolor=COLOUR_IRRADIANCE)
        ax.legend(loc="upper right")
        self._save(fig, "voltage_tracking", save_path)

    def _most_interesting_time(self, results: SimulationResults) -> float:
        """Return the time of the largest short-term irradiance change, in seconds."""
        irradiance = results.irradiance
        dt = results.time_step_seconds
        window = max(int(round(30.0 / dt)), 1)
        if len(irradiance) <= window:
            return float(results.time_seconds[len(results.time_seconds) // 2])
        change = np.abs(irradiance[window:] - irradiance[:-window])
        index = int(np.argmax(change)) + window
        return float(results.time_seconds[index])

    def plot_iv_curves(
        self,
        cell_model,
        irradiance_levels: list[float],
        save_path: str = "outputs/plots/",
    ) -> None:
        """Plot I-V and P-V curves at several irradiance levels to show the flattening peak."""
        colours = plt.cm.viridis(np.linspace(0.0, 0.9, len(irradiance_levels)))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for colour, irradiance in zip(colours, irradiance_levels):
            voltages, currents, powers = cell_model.generate_iv_curve(irradiance, num_points=300)
            ax1.plot(voltages, currents * 1000.0, color=colour, label="%d W/m2" % irradiance)
            ax2.plot(voltages, powers * 1000.0, color=colour, label="%d W/m2" % irradiance)
            v_mpp, _, p_mpp = cell_model.find_true_mpp(irradiance)
            ax2.plot(v_mpp, p_mpp * 1000.0, "o", color=colour, ms=5)
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Current (mA)")
        ax1.set_title("I-V curves")
        ax1.legend()
        ax2.set_xlabel("Voltage (V)")
        ax2.set_ylabel("Power (mW)")
        ax2.set_title("P-V curves with the true MPP marked")
        ax2.legend()
        fig.suptitle("Cell response flattens and broadens as light dims")
        fig.tight_layout()
        self._save(fig, "iv_curves", save_path)

    def plot_cumulative_energy(
        self, results: SimulationResults, save_path: str = "outputs/plots/"
    ) -> None:
        """Plot cumulative energy harvested over time; the widening gap is the AGS advantage."""
        dt = results.time_step_seconds
        po_cumulative = np.cumsum(results.po_power) * dt / 3600.0 * 1000.0
        ags_cumulative = np.cumsum(results.ags_power) * dt / 3600.0 * 1000.0
        hours, po_cumulative, ags_cumulative = _downsample(
            results.time_seconds / 3600.0, po_cumulative, ags_cumulative
        )
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(hours, po_cumulative, color=COLOUR_PO, lw=1.5, label="P&O (baseline)")
        ax.plot(hours, ags_cumulative, color=COLOUR_AGS, lw=1.5, label="AGS-MPPT (3x)")
        ax.fill_between(hours, po_cumulative, ags_cumulative, color=COLOUR_AGS, alpha=0.15)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Cumulative energy (mWh)")
        ax.set_title("Cumulative energy harvested")
        ax.legend(loc="upper left")
        self._save(fig, "cumulative_energy", save_path)

    def generate_all_plots(
        self,
        results: SimulationResults,
        cell_model=None,
        save_path: str = "outputs/plots/",
    ) -> None:
        """Generate every plot and save it to disk."""
        self.plot_power_comparison(results, save_path)
        self.plot_efficiency_comparison(results, save_path)
        self.plot_band_analysis(results, save_path)
        self.plot_step_size_adaptation(results, save_path)
        self.plot_voltage_tracking(results, save_path)
        self.plot_cumulative_energy(results, save_path)
        if cell_model is not None:
            self.plot_iv_curves(
                cell_model, [1000, 500, 200, 100, 50, 20], save_path
            )
