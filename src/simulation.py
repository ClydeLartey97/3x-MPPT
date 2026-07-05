"""Simulation engine that runs both MPPT controllers through identical conditions.

The engine drives the Perturb and Observe baseline and the Adaptive Gradient-Scaled
controller through the same irradiance profile, records every quantity of interest, and
compares the energy each one harvests against the true maximum available. Harvested power is
clamped to be non-negative to represent the blocking diode of a real energy harvester, which
prevents reverse power flow when the operating voltage exceeds the open-circuit voltage in
very dim light.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_BANDS: list[tuple[float, float]] = [
    (0, 10),
    (10, 30),
    (30, 50),
    (50, 80),
    (80, 150),
    (150, 500),
    (500, 1100),
]

BAND_LABELS = {
    (0, 10): "Near darkness",
    (10, 30): "Very dim",
    (30, 50): "Dim corridor",
    (50, 80): "Moderate indoor",
    (80, 150): "Bright indoor",
    (150, 500): "Direct window light",
    (500, 1100): "Outdoor sunlight",
}


def _count_direction_reversals(voltage: np.ndarray) -> int:
    """Count the number of direction reversals in a voltage trajectory."""
    if len(voltage) < 3:
        return 0
    deltas = np.diff(voltage)
    signs = np.sign(deltas)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


@dataclass
class SimulationResults:
    """Container for all simulation data and the analyses derived from it."""

    time_seconds: np.ndarray
    irradiance: np.ndarray

    po_voltage: np.ndarray
    po_power: np.ndarray
    po_efficiency: np.ndarray

    ags_voltage: np.ndarray
    ags_power: np.ndarray
    ags_efficiency: np.ndarray
    ags_step_size: np.ndarray
    ags_mode: list[str]

    true_mpp_power: np.ndarray | None

    po_total_energy_wh: float = 0.0
    ags_total_energy_wh: float = 0.0
    true_total_energy_wh: float = 0.0
    improvement_percent: float = 0.0
    time_step_seconds: float = 1.0

    def _energy_wh(self, power: np.ndarray) -> float:
        """Integrate a power series in watts into energy in watt-hours."""
        return float(np.sum(power) * self.time_step_seconds / 3600.0)

    def compute_band_analysis(
        self, bands: list[tuple[float, float]] | None = None
    ) -> pd.DataFrame:
        """Break down performance by irradiance band.

        Parameters:
            bands: Optional list of (low, high) irradiance bounds in W/m^2. The default set
                spans near darkness through to outdoor sunlight.

        Returns:
            A DataFrame with the time spent, energy harvested, improvement, and average
            tracking efficiency of each controller within each band.
        """
        if bands is None:
            bands = DEFAULT_BANDS

        dt = self.time_step_seconds
        total_time = len(self.irradiance) * dt
        rows = []
        for low, high in bands:
            mask = (self.irradiance >= low) & (self.irradiance < high)
            time_in_band = float(np.sum(mask) * dt)
            po_energy = self._energy_wh(self.po_power[mask]) if np.any(mask) else 0.0
            ags_energy = self._energy_wh(self.ags_power[mask]) if np.any(mask) else 0.0
            improvement = (
                100.0 * (ags_energy - po_energy) / po_energy if po_energy > 1e-12 else np.nan
            )
            po_eff = float(np.nanmean(self.po_efficiency[mask])) if np.any(mask) else np.nan
            ags_eff = float(np.nanmean(self.ags_efficiency[mask])) if np.any(mask) else np.nan
            rows.append(
                {
                    "band_label": "%d - %d (%s)" % (low, high, BAND_LABELS.get((low, high), "")),
                    "time_in_band_seconds": time_in_band,
                    "time_in_band_percent": 100.0 * time_in_band / total_time if total_time else 0.0,
                    "po_energy_wh": po_energy,
                    "ags_energy_wh": ags_energy,
                    "improvement_percent": improvement,
                    "po_avg_efficiency": po_eff,
                    "ags_avg_efficiency": ags_eff,
                }
            )
        return pd.DataFrame(rows)

    def _detect_shifts(self) -> list[int]:
        """Auto-detect irradiance shift indices, where light changes sharply and quickly."""
        dt = self.time_step_seconds
        window = max(int(round(5.0 / dt)), 1)
        shifts = []
        previous = -window
        for i in range(window, len(self.irradiance)):
            reference = self.irradiance[i - window]
            if reference <= 1e-6:
                continue
            if abs(self.irradiance[i] - reference) / reference > 0.20:
                if i - previous >= window:
                    shifts.append(i)
                    previous = i
        return shifts

    def compute_convergence_metrics(
        self, shift_times: list[float] | None = None
    ) -> pd.DataFrame:
        """Measure how quickly each controller re-converges after an irradiance shift.

        Convergence is the time taken to reach within 95 per cent of the true maximum power
        point following the shift. If shift times are not supplied they are auto-detected as
        points where the irradiance changes by more than 20 per cent within five seconds.

        Parameters:
            shift_times: Optional list of shift times in seconds.

        Returns:
            A DataFrame with one row per shift and the convergence time of each controller.
        """
        if self.true_mpp_power is None:
            return pd.DataFrame(columns=["shift_time_s", "po_convergence_s", "ags_convergence_s"])

        dt = self.time_step_seconds
        if shift_times is None:
            shift_indices = self._detect_shifts()
        else:
            shift_indices = [int(round(t / dt)) for t in shift_times]

        search_limit = max(int(round(30.0 / dt)), 1)
        rows = []
        for start in shift_indices:
            end = min(start + search_limit, len(self.irradiance))
            row = {"shift_time_s": start * dt}
            for name, power in (("po", self.po_power), ("ags", self.ags_power)):
                converged = np.nan
                for j in range(start, end):
                    target = self.true_mpp_power[j]
                    if target <= 1e-12 or power[j] >= 0.95 * target:
                        converged = (j - start) * dt
                        break
                row["%s_convergence_s" % name] = converged
            rows.append(row)
        return pd.DataFrame(rows)

    def compute_oscillation_metrics(self) -> dict:
        """Count voltage direction reversals for each controller.

        Returns:
            A dictionary with the reversal count of each controller. More reversals mean
            more energy wasted chattering about the operating point.
        """
        return {
            "po_direction_reversals": _count_direction_reversals(self.po_voltage),
            "ags_direction_reversals": _count_direction_reversals(self.ags_voltage),
        }

    def to_csv(self, filepath: str) -> None:
        """Export the full time-series data to a CSV file."""
        frame = pd.DataFrame(
            {
                "time_seconds": self.time_seconds,
                "irradiance_wm2": self.irradiance,
                "po_voltage_v": self.po_voltage,
                "po_power_w": self.po_power,
                "po_efficiency": self.po_efficiency,
                "ags_voltage_v": self.ags_voltage,
                "ags_power_w": self.ags_power,
                "ags_efficiency": self.ags_efficiency,
                "ags_step_size_v": self.ags_step_size,
                "ags_mode": self.ags_mode,
            }
        )
        if self.true_mpp_power is not None:
            frame["true_mpp_power_w"] = self.true_mpp_power
        frame.to_csv(filepath, index=False)

    def summary(self) -> str:
        """Return a formatted text summary of the results."""
        duration_hours = len(self.irradiance) * self.time_step_seconds / 3600.0
        lines = []
        lines.append("=" * 60)
        lines.append("3x MPPT SIMULATION RESULTS")
        lines.append("=" * 60)
        lines.append(
            "Profile duration: %.1f hours (%d time steps at %gs resolution)"
            % (duration_hours, len(self.irradiance), self.time_step_seconds)
        )
        lines.append("")
        lines.append("ENERGY HARVESTED")
        lines.append("-" * 16)
        lines.append("P&O (baseline):    %8.3f mWh" % (self.po_total_energy_wh * 1000.0))
        lines.append("AGS-MPPT (3x):     %8.3f mWh" % (self.ags_total_energy_wh * 1000.0))
        lines.append("Improvement:       %+7.1f%%" % self.improvement_percent)
        lines.append("")
        if self.true_total_energy_wh > 0.0:
            po_track = 100.0 * self.po_total_energy_wh / self.true_total_energy_wh
            ags_track = 100.0 * self.ags_total_energy_wh / self.true_total_energy_wh
            lines.append(
                "True theoretical maximum: %.3f mWh" % (self.true_total_energy_wh * 1000.0)
            )
            lines.append("P&O tracking efficiency: %.1f%%" % po_track)
            lines.append("AGS tracking efficiency: %.1f%%" % ags_track)
            lines.append("")

        lines.append("PERFORMANCE BY LIGHT LEVEL")
        lines.append("-" * 26)
        lines.append(
            "%-22s %10s %10s %12s" % ("Band (W/m2)", "P&O (mWh)", "AGS (mWh)", "Improvement")
        )
        band_frame = self.compute_band_analysis()
        for _, r in band_frame.iterrows():
            improvement = r["improvement_percent"]
            improvement_text = "n/a" if np.isnan(improvement) else "%+.1f%%" % improvement
            lines.append(
                "%-22s %10.3f %10.3f %12s"
                % (
                    r["band_label"].split(" (")[0],
                    r["po_energy_wh"] * 1000.0,
                    r["ags_energy_wh"] * 1000.0,
                    improvement_text,
                )
            )
        lines.append("")

        convergence = self.compute_convergence_metrics()
        if not convergence.empty:
            po_mean = convergence["po_convergence_s"].mean(skipna=True)
            ags_mean = convergence["ags_convergence_s"].mean(skipna=True)
            lines.append("CONVERGENCE SPEED")
            lines.append("-" * 17)
            lines.append("Detected irradiance shifts: %d" % len(convergence))
            lines.append("Avg convergence time (P&O): %6.1f seconds" % (po_mean if not np.isnan(po_mean) else 0.0))
            lines.append("Avg convergence time (AGS): %6.1f seconds" % (ags_mean if not np.isnan(ags_mean) else 0.0))
            lines.append("")

        oscillation = self.compute_oscillation_metrics()
        lines.append("OSCILLATION")
        lines.append("-" * 11)
        lines.append("P&O direction reversals: %8d" % oscillation["po_direction_reversals"])
        lines.append("AGS direction reversals: %8d" % oscillation["ags_direction_reversals"])
        lines.append("=" * 60)
        return "\n".join(lines)


class SimulationEngine:
    """Runs the P&O and AGS-MPPT controllers side by side through identical conditions.

    For each time step the engine reads the current irradiance, optionally computes the true
    maximum power point for the efficiency calculation, lets each controller choose its
    operating voltage and harvest power, and logs everything. The true maximum power point is
    expensive to compute, so results are cached by irradiance to keep long runs practical.
    """

    def __init__(self, cell_model, po_algorithm, ags_algorithm) -> None:
        self.cell_model = cell_model
        self.po = po_algorithm
        self.ags = ags_algorithm
        self.results: SimulationResults | None = None
        self._mpp_cache: dict[float, float] = {}

    def _true_mpp_power(self, irradiance: float) -> float:
        """Return the true maximum power for an irradiance, cached by rounded value."""
        key = round(irradiance, 1)
        cached = self._mpp_cache.get(key)
        if cached is None:
            cached = self.cell_model.find_true_mpp(key)[2]
            self._mpp_cache[key] = cached
        return cached

    def run(
        self,
        time_array: np.ndarray,
        irradiance_array: np.ndarray,
        compute_true_mpp: bool = True,
        log_interval: int = 100,
        verbose: bool = True,
    ) -> SimulationResults:
        """Run the full simulation over a time and irradiance series.

        Parameters:
            time_array: Array of time stamps in seconds.
            irradiance_array: Array of irradiance values in W/m^2.
            compute_true_mpp: Whether to compute the true maximum power at each step, which
                is slower but enables the efficiency metrics.
            log_interval: Print a status line to the console every this many steps.
            verbose: Whether to print console output.

        Returns:
            A SimulationResults instance containing all logged data.
        """
        self.po.reset()
        self.ags.reset()

        num_steps = len(time_array)
        time_step = float(time_array[1] - time_array[0]) if num_steps > 1 else 1.0

        po_voltage = np.empty(num_steps)
        po_power = np.empty(num_steps)
        ags_voltage = np.empty(num_steps)
        ags_power = np.empty(num_steps)
        true_mpp = np.empty(num_steps) if compute_true_mpp else None
        po_eff = np.full(num_steps, np.nan)
        ags_eff = np.full(num_steps, np.nan)

        for i in range(num_steps):
            irradiance = float(irradiance_array[i])

            v_po, p_po = self.po.step(irradiance, self.cell_model)
            v_ags, p_ags = self.ags.step(irradiance, self.cell_model)
            p_po = max(p_po, 0.0)
            p_ags = max(p_ags, 0.0)

            po_voltage[i] = v_po
            po_power[i] = p_po
            ags_voltage[i] = v_ags
            ags_power[i] = p_ags

            if compute_true_mpp:
                mpp = self._true_mpp_power(irradiance)
                true_mpp[i] = mpp
                if mpp > 1e-12:
                    po_eff[i] = min(p_po / mpp, 1.0)
                    ags_eff[i] = min(p_ags / mpp, 1.0)

            if verbose and (i % log_interval == 0):
                self._log_line(time_array[i], irradiance, v_po, p_po, v_ags, p_ags,
                               true_mpp[i] if compute_true_mpp else None)

        po_energy = float(np.sum(po_power) * time_step / 3600.0)
        ags_energy = float(np.sum(ags_power) * time_step / 3600.0)
        true_energy = float(np.sum(true_mpp) * time_step / 3600.0) if compute_true_mpp else 0.0
        improvement = 100.0 * (ags_energy - po_energy) / po_energy if po_energy > 1e-12 else 0.0

        self.results = SimulationResults(
            time_seconds=np.asarray(time_array, dtype=float),
            irradiance=np.asarray(irradiance_array, dtype=float),
            po_voltage=po_voltage,
            po_power=po_power,
            po_efficiency=po_eff,
            ags_voltage=ags_voltage,
            ags_power=ags_power,
            ags_efficiency=ags_eff,
            ags_step_size=np.asarray(self.ags.step_size_history, dtype=float),
            ags_mode=list(self.ags.mode_history),
            true_mpp_power=true_mpp,
            po_total_energy_wh=po_energy,
            ags_total_energy_wh=ags_energy,
            true_total_energy_wh=true_energy,
            improvement_percent=improvement,
            time_step_seconds=time_step,
        )
        if verbose:
            print(self.results.summary())
        return self.results

    def _log_line(self, t, irradiance, v_po, p_po, v_ags, p_ags, true_mpp) -> None:
        """Print a single formatted status line to the console."""
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        mode = self.ags.mode_history[-1] if self.ags.mode_history else "normal"
        step = self.ags.step_size_history[-1] if self.ags.step_size_history else 0.0
        mpp_text = "%.3fmW" % (true_mpp * 1000.0) if true_mpp is not None else "n/a"
        print(
            "[%02d:%02d:%02d] G=%6.1f W/m2 | P&O: V=%.3fV P=%.3fmW | "
            "AGS: V=%.3fV P=%.3fmW step=%.4fmV mode=%s | True MPP: %s"
            % (
                hours, minutes, seconds, irradiance,
                v_po, p_po * 1000.0,
                v_ags, p_ags * 1000.0, step * 1000.0, mode,
                mpp_text,
            )
        )

    def run_batch(self, profiles: dict[str, tuple], **kwargs) -> dict[str, SimulationResults]:
        """Run the simulation across multiple named irradiance profiles.

        Parameters:
            profiles: Mapping of profile name to a (time_array, irradiance_array) tuple.

        Returns:
            A mapping of profile name to its SimulationResults.
        """
        outcomes = {}
        for name, (time_array, irradiance_array) in profiles.items():
            if kwargs.get("verbose", True):
                print("\nRunning profile: %s" % name)
            self._mpp_cache = {}
            outcomes[name] = self.run(time_array, irradiance_array, **kwargs)
        return outcomes
