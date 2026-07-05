"""Integration tests for the full simulation engine."""

import os

import cell_model
import irradiance_profiles as profiles
from algorithms.ags_mppt import AdaptiveGradientScaledMPPT
from algorithms.perturb_observe import PerturbAndObserve
from simulation import SimulationEngine


def _engine():
    """Return a simulation engine with fresh controllers."""
    return SimulationEngine(cell_model, PerturbAndObserve(), AdaptiveGradientScaledMPPT())


def test_full_simulation_runs():
    """A full one-hour simulation should complete without errors."""
    time_array, irradiance = profiles.generate_indoor_irradiance_profile(1.0, 1.0)
    results = _engine().run(time_array, irradiance, verbose=False)
    assert len(results.po_power) == len(time_array)
    assert len(results.ags_power) == len(time_array)
    assert results.ags_total_energy_wh >= 0.0


def test_results_export_csv(tmp_path):
    """Results should export to a valid CSV file."""
    time_array, irradiance = profiles.generate_constant_low_light(50.0, 0.05, 1.0)
    results = _engine().run(time_array, irradiance, verbose=False)
    target = os.path.join(tmp_path, "results.csv")
    results.to_csv(target)
    assert os.path.exists(target)
    with open(target, encoding="utf-8") as handle:
        header = handle.readline()
    assert "time_seconds" in header and "ags_power_w" in header


def test_band_analysis_covers_all_time():
    """The sum of time in all bands should equal the total simulation time."""
    time_array, irradiance = profiles.generate_indoor_irradiance_profile(1.0, 1.0)
    results = _engine().run(time_array, irradiance, verbose=False)
    frame = results.compute_band_analysis()
    total_band_time = frame["time_in_band_seconds"].sum()
    expected = len(time_array) * results.time_step_seconds
    assert abs(total_band_time - expected) < 1e-6
