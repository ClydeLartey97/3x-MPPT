"""Unit tests for the P&O and AGS-MPPT tracking algorithms."""

import numpy as np

import cell_model
from algorithms.ags_mppt import AdaptiveGradientScaledMPPT
from algorithms.perturb_observe import PerturbAndObserve
from cell_model import find_true_mpp


def _steps_to_converge(algorithm, irradiance, max_steps, fraction=0.95):
    """Return the number of steps for a controller to reach the target share of the MPP."""
    algorithm.reset()
    _, _, p_mpp = find_true_mpp(irradiance)
    for i in range(max_steps):
        _, power = algorithm.step(irradiance, cell_model)
        if p_mpp <= 1e-12 or power >= fraction * p_mpp:
            return i + 1
    return max_steps + 1


def _total_energy(algorithm, irradiance, num_steps):
    """Return the total non-negative power a controller extracts over a constant run."""
    algorithm.reset()
    return sum(max(algorithm.step(irradiance, cell_model)[1], 0.0) for _ in range(num_steps))


def test_po_converges_in_bright_light():
    """P&O should reach within 95 per cent of the true MPP within 50 steps at 1000 W/m2."""
    assert _steps_to_converge(PerturbAndObserve(), 1000.0, 50) <= 50


def test_ags_converges_in_bright_light():
    """AGS should reach within 95 per cent of the true MPP within 30 steps at 1000 W/m2."""
    assert _steps_to_converge(AdaptiveGradientScaledMPPT(), 1000.0, 30) <= 30


def test_ags_converges_in_dim_light():
    """AGS should reach within 95 per cent of the true MPP within 50 steps at 30 W/m2."""
    assert _steps_to_converge(AdaptiveGradientScaledMPPT(), 30.0, 50) <= 50


def test_ags_outperforms_po_in_dim_light():
    """Over 1000 steps at 30 W/m2 AGS should extract at least as much energy as P&O."""
    po = _total_energy(PerturbAndObserve(), 30.0, 1000)
    ags = _total_energy(AdaptiveGradientScaledMPPT(), 30.0, 1000)
    assert ags >= po


def test_ags_reacquisition_on_irradiance_shift():
    """After a sudden irradiance change AGS should re-converge no slower than P&O."""
    def reacquire(algorithm):
        algorithm.reset()
        for _ in range(200):
            algorithm.step(30.0, cell_model)
        _, _, p_mpp = find_true_mpp(150.0)
        for i in range(100):
            _, power = algorithm.step(150.0, cell_model)
            if power >= 0.95 * p_mpp:
                return i + 1
        return 101

    assert reacquire(AdaptiveGradientScaledMPPT()) <= reacquire(PerturbAndObserve())


def test_ags_matches_po_in_bright_light():
    """In bright steady light AGS should perform within 2 per cent of P&O, and not worse."""
    po = _total_energy(PerturbAndObserve(), 1000.0, 2000)
    ags = _total_energy(AdaptiveGradientScaledMPPT(), 1000.0, 2000)
    assert ags >= 0.98 * po


def test_algorithms_handle_zero_irradiance():
    """Both controllers should handle G=0 gracefully without errors."""
    for algorithm in (PerturbAndObserve(), AdaptiveGradientScaledMPPT()):
        for _ in range(50):
            voltage, power = algorithm.step(0.0, cell_model)
            assert np.isfinite(voltage) and np.isfinite(power)


def test_algorithms_handle_very_high_irradiance():
    """Both controllers should handle G=1200 W/m2 without errors."""
    for algorithm in (PerturbAndObserve(), AdaptiveGradientScaledMPPT()):
        for _ in range(50):
            voltage, power = algorithm.step(1200.0, cell_model)
            assert np.isfinite(voltage) and np.isfinite(power)
