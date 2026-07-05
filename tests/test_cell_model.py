"""Unit tests for the single-diode solar cell model."""

import cell_model
from cell_model import (
    I_PH_REF,
    cell_current,
    cell_power,
    find_true_mpp,
    open_circuit_voltage,
    photocurrent,
)


def test_cell_current_at_zero_voltage():
    """At V=0 the current should equal approximately the photocurrent (short circuit)."""
    for G in (1000.0, 200.0, 50.0):
        expected = photocurrent(G)
        measured = cell_current(0.0, G)
        assert abs(measured - expected) < 0.05 * expected


def test_cell_current_at_voc():
    """At the open-circuit voltage the current should be approximately zero."""
    for G in (1000.0, 100.0, 30.0):
        v_oc = open_circuit_voltage(G)
        assert abs(cell_current(v_oc, G)) < 1e-6


def test_power_positive_in_operating_range():
    """Power should be positive for voltages between 0 and V_oc at any positive irradiance."""
    for G in (1000.0, 100.0, 30.0):
        v_oc = open_circuit_voltage(G)
        for fraction in (0.25, 0.5, 0.75):
            assert cell_power(fraction * v_oc, G) > 0.0


def test_power_scales_with_irradiance():
    """The MPP power at 500 W/m2 should be roughly half of that at 1000 W/m2."""
    _, _, p_1000 = find_true_mpp(1000.0)
    _, _, p_500 = find_true_mpp(500.0)
    assert 0.4 < p_500 / p_1000 < 0.6


def test_mpp_voltage_decreases_with_irradiance():
    """The MPP voltage should decrease as irradiance decreases."""
    levels = [1000.0, 500.0, 200.0, 100.0, 50.0]
    mpp_voltages = [find_true_mpp(G)[0] for G in levels]
    for higher, lower in zip(mpp_voltages, mpp_voltages[1:]):
        assert higher > lower


def test_zero_irradiance_gives_zero_power():
    """At G=0 all power outputs should be zero or negligible."""
    for V in (0.0, 0.2, 0.4, 0.6):
        assert abs(cell_power(V, 0.0)) < 1e-9
    assert I_PH_REF > 0.0  # sanity check that the reference constant is present
    assert cell_model.find_true_mpp(0.0) == (0.0, 0.0, 0.0)
