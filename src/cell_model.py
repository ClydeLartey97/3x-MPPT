"""Single-diode equivalent circuit model for a miniature photovoltaic cell.

This module implements the industry-standard single-diode model of a solar cell and
provides helper functions to compute current, power, the true maximum power point, and
full I-V and P-V curves across a wide range of irradiance levels. The governing equation
is implicit in the current, but it has an exact closed-form solution in terms of the
Lambert W function, which this module uses. The explicit solution is both faster than an
iterative root solve and vectorises naturally across arrays of voltages.

All quantities use SI units unless stated otherwise: volts, amps, watts, ohms, kelvin, and
watts per square metre for irradiance.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

# Physical constants.
BOLTZMANN = 1.381e-23           # Boltzmann constant (J/K).
ELECTRON_CHARGE = 1.602e-19     # Elementary charge (C).

# Reference cell parameters for a miniature indoor harvesting cell.
I_PH_REF = 0.1      # Photogenerated current at the reference irradiance (A).
G_REF = 1000.0      # Reference irradiance, standard test conditions (W/m^2).
I_0 = 1e-10         # Reverse saturation current (A).
R_S = 0.5           # Series resistance (ohms).
R_SH = 500.0        # Shunt resistance (ohms).
N_IDEALITY = 1.3    # Diode ideality factor (dimensionless).
T_REF = 298.15      # Reference cell temperature, 25 degrees Celsius (K).

# Numerical guard to keep the exponential term within floating point range.
_MAX_EXP_ARGUMENT = 700.0


def thermal_voltage(T: float = T_REF) -> float:
    """Return the thermal voltage V_t = k * T / q for a given temperature.

    Parameters:
        T: Cell temperature (K).

    Returns:
        Thermal voltage in volts. At 298.15 K this is approximately 0.02569 V.
    """
    return BOLTZMANN * T / ELECTRON_CHARGE


def photocurrent(G: float) -> float:
    """Return the photogenerated current, scaled linearly with irradiance.

    Parameters:
        G: Irradiance (W/m^2).

    Returns:
        Photogenerated current in amps. Zero for non-positive irradiance.
    """
    if G <= 0.0:
        return 0.0
    return I_PH_REF * (G / G_REF)


def _cell_current_array(V: np.ndarray, G: float, T: float) -> np.ndarray:
    """Vectorised current calculation using the explicit Lambert W solution.

    The single-diode equation
        I = I_ph - I_0 * (exp((V + I*R_s) / (n * V_t)) - 1) - (V + I*R_s) / R_sh
    is implicit in I, but rearranges exactly to
        I = (R_sh * (I_ph + I_0) - V) / (R_s + R_sh) - (a / R_s) * W(z)
    with a = n * V_t and
        z = (R_s * R_sh * I_0) / (a * (R_s + R_sh))
            * exp(R_sh * (R_s * (I_ph + I_0) + V) / (a * (R_s + R_sh)))
    where W is the principal branch of the Lambert W function. This is the standard
    closed-form solution for the single-diode model.

    Parameters:
        V: Array of terminal voltages (V).
        G: Irradiance (W/m^2).
        T: Cell temperature (K).

    Returns:
        Array of currents in amps.
    """
    if G <= 0.0:
        return np.zeros_like(V)

    i_ph = photocurrent(G)
    a = N_IDEALITY * thermal_voltage(T)
    r_total = R_S + R_SH

    exponent = R_SH * (R_S * (i_ph + I_0) + V) / (a * r_total)
    # The exponent stays far below this guard for any physical voltage; the clamp only
    # protects against overflow if a caller probes absurdly large voltages.
    exponent = np.minimum(exponent, _MAX_EXP_ARGUMENT)
    z = (R_S * R_SH * I_0) / (a * r_total) * np.exp(exponent)

    current = (R_SH * (i_ph + I_0) - V) / r_total - (a / R_S) * lambertw(z).real
    return current


def cell_current(V: float, G: float, T: float = T_REF) -> float:
    """Calculate the output current of the solar cell at a given voltage and irradiance.

    The implicit single-diode equation is evaluated through its exact Lambert W
    closed-form solution, so no iterative root finding is needed.

    Parameters:
        V: Terminal voltage (V).
        G: Irradiance (W/m^2).
        T: Cell temperature (K), default 25 degrees Celsius.

    Returns:
        Current in amps.
    """
    return float(_cell_current_array(np.asarray(V, dtype=float), G, T))


def cell_power(V: float, G: float, T: float = T_REF) -> float:
    """Calculate the output power (V times I) at a given voltage and irradiance.

    Parameters:
        V: Terminal voltage (V).
        G: Irradiance (W/m^2).
        T: Cell temperature (K).

    Returns:
        Power in watts.
    """
    return V * cell_current(V, G, T)


def open_circuit_voltage(G: float, T: float = T_REF) -> float:
    """Find the open-circuit voltage, the voltage at which the current falls to zero.

    Parameters:
        G: Irradiance (W/m^2).
        T: Cell temperature (K).

    Returns:
        Open-circuit voltage in volts. Zero for non-positive irradiance.
    """
    if G <= 0.0:
        return 0.0
    # Current is positive at V = 0 and negative at a modest upper bound, so a root exists.
    return float(brentq(lambda v: cell_current(v, G, T), 1e-9, 1.5, xtol=1e-9))


def find_true_mpp(G: float, T: float = T_REF) -> tuple[float, float, float]:
    """Find the true maximum power point for a given irradiance.

    The voltage is swept from zero to the open-circuit voltage in a single vectorised
    pass and the point of greatest power is selected, followed by a local refinement
    sweep around the coarse peak.

    Parameters:
        G: Irradiance (W/m^2).
        T: Cell temperature (K).

    Returns:
        A tuple of (V_mpp, I_mpp, P_mpp).
    """
    if G <= 0.0:
        return 0.0, 0.0, 0.0

    v_oc = open_circuit_voltage(G, T)
    if v_oc <= 0.0:
        return 0.0, 0.0, 0.0

    voltages = np.linspace(0.0, v_oc, 400)
    powers = voltages * _cell_current_array(voltages, G, T)
    coarse_index = int(np.argmax(powers))

    # Refine locally around the coarse peak for a tighter estimate.
    low = voltages[max(coarse_index - 1, 0)]
    high = voltages[min(coarse_index + 1, len(voltages) - 1)]
    fine_voltages = np.linspace(low, high, 100)
    fine_powers = fine_voltages * _cell_current_array(fine_voltages, G, T)
    fine_index = int(np.argmax(fine_powers))

    v_mpp = float(fine_voltages[fine_index])
    p_mpp = float(fine_powers[fine_index])
    i_mpp = cell_current(v_mpp, G, T)
    return v_mpp, i_mpp, p_mpp


def generate_iv_curve(
    G: float, T: float = T_REF, num_points: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the full I-V and P-V curves for plotting.

    Parameters:
        G: Irradiance (W/m^2).
        T: Cell temperature (K).
        num_points: Number of voltage samples between zero and the open-circuit voltage.

    Returns:
        A tuple of (voltages, currents, powers) arrays.
    """
    if G <= 0.0:
        voltages = np.linspace(0.0, 0.7, num_points)
        return voltages, np.zeros(num_points), np.zeros(num_points)

    v_oc = open_circuit_voltage(G, T)
    voltages = np.linspace(0.0, v_oc, num_points)
    currents = _cell_current_array(voltages, G, T)
    powers = voltages * currents
    return voltages, currents, powers
