"""Standard Perturb and Observe (P&O) maximum power point tracker.

This is the industry-standard algorithm used in virtually all commercial MPPT controllers.
It perturbs the operating voltage by a fixed step, observes whether the extracted power
rose or fell, and moves in whichever direction increases power. It is intentionally simple
and serves as the baseline that the proprietary controller must beat.

The key weakness is the fixed step size. In bright light the P-V peak is sharp, so small
fixed steps settle close to the peak. In dim light the peak is flat and broad, so the same
fixed steps overshoot the peak repeatedly in both directions, and each overshoot wastes a
meaningful fraction of the already tiny available power.
"""

from __future__ import annotations

from .base import MPPTAlgorithm


class PerturbAndObserve(MPPTAlgorithm):
    """Standard Perturb and Observe MPPT algorithm with a fixed voltage step.

    Parameters:
        step_size: Fixed voltage perturbation step in volts.
        initial_voltage: Starting operating voltage in volts.
        v_min: Minimum allowed voltage in volts.
        v_max: Maximum allowed voltage in volts, a typical open-circuit voltage for a
            silicon cell under moderate light.
    """

    def __init__(
        self,
        step_size: float = 0.01,
        initial_voltage: float = 0.4,
        v_min: float = 0.0,
        v_max: float = 0.7,
    ) -> None:
        super().__init__(initial_voltage=initial_voltage, v_min=v_min, v_max=v_max)
        self.step_size = step_size

    def step(self, irradiance: float, cell_model) -> tuple[float, float]:
        """Execute one fixed-step tracking decision.

        Parameters:
            irradiance: Current irradiance in W/m^2.
            cell_model: Module exposing cell_power(V, G).

        Returns:
            A tuple of (operating_voltage, extracted_power) for this time step.
        """
        voltage = self.voltage
        power = cell_model.cell_power(voltage, irradiance)

        delta_power = power - self.prev_power
        delta_voltage = voltage - self.prev_voltage

        # Standard four-quadrant decision logic. On a tie in voltage the controller keeps
        # nudging upward so that it never stalls at the very start of a run.
        if delta_power > 0.0:
            direction = 1.0 if delta_voltage >= 0.0 else -1.0
        elif delta_power < 0.0:
            direction = -1.0 if delta_voltage > 0.0 else 1.0
        else:
            direction = 1.0

        next_voltage = self._clamp(voltage + direction * self.step_size)
        # If a rail blocks the chosen move the voltage cannot change, which would pin the
        # controller against the rail. Reverse direction so it escapes back towards the peak.
        if next_voltage == voltage:
            direction = -direction
            next_voltage = self._clamp(voltage + direction * self.step_size)

        # Store the present operating point as the reference for the next comparison.
        self.prev_power = power
        self.prev_voltage = voltage
        self.voltage = next_voltage

        self.voltage_history.append(voltage)
        self.power_history.append(power)
        return voltage, power
