"""3x proprietary Adaptive Gradient-Scaled MPPT (AGS-MPPT) algorithm.

This is the core intellectual property of the company. It addresses the fundamental
weakness of Perturb and Observe, the fixed step size, by adapting the voltage step on every
tracking decision using three complementary signals:

Signal 1, power gradient magnitude: a large local slope dP/dV means the operating point is
far from the peak, on the steep flank of the P-V curve, so large steps are taken to reach
the peak quickly. A small slope means the operating point is near the flat top, so tiny
steps are taken to settle precisely without overshooting.

Signal 2, irradiance change detection: the recent history of extracted power is monitored.
A rapid power change that is not explained by the algorithm's own small voltage moves
indicates that the environment has changed. The controller then enters a brief
re-acquisition mode and searches with large steps for the new peak.

Signal 3, oscillation damping: the recent direction decisions are tracked. Frequent
reversals indicate chattering across a flat peak, so the step size is reduced to settle the
controller and stop wasting energy.

Combined, the controller matches P&O in bright, stable light where P&O already works well,
and it significantly outperforms P&O in dim or variable light where fixed-step oscillation
losses dominate.
"""

from __future__ import annotations

from collections import deque

from .base import MPPTAlgorithm


class AdaptiveGradientScaledMPPT(MPPTAlgorithm):
    """Adaptive Gradient-Scaled MPPT controller.

    Parameters:
        base_step: Base voltage step in volts.
        min_multiplier: Minimum step size multiplier, used near the peak.
        max_multiplier: Maximum step size multiplier, used far from the peak.
        gradient_high_threshold: Normalised gradient magnitude at or above which the
            maximum multiplier is used. The raw gradient is divided by the irradiance ratio
            before comparison, so a single pair of thresholds behaves consistently from
            bright sun to dim indoor light.
        gradient_low_threshold: Normalised gradient magnitude at or below which the minimum
            multiplier is used.
        reference_irradiance: Irradiance used to normalise the gradient, in W/m^2.
        irradiance_shift_window: Number of past steps monitored for irradiance changes.
        irradiance_shift_threshold: Relative power change over the window that triggers
            re-acquisition mode, expressed as a fraction.
        reacquisition_steps: Number of steps to remain in re-acquisition mode after a shift.
        oscillation_buffer_size: Number of recent direction decisions tracked.
        oscillation_threshold: Number of direction changes in the buffer that triggers
            damping.
        damping_factor: Multiplier applied to the step size when oscillation is detected.
        min_step: Absolute minimum step size in volts, preventing the controller stalling.
        max_step: Absolute maximum step size in volts, preventing overshoot in dim light.
        low_light_threshold: Irradiance in W/m^2 below which the controller holds its
            operating voltage steady instead of tracking the negligible available peak.
        gradient_probe: Voltage offset in volts used to estimate the local gradient.
        initial_voltage: Starting voltage in volts.
        v_min: Minimum voltage in volts.
        v_max: Maximum voltage in volts.
    """

    def __init__(
        self,
        base_step: float = 0.008,
        min_multiplier: float = 0.1,
        max_multiplier: float = 3.0,
        gradient_high_threshold: float = 0.1,
        gradient_low_threshold: float = 0.01,
        irradiance_shift_window: int = 10,
        irradiance_shift_threshold: float = 0.15,
        reacquisition_steps: int = 20,
        oscillation_buffer_size: int = 6,
        oscillation_threshold: int = 4,
        damping_factor: float = 0.5,
        min_step: float = 0.0005,
        max_step: float = 0.02,
        low_light_threshold: float = 8.0,
        gradient_probe: float = 0.005,
        reference_irradiance: float = 1000.0,
        initial_voltage: float = 0.4,
        v_min: float = 0.0,
        v_max: float = 0.7,
    ) -> None:
        super().__init__(initial_voltage=initial_voltage, v_min=v_min, v_max=v_max)
        self.base_step = base_step
        self.reference_irradiance = reference_irradiance
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.gradient_high_threshold = gradient_high_threshold
        self.gradient_low_threshold = gradient_low_threshold
        self.irradiance_shift_window = irradiance_shift_window
        self.irradiance_shift_threshold = irradiance_shift_threshold
        self.reacquisition_steps = reacquisition_steps
        self.oscillation_buffer_size = oscillation_buffer_size
        self.oscillation_threshold = oscillation_threshold
        self.damping_factor = damping_factor
        self.min_step = min_step
        self.max_step = max_step
        self.low_light_threshold = low_light_threshold
        self.gradient_probe = gradient_probe

        # Internal tracking state.
        self.direction_buffer: deque[int] = deque(maxlen=oscillation_buffer_size)
        self.power_history_window: deque[float] = deque(maxlen=irradiance_shift_window)
        self.in_reacquisition = False
        self.reacquisition_counter = 0

        # Logging specific to this controller, in addition to the base class logs.
        self.step_size_history: list[float] = []
        self.mode_history: list[str] = []

    def _compute_adaptive_step(self, gradient: float) -> float:
        """Compute the adapted step size from the normalised power gradient magnitude.

        The magnitude is mapped to a multiplier by linear interpolation between the minimum
        and maximum multipliers across the low and high gradient thresholds.

        Parameters:
            gradient: Normalised local power gradient magnitude, dimensionless.

        Returns:
            The adapted step size in volts, before oscillation damping and the floor.
        """
        magnitude = abs(gradient)
        if magnitude >= self.gradient_high_threshold:
            multiplier = self.max_multiplier
        elif magnitude <= self.gradient_low_threshold:
            multiplier = self.min_multiplier
        else:
            span = self.gradient_high_threshold - self.gradient_low_threshold
            fraction = (magnitude - self.gradient_low_threshold) / span
            multiplier = self.min_multiplier + fraction * (
                self.max_multiplier - self.min_multiplier
            )
        return self.base_step * multiplier

    def _detect_irradiance_shift(self, current_power: float) -> bool:
        """Check whether recent power changes indicate an environmental irradiance shift.

        The controller takes small steps near the peak, so a large relative change in power
        across the monitoring window is attributed to the environment rather than to the
        controller's own voltage moves.

        Parameters:
            current_power: The power extracted at the present operating point.

        Returns:
            True if an irradiance shift is detected, otherwise False.
        """
        if len(self.power_history_window) < self.irradiance_shift_window:
            return False
        reference = self.power_history_window[0]
        if reference <= 1e-12:
            return current_power > 1e-9
        relative_change = abs(current_power - reference) / reference
        return relative_change > self.irradiance_shift_threshold

    def _detect_oscillation(self) -> bool:
        """Check the direction buffer for excessive reversals that indicate oscillation.

        Returns:
            True if the number of direction changes in the buffer meets or exceeds the
            oscillation threshold, otherwise False.
        """
        if len(self.direction_buffer) < self.oscillation_buffer_size:
            return False
        reversals = sum(
            1
            for earlier, later in zip(self.direction_buffer, list(self.direction_buffer)[1:])
            if earlier != later
        )
        return reversals >= self.oscillation_threshold

    def _local_gradient(self, voltage: float, power: float, irradiance: float, cell_model) -> float:
        """Estimate the local slope dP/dV using a small forward voltage probe."""
        probe_voltage = self._clamp(voltage + self.gradient_probe)
        if probe_voltage == voltage:
            probe_voltage = self._clamp(voltage - self.gradient_probe)
            if probe_voltage == voltage:
                return 0.0
            probe_power = cell_model.cell_power(probe_voltage, irradiance)
            return (power - probe_power) / (voltage - probe_voltage)
        probe_power = cell_model.cell_power(probe_voltage, irradiance)
        return (probe_power - power) / (probe_voltage - voltage)

    def step(self, irradiance: float, cell_model) -> tuple[float, float]:
        """Execute one tracking step with adaptive gradient scaling.

        Parameters:
            irradiance: Current irradiance in W/m^2.
            cell_model: Module exposing cell_power(V, G).

        Returns:
            A tuple of (operating_voltage, extracted_power) for this time step.
        """
        voltage = self.voltage
        power = cell_model.cell_power(voltage, irradiance)

        # Low-light hold: below the threshold the harvestable power is negligible, so the
        # available peak is not worth chasing. Holding the operating voltage steady keeps
        # the controller poised near the last worthwhile operating point, ready to harvest
        # the instant the light returns rather than having wandered off tracking noise.
        if irradiance < self.low_light_threshold:
            self.prev_power = power
            self.prev_voltage = voltage
            self.voltage_history.append(voltage)
            self.power_history.append(power)
            self.step_size_history.append(0.0)
            self.mode_history.append("hold")
            return voltage, power

        # Signal 2: detect an environmental irradiance shift before updating the window.
        if not self.in_reacquisition and self._detect_irradiance_shift(power):
            self.in_reacquisition = True
            self.reacquisition_counter = self.reacquisition_steps
        self.power_history_window.append(power)

        # Signal 1: gradient-scaled step size. The raw local slope scales with irradiance,
        # so it is normalised by the irradiance ratio to keep the thresholds meaningful
        # across the full range of light levels. Far from the peak the normalised gradient
        # is large and steps are big for a fast approach; near the peak it is small and
        # steps shrink for precise settling. This behaviour alone re-acquires the peak after
        # an irradiance shift, since the shift leaves the operating point far from the new
        # peak on a steep flank.
        gradient = self._local_gradient(voltage, power, irradiance, cell_model)
        irradiance_ratio = max(irradiance / self.reference_irradiance, 0.005)
        normalised_gradient = gradient / irradiance_ratio
        step_size = self._compute_adaptive_step(normalised_gradient)

        # Hill-climbing direction from the observed trajectory.
        delta_power = power - self.prev_power
        delta_voltage = voltage - self.prev_voltage
        if delta_power > 0.0:
            direction = 1 if delta_voltage >= 0.0 else -1
        elif delta_power < 0.0:
            direction = -1 if delta_voltage > 0.0 else 1
        else:
            direction = 1
        self.direction_buffer.append(direction)

        # Signal 3: damp the step if the controller is oscillating and not re-acquiring.
        if self.in_reacquisition:
            mode = "reacquisition"
        elif self._detect_oscillation():
            step_size *= self.damping_factor
            mode = "damping"
        else:
            mode = "normal"

        step_size = min(max(step_size, self.min_step), self.max_step)
        next_voltage = self._clamp(voltage + direction * step_size)
        # Escape a rail rather than pinning against it, mirroring the baseline controller.
        if next_voltage == voltage:
            direction = -direction
            self.direction_buffer[-1] = direction
            next_voltage = self._clamp(voltage + direction * step_size)

        # Advance state.
        self.prev_power = power
        self.prev_voltage = voltage
        self.voltage = next_voltage
        if self.in_reacquisition:
            self.reacquisition_counter -= 1
            if self.reacquisition_counter <= 0:
                self.in_reacquisition = False

        self.voltage_history.append(voltage)
        self.power_history.append(power)
        self.step_size_history.append(step_size)
        self.mode_history.append(mode)
        return voltage, power

    def reset(self) -> None:
        """Reset the controller state and clear all logs for a fresh run."""
        super().reset()
        self.direction_buffer = deque(maxlen=self.oscillation_buffer_size)
        self.power_history_window = deque(maxlen=self.irradiance_shift_window)
        self.in_reacquisition = False
        self.reacquisition_counter = 0
        self.step_size_history = []
        self.mode_history = []
