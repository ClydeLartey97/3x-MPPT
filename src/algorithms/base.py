"""Abstract base class shared by the MPPT tracking algorithms.

Both the standard Perturb and Observe controller and the proprietary Adaptive
Gradient-Scaled controller expose the same interface: a single tracking step that consumes
the current irradiance and returns the chosen operating voltage together with the power
extracted at that voltage. A common base class keeps the logging structure identical, which
makes the head-to-head comparison in the simulation engine fair.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MPPTAlgorithm(ABC):
    """Common interface and logging state for a maximum power point tracker.

    Subclasses must implement the tracking logic in :meth:`step`. The base class provides
    the operating voltage bounds and the per-step logging lists that the simulation engine
    reads after a run.

    Parameters:
        initial_voltage: Starting operating voltage in volts.
        v_min: Minimum allowed operating voltage in volts.
        v_max: Maximum allowed operating voltage in volts.
    """

    def __init__(
        self,
        initial_voltage: float = 0.4,
        v_min: float = 0.0,
        v_max: float = 0.7,
    ) -> None:
        self.initial_voltage = initial_voltage
        self.v_min = v_min
        self.v_max = v_max

        self.voltage = initial_voltage
        self.prev_power = 0.0
        self.prev_voltage = initial_voltage

        # Logging lists, populated on every call to step.
        self.voltage_history: list[float] = []
        self.power_history: list[float] = []
        self.efficiency_history: list[float] = []

    def _clamp(self, voltage: float) -> float:
        """Clamp a candidate voltage to the permitted operating range."""
        return min(max(voltage, self.v_min), self.v_max)

    @abstractmethod
    def step(self, irradiance: float, cell_model) -> tuple[float, float]:
        """Execute one tracking step.

        Parameters:
            irradiance: Current irradiance in W/m^2.
            cell_model: Module or object exposing cell_current(V, G) and cell_power(V, G).

        Returns:
            A tuple of (operating_voltage, extracted_power) for this time step.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the algorithm state and clear all logs for a fresh run."""
        self.voltage = self.initial_voltage
        self.prev_power = 0.0
        self.prev_voltage = self.initial_voltage
        self.voltage_history = []
        self.power_history = []
        self.efficiency_history = []
