"""Realistic 24-hour indoor irradiance profile generators.

Each generator returns a tuple of (time_array_seconds, irradiance_array_wm2). The profiles
model the weak, fluctuating light found in commercial interiors: near darkness overnight,
mixed natural and artificial light during working hours, brief occlusion events as people
and vehicles pass sensors, and step changes as lights are switched. All irradiance values
are clamped to be non-negative.

The generators share a small set of private helpers so that noise, step changes, and
occlusion events behave consistently across environments.
"""

from __future__ import annotations

import numpy as np


def _time_axis(duration_hours: float, time_step_seconds: float) -> np.ndarray:
    """Return the array of time stamps in seconds for the requested duration."""
    total_seconds = duration_hours * 3600.0
    num_samples = int(round(total_seconds / time_step_seconds))
    return np.arange(num_samples) * time_step_seconds


def _slow_fluctuation(
    rng: np.random.Generator,
    num_samples: int,
    time_step_seconds: float,
    amplitude: float,
    period_seconds: float = 300.0,
) -> np.ndarray:
    """Generate a smoothly varying random signal by interpolating coarse control points.

    This models the slow drift of indoor light, for example clouds passing a window or
    people moving between zones, without the jitter of per-sample noise.
    """
    control_count = max(int(num_samples * time_step_seconds / period_seconds) + 2, 2)
    control_times = np.linspace(0.0, num_samples * time_step_seconds, control_count)
    control_values = rng.uniform(-amplitude, amplitude, size=control_count)
    sample_times = np.arange(num_samples) * time_step_seconds
    return np.interp(sample_times, control_times, control_values)


def _apply_step_changes(
    irradiance: np.ndarray,
    rng: np.random.Generator,
    time_step_seconds: float,
    events_per_hour: float,
    magnitude_range: tuple[float, float],
    duration_range_seconds: tuple[float, float],
) -> None:
    """Apply sudden step changes in place, modelling lights being switched on or off."""
    num_samples = len(irradiance)
    total_hours = num_samples * time_step_seconds / 3600.0
    num_events = rng.poisson(events_per_hour * total_hours)
    for _ in range(num_events):
        start = rng.integers(0, num_samples)
        duration_seconds = rng.uniform(*duration_range_seconds)
        length = max(int(duration_seconds / time_step_seconds), 1)
        end = min(start + length, num_samples)
        offset = rng.uniform(*magnitude_range) * rng.choice([-1.0, 1.0])
        irradiance[start:end] += offset


def _apply_occlusions(
    irradiance: np.ndarray,
    rng: np.random.Generator,
    time_step_seconds: float,
    active_mask: np.ndarray,
    events_per_hour: float,
    drop_fraction_range: tuple[float, float],
    duration_range_seconds: tuple[float, float],
) -> None:
    """Apply brief occlusion dips in place where a shadow falls across the sensor.

    Occlusions only occur within the active mask, typically the occupied hours of the day.
    """
    num_samples = len(irradiance)
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size == 0:
        return
    active_hours = active_indices.size * time_step_seconds / 3600.0
    num_events = rng.poisson(events_per_hour * active_hours)
    for _ in range(num_events):
        start = int(rng.choice(active_indices))
        duration_seconds = rng.uniform(*duration_range_seconds)
        length = max(int(duration_seconds / time_step_seconds), 1)
        end = min(start + length, num_samples)
        drop = rng.uniform(*drop_fraction_range)
        irradiance[start:end] *= (1.0 - drop)


def generate_indoor_irradiance_profile(
    duration_hours: float = 24.0,
    time_step_seconds: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a realistic 24-hour indoor irradiance profile for a commercial office.

    The profile models a typical commercial office environment across the day: near
    darkness overnight, a morning ramp-up as people arrive and lights come on, sustained
    working-hours light that mixes daylight and artificial lighting, a lunchtime dip, an
    afternoon decline in natural light, an evening wind-down, and dim security lighting at
    night with occasional motion-triggered spikes.

    Parameters:
        duration_hours: Length of simulation in hours.
        time_step_seconds: Time step resolution in seconds.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (time_array_seconds, irradiance_array_wm2).
    """
    rng = np.random.default_rng(seed)
    time_array = _time_axis(duration_hours, time_step_seconds)
    num_samples = len(time_array)
    hours = time_array / 3600.0

    # Baseline light level by hour of day, interpolated between control points.
    control_hours = [0, 6, 6.5, 7, 8, 9, 12, 12.5, 13, 15, 16, 17, 18, 19, 19.5, 21, 24]
    control_levels = [2, 3, 12, 30, 55, 70, 70, 50, 62, 78, 70, 55, 25, 10, 5, 3, 2]
    # Wrap the hour of day so profiles longer than 24 hours repeat the daily pattern.
    hour_of_day = np.mod(hours, 24.0)
    baseline = np.interp(hour_of_day, control_hours, control_levels)

    # Slow working-hours fluctuation of roughly plus or minus 30 W/m^2.
    office_mask = (hour_of_day >= 9.0) & (hour_of_day < 17.0)
    fluctuation = _slow_fluctuation(rng, num_samples, time_step_seconds, amplitude=30.0)
    irradiance = baseline + fluctuation * office_mask

    # Per-sample Gaussian noise for flicker and movement.
    irradiance += rng.normal(0.0, 3.5, size=num_samples)

    # Step changes as lights are switched on or off through the day.
    _apply_step_changes(
        irradiance, rng, time_step_seconds,
        events_per_hour=1.5, magnitude_range=(10.0, 30.0),
        duration_range_seconds=(60.0, 300.0),
    )

    # Occlusion events during occupied hours as people pass the sensor.
    active_mask = (hour_of_day >= 7.0) & (hour_of_day < 19.0)
    _apply_occlusions(
        irradiance, rng, time_step_seconds, active_mask,
        events_per_hour=2.5, drop_fraction_range=(0.30, 0.50),
        duration_range_seconds=(5.0, 20.0),
    )

    # Night-time motion-triggered lighting spikes.
    night_mask = (hour_of_day >= 19.0) | (hour_of_day < 6.0)
    night_indices = np.flatnonzero(night_mask)
    if night_indices.size > 0:
        night_hours = night_indices.size * time_step_seconds / 3600.0
        num_spikes = rng.poisson(0.5 * night_hours)
        for _ in range(num_spikes):
            start = int(rng.choice(night_indices))
            length = max(int(rng.uniform(30.0, 60.0) / time_step_seconds), 1)
            end = min(start + length, num_samples)
            irradiance[start:end] += rng.uniform(20.0, 30.0)

    return time_array, np.clip(irradiance, 0.0, None)


def generate_warehouse_profile(
    duration_hours: float = 24.0,
    time_step_seconds: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a warehouse irradiance profile.

    A warehouse has lower baseline lighting than an office during operating hours, roughly
    20 to 50 W/m^2, with more frequent occlusion events from forklifts and workers and
    longer dark periods outside shift hours.

    Parameters:
        duration_hours: Length of simulation in hours.
        time_step_seconds: Time step resolution in seconds.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (time_array_seconds, irradiance_array_wm2).
    """
    rng = np.random.default_rng(seed)
    time_array = _time_axis(duration_hours, time_step_seconds)
    num_samples = len(time_array)
    hour_of_day = np.mod(time_array / 3600.0, 24.0)

    control_hours = [0, 7, 7.5, 8, 18, 18.5, 19, 24]
    control_levels = [1, 1, 20, 35, 35, 15, 2, 1]
    baseline = np.interp(hour_of_day, control_hours, control_levels)

    operating_mask = (hour_of_day >= 8.0) & (hour_of_day < 18.0)
    fluctuation = _slow_fluctuation(rng, num_samples, time_step_seconds, amplitude=12.0)
    irradiance = baseline + fluctuation * operating_mask
    irradiance += rng.normal(0.0, 2.5, size=num_samples)

    _apply_step_changes(
        irradiance, rng, time_step_seconds,
        events_per_hour=1.0, magnitude_range=(5.0, 20.0),
        duration_range_seconds=(60.0, 240.0),
    )
    _apply_occlusions(
        irradiance, rng, time_step_seconds, operating_mask,
        events_per_hour=8.0, drop_fraction_range=(0.30, 0.60),
        duration_range_seconds=(3.0, 15.0),
    )
    return time_array, np.clip(irradiance, 0.0, None)


def generate_retail_profile(
    duration_hours: float = 24.0,
    time_step_seconds: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a retail store irradiance profile.

    A retail store maintains very consistent artificial lighting during trading hours,
    roughly 60 to 90 W/m^2, with abrupt on and off transitions at opening and closing
    times and minimal natural light variability.

    Parameters:
        duration_hours: Length of simulation in hours.
        time_step_seconds: Time step resolution in seconds.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (time_array_seconds, irradiance_array_wm2).
    """
    rng = np.random.default_rng(seed)
    time_array = _time_axis(duration_hours, time_step_seconds)
    num_samples = len(time_array)
    hour_of_day = np.mod(time_array / 3600.0, 24.0)

    # Sharp opening at 9am and closing at 9pm; steady bright light in between.
    open_mask = (hour_of_day >= 9.0) & (hour_of_day < 21.0)
    irradiance = np.where(open_mask, 75.0, 3.0).astype(float)

    fluctuation = _slow_fluctuation(rng, num_samples, time_step_seconds, amplitude=8.0)
    irradiance += fluctuation * open_mask
    irradiance += rng.normal(0.0, 2.0, size=num_samples)

    _apply_occlusions(
        irradiance, rng, time_step_seconds, open_mask,
        events_per_hour=3.0, drop_fraction_range=(0.20, 0.40),
        duration_range_seconds=(4.0, 12.0),
    )
    return time_array, np.clip(irradiance, 0.0, None)


def generate_constant_low_light(
    irradiance: float = 30.0,
    duration_hours: float = 24.0,
    time_step_seconds: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a constant irradiance profile for controlled testing.

    There is no noise and no variability, which makes this profile useful for isolating
    algorithm performance at a single, precisely known light level.

    Parameters:
        irradiance: The fixed irradiance level in W/m^2.
        duration_hours: Length of simulation in hours.
        time_step_seconds: Time step resolution in seconds.
        seed: Random seed, retained for a consistent interface though it is unused.

    Returns:
        A tuple of (time_array_seconds, irradiance_array_wm2).
    """
    time_array = _time_axis(duration_hours, time_step_seconds)
    irradiance_array = np.full(len(time_array), max(irradiance, 0.0))
    return time_array, irradiance_array


def generate_stress_test_profile(
    duration_hours: float = 2.0,
    time_step_seconds: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a stress-test profile of rapid, extreme irradiance changes.

    The profile cycles between near darkness at 5 W/m^2 and moderate light at 80 W/m^2
    with sharp step transitions every 30 to 120 seconds. It is designed to test how
    quickly a tracking algorithm re-acquires the maximum power point after a shock.

    Parameters:
        duration_hours: Length of simulation in hours.
        time_step_seconds: Time step resolution in seconds.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (time_array_seconds, irradiance_array_wm2).
    """
    rng = np.random.default_rng(seed)
    time_array = _time_axis(duration_hours, time_step_seconds)
    num_samples = len(time_array)
    irradiance = np.empty(num_samples)

    index = 0
    high = True
    while index < num_samples:
        segment_seconds = rng.uniform(30.0, 120.0)
        length = max(int(segment_seconds / time_step_seconds), 1)
        end = min(index + length, num_samples)
        irradiance[index:end] = 80.0 if high else 5.0
        high = not high
        index = end

    irradiance += rng.normal(0.0, 1.5, size=num_samples)
    return time_array, np.clip(irradiance, 0.0, None)
