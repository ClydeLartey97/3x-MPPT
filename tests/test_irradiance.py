"""Unit tests for the irradiance profile generators."""

import numpy as np

import irradiance_profiles as profiles


def test_office_profile_length():
    """24 hours at 1 second steps should produce 86400 data points."""
    time_array, irradiance = profiles.generate_indoor_irradiance_profile(24.0, 1.0)
    assert len(time_array) == 86400
    assert len(irradiance) == 86400


def test_irradiance_never_negative():
    """All irradiance values must be greater than or equal to zero for every profile."""
    generators = [
        profiles.generate_indoor_irradiance_profile(),
        profiles.generate_warehouse_profile(),
        profiles.generate_retail_profile(),
        profiles.generate_constant_low_light(),
        profiles.generate_stress_test_profile(),
    ]
    for _, irradiance in generators:
        assert np.all(irradiance >= 0.0)


def test_profile_reproducibility():
    """The same seed should produce identical profiles."""
    _, first = profiles.generate_indoor_irradiance_profile(seed=7)
    _, second = profiles.generate_indoor_irradiance_profile(seed=7)
    assert np.array_equal(first, second)


def test_office_profile_has_dark_period():
    """The night hours should have near-zero irradiance."""
    time_array, irradiance = profiles.generate_indoor_irradiance_profile()
    hour_of_day = np.mod(time_array / 3600.0, 24.0)
    night = irradiance[hour_of_day < 5.0]
    assert night.mean() < 10.0


def test_office_profile_has_peak_period():
    """The daytime hours should have irradiance above 30 W/m2 on average."""
    time_array, irradiance = profiles.generate_indoor_irradiance_profile()
    hour_of_day = np.mod(time_array / 3600.0, 24.0)
    daytime = irradiance[(hour_of_day >= 9.0) & (hour_of_day < 17.0)]
    assert daytime.mean() > 30.0
