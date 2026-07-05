# Algorithm Documentation

This document describes the physics model, the irradiance profiles, and both tracking
algorithms used in the 3x MPPT engine. British English is used throughout and no em dashes
appear anywhere in the code or documentation.

## 1. Solar Cell Model

The cell is modelled with the single-diode equivalent circuit, the industry-standard physics
model for a photovoltaic cell. The governing equation is:

```
I = I_ph - I_0 * (exp((V + I*R_s) / (n * V_t)) - 1) - (V + I*R_s) / R_sh
```

where I is the output current, V is the terminal voltage, I_ph is the photogenerated current,
I_0 is the reverse saturation current, R_s is the series resistance, R_sh is the shunt
resistance, n is the ideality factor, and V_t is the thermal voltage k * T / q.

The reference parameters model a miniature indoor harvesting cell:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| I_ph_ref  | 0.1 A at 1000 W/m2 | Photocurrent, scaled linearly with irradiance |
| I_0       | 1e-10 A | Reverse saturation current |
| R_s       | 0.5 ohms | Series resistance |
| R_sh      | 500 ohms | Shunt resistance |
| n         | 1.3 | Ideality factor |
| T         | 298.15 K | Cell temperature, 25 degrees Celsius |

The photocurrent scales linearly with irradiance: `I_ph = I_ph_ref * (G / 1000)`.

The equation is implicit in the current, since the current appears on both sides, so it is
solved numerically with a bracketed root finder, `scipy.optimize.brentq`. The exponential
argument is clamped to keep the solver within floating point range at large voltages.

Derived helpers compute the power, the open-circuit voltage, the true maximum power point by a
voltage sweep with a local refinement, and the full I-V and P-V curves.

### Validated behaviour

- At high irradiance the P-V curve has a sharp, well-defined peak.
- At low irradiance the P-V curve is flatter and broader, which is the physical phenomenon the
  company exploits.
- Power output scales roughly linearly with irradiance.
- The open-circuit voltage decreases as irradiance decreases.

## 2. Irradiance Profiles

Five generators produce realistic indoor irradiance series, each returning a time array in
seconds and an irradiance array in W/m2:

- **Office**: near darkness overnight, a morning ramp, sustained working-hours light around 50
  to 100 W/m2 with slow fluctuation, a lunchtime dip, an afternoon decline, an evening
  wind-down, and dim security lighting with occasional motion spikes at night.
- **Warehouse**: a lower operating baseline of 20 to 50 W/m2, more frequent occlusion events
  from forklifts and workers, and longer dark periods.
- **Retail**: very consistent bright lighting during trading hours with abrupt transitions at
  opening and closing.
- **Constant low light**: a fixed level with no noise, useful for controlled tests.
- **Stress test**: rapid cycling between 5 and 80 W/m2 with sharp transitions to test
  responsiveness.

All profiles add Gaussian flicker, occasional step changes, and brief occlusion dips, and every
value is clamped to be non-negative.

## 3. Perturb and Observe (Baseline)

Perturb and Observe perturbs the operating voltage by a fixed step, observes whether the
extracted power rose or fell, and moves in whichever direction increases power. The fixed step
is its defining weakness. In bright light the sharp peak means small steps settle close to the
peak. In dim light the flat, broad peak means the same fixed steps oscillate across the top and
lag behind changes in the light, wasting a meaningful fraction of the already small available
power. If a rail blocks the chosen move the controller reverses direction so it does not pin
itself against the voltage limits.

## 4. Adaptive Gradient-Scaled MPPT

AGS-MPPT addresses the fixed-step weakness with several complementary mechanisms.

### Signal 1: power gradient magnitude

The local slope dP/dV is estimated with a small voltage probe and normalised by the irradiance
ratio. Normalisation matters because the raw gradient scales with irradiance, so a single pair
of thresholds would otherwise only suit one light level. The normalised magnitude is mapped to a
step multiplier by linear interpolation: a large magnitude means the operating point is far from
the peak on a steep flank, so the step is large for a fast approach; a small magnitude means the
operating point is near the flat top, so the step shrinks for precise settling. Because a change
in the light leaves the operating point far from the new peak, this mechanism alone re-acquires
the peak quickly.

### Signal 2: irradiance change detection

The recent history of extracted power is monitored over a sliding window. A large relative change
that the controller's own small steps cannot explain is attributed to the environment. The
controller then enters a brief re-acquisition mode during which oscillation damping is suspended,
so the gradient-scaled steps can move freely to the new peak.

### Signal 3: oscillation damping

The recent direction decisions are tracked in a short buffer. Frequent reversals indicate
chattering across a flat peak, so the step size is halved to settle the controller. A minimum
step floor prevents it from stalling entirely, and a maximum step ceiling prevents overshoot in
dim light.

### Low-light hold

Below a low-light threshold the harvestable power is negligible and the available peak is not
worth chasing. Rather than wandering with the sensor noise, the controller parks at a low, safe
voltage that stays below the open-circuit voltage even in near darkness. A little power is still
captured, and the controller sits close enough to the daytime peak to recover the instant the
light returns. This is the key advantage over the naive baseline, which drifts away chasing the
near-zero dark peak and is then slow to recover at dawn.

### Combined behaviour

- In bright, stable light AGS-MPPT performs within a fraction of a per cent of P&O, since the
  sharp peak gives P&O little to lose.
- In dim, stable light AGS-MPPT settles more tightly than the fixed step, capturing a little
  more.
- In variable light at any level AGS-MPPT recovers faster after each change, which is where most
  of its advantage over a full day comes from.

## 5. Tuned Parameters

The default parameters were tuned so that AGS-MPPT outperforms P&O across every profile while
matching it in bright light:

| Parameter | Value |
|-----------|-------|
| base_step | 0.008 V |
| min_multiplier | 0.1 |
| max_multiplier | 3.0 |
| gradient_high_threshold | 0.10 |
| gradient_low_threshold | 0.01 |
| max_step | 0.02 V |
| min_step | 0.0005 V |
| low_light_threshold | 8.0 W/m2 |
| hold_voltage | 0.26 V |

## 6. Simulation and Metrics

The simulation engine runs both controllers through the same irradiance series, computes the true
maximum power point at each step for the efficiency calculation, and clamps harvested power to be
non-negative to represent the blocking diode of a real harvester. It reports total energy,
tracking efficiency against the true maximum, a breakdown by irradiance band, convergence times
after detected shifts, and direction-reversal counts as a measure of oscillation.
