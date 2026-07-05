# 3x MPPT Engine

Low-light photovoltaic energy harvesting simulation, comparing a standard Perturb and Observe
controller against the proprietary 3x Adaptive Gradient-Scaled MPPT algorithm.

## The Problem

Standard Maximum Power Point Tracking controllers are designed for bright sunlight, where the
power-voltage curve of a solar cell has a sharp, well-defined peak. In low and indoor light the
peak flattens and broadens, and the small fixed voltage steps that work well in the sun begin to
oscillate across the flat top and lag behind every change in the light. For a miniature cell
trying to harvest milliwatts, that wasted energy is significant.

## The Solution

The Adaptive Gradient-Scaled MPPT algorithm, AGS-MPPT, adapts its voltage step on every decision.
It scales the step by the local power gradient so that it strides quickly towards the peak and
then settles precisely on it, detects sudden changes in the light and re-acquires the new peak
rapidly, damps oscillation when it starts to chatter, and parks at a safe voltage in near
darkness so it is poised to harvest the instant the light returns. The result is more energy
captured across the dim, variable conditions that indoor harvesting actually faces.

## Key Results

Energy harvested over a 24-hour profile, AGS-MPPT versus the P&O baseline. Positive figures show
the additional energy the proprietary algorithm captures.

| Profile   | P&O (mWh) | AGS-MPPT (mWh) | Improvement |
|-----------|-----------|----------------|-------------|
| Office    | 27.246    | 27.948         | +2.6%       |
| Warehouse | 11.684    | 12.061         | +3.2%       |
| Retail    | 36.445    | 36.810         | +1.0%       |
| Constant 30 W/m2 | 21.849 | 21.878      | +0.1%       |
| Stress test | 37.648  | 37.965         | +0.8%       |

On the office profile AGS-MPPT reaches 99 per cent of the theoretical maximum energy against
96.7 per cent for P&O, and the improvement is largest in the dimmest bands, for example +5 per
cent between 10 and 30 W/m2 and +4 per cent between 30 and 50 W/m2. In bright, steady light the
two algorithms perform within a fraction of a per cent of each other, as expected.

## Quick Start

```bash
pip install -r requirements.txt
python main.py                       # Run the default office profile over 24 hours
python main.py --profile warehouse   # Use the warehouse profile
python main.py --all-profiles        # Run every profile and compare
python main.py --export results.csv  # Export the full time series to CSV
python main.py --help                # See every option
```

Plots are written to `outputs/plots/`, validation figures to `outputs/validation/`, and any
exported data to the path you provide.

## Dashboard

```bash
streamlit run dashboard.py
```

The dashboard provides a live simulation with real-time plots and parameter sliders, a results
analysis view with the full figure set and a CSV download, a cell model explorer, and a
parameter sensitivity heatmap.

## Project Structure

```
3x-mppt-engine/
├── MASTER_PLAN.md              # Step tracking document
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── .gitignore                  # Standard Python gitignore
├── main.py                     # CLI entry point
├── dashboard.py                # Streamlit dashboard entry point
│
├── src/
│   ├── cell_model.py           # Solar cell physics model
│   ├── irradiance_profiles.py  # Irradiance profile generators
│   ├── algorithms/
│   │   ├── base.py             # Abstract base class for MPPT algorithms
│   │   ├── perturb_observe.py  # Standard P&O implementation
│   │   └── ags_mppt.py         # 3x AGS-MPPT implementation
│   ├── simulation.py           # Simulation engine and results dataclass
│   └── visualisation.py        # All plotting functions
│
├── tests/                      # Unit and integration tests
├── outputs/                    # Generated plots and data
└── docs/
    └── ALGORITHM.md            # Detailed algorithm documentation
```

## Algorithm Details

See [docs/ALGORITHM.md](docs/ALGORITHM.md) for a full description of the cell model, the
irradiance profiles, and both tracking algorithms.

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Author

Clyde Lartey, Founder, 3x.
