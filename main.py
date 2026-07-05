"""Command-line entry point for the 3x MPPT simulation engine.

This script runs the Perturb and Observe baseline against the Adaptive Gradient-Scaled
controller over a chosen irradiance profile, prints a results summary, and optionally
exports the time series and generates the full set of plots. It is the headless counterpart
to the Streamlit dashboard.

Usage examples:
    python main.py                       Run the default office profile over 24 hours.
    python main.py --profile warehouse   Use the warehouse profile.
    python main.py --profile retail      Use the retail profile.
    python main.py --duration 48         Run for 48 hours.
    python main.py --step-size 0.5       Use half-second time steps.
    python main.py --no-plots            Skip plot generation.
    python main.py --live-plot           Show a live matplotlib animation.
    python main.py --export results.csv  Export the results to CSV.
    python main.py --all-profiles        Run every profile and compare.
    python main.py --quiet               Suppress console output.
    python main.py --seed 123            Set the random seed.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cell_model
import irradiance_profiles as profiles
from algorithms.ags_mppt import AdaptiveGradientScaledMPPT
from algorithms.perturb_observe import PerturbAndObserve
from simulation import SimulationEngine
from visualisation import Visualiser


def build_profile(name: str, duration_hours: float, time_step: float, seed: int, irradiance: float):
    """Return a (time_array, irradiance_array) tuple for the named profile."""
    if name == "office":
        return profiles.generate_indoor_irradiance_profile(duration_hours, time_step, seed)
    if name == "warehouse":
        return profiles.generate_warehouse_profile(duration_hours, time_step, seed)
    if name == "retail":
        return profiles.generate_retail_profile(duration_hours, time_step, seed)
    if name == "constant":
        return profiles.generate_constant_low_light(irradiance, duration_hours, time_step, seed)
    if name == "stress":
        return profiles.generate_stress_test_profile(duration_hours, time_step, seed)
    raise ValueError("Unknown profile: %s" % name)


def make_engine() -> SimulationEngine:
    """Construct a simulation engine with fresh controllers."""
    return SimulationEngine(cell_model, PerturbAndObserve(), AdaptiveGradientScaledMPPT())


def run_live_plot(results) -> None:
    """Show a simple live animation of the extracted power building up over time."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames = 400
    stride = max(len(results.time_seconds) // frames, 1)
    hours = results.time_seconds[::stride] / 3600.0
    po = results.po_power[::stride] * 1000.0
    ags = results.ags_power[::stride] * 1000.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(hours.min(), hours.max())
    ax.set_ylim(0, max(ags.max(), po.max()) * 1.1 + 1e-6)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Power (mW)")
    ax.set_title("Live power tracking: AGS-MPPT versus P&O")
    (po_line,) = ax.plot([], [], color="#E63946", label="P&O")
    (ags_line,) = ax.plot([], [], color="#2A9D8F", label="AGS-MPPT")
    ax.legend(loc="upper right")

    def update(frame):
        po_line.set_data(hours[:frame], po[:frame])
        ags_line.set_data(hours[:frame], ags[:frame])
        return po_line, ags_line

    FuncAnimation(fig, update, frames=len(hours), interval=20, blit=True, repeat=False)
    plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the 3x low-light MPPT simulation and compare AGS-MPPT with P&O."
    )
    parser.add_argument(
        "--profile", default="office",
        choices=["office", "warehouse", "retail", "constant", "stress"],
        help="Irradiance profile to simulate. Defaults to the office profile.",
    )
    parser.add_argument(
        "--duration", type=float, default=24.0,
        help="Simulation duration in hours. Defaults to 24.",
    )
    parser.add_argument(
        "--step-size", type=float, default=1.0, dest="time_step",
        help="Time step resolution in seconds. Defaults to 1 second.",
    )
    parser.add_argument(
        "--irradiance", type=float, default=30.0,
        help="Irradiance in W/m2 for the constant profile. Defaults to 30.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generation of the result plots.",
    )
    parser.add_argument(
        "--live-plot", action="store_true",
        help="Show a live matplotlib animation of the tracked power.",
    )
    parser.add_argument(
        "--export", default=None, metavar="PATH",
        help="Export the full time-series results to this CSV file path.",
    )
    parser.add_argument(
        "--all-profiles", action="store_true",
        help="Run every profile in turn and print a comparison table.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the per-step and summary console output.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible profiles. Defaults to 42.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the simulation according to the parsed command-line arguments."""
    args = parse_args(argv)
    verbose = not args.quiet

    if args.all_profiles:
        names = ["office", "warehouse", "retail", "constant", "stress"]
        built = {
            name: build_profile(name, args.duration, args.time_step, args.seed, args.irradiance)
            for name in names
        }
        engine = make_engine()
        outcomes = engine.run_batch(built, verbose=verbose, log_interval=10000)
        print("\nPROFILE COMPARISON")
        print("-" * 60)
        print("%-12s %12s %12s %12s" % ("Profile", "P&O (mWh)", "AGS (mWh)", "Improvement"))
        for name, result in outcomes.items():
            print(
                "%-12s %12.3f %12.3f %+11.1f%%"
                % (
                    name,
                    result.po_total_energy_wh * 1000.0,
                    result.ags_total_energy_wh * 1000.0,
                    result.improvement_percent,
                )
            )
        return 0

    time_array, irradiance_array = build_profile(
        args.profile, args.duration, args.time_step, args.seed, args.irradiance
    )
    engine = make_engine()
    if verbose:
        print("Running %s profile over %.1f hours at %gs steps."
              % (args.profile, args.duration, args.time_step))
    results = engine.run(time_array, irradiance_array, verbose=verbose)

    if args.export:
        os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
        results.to_csv(args.export)
        if verbose:
            print("Exported results to %s" % args.export)

    if not args.no_plots:
        Visualiser().generate_all_plots(results, cell_model=cell_model)
        if verbose:
            print("Saved plots to outputs/plots/")

    if args.live_plot:
        run_live_plot(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
