"""Fit a function from ``pwlopt.fitting.functions`` with adaptive triangulation.

Example:
    python scripts/fit_function.py fun2 --eps 0.01 --min-angle 20 \
        --output results/fitting/fun2.txt
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyintval import Interval

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pwlopt.fitting import functions
from pwlopt.fitting.fitting import PWLApproximation
from pwlopt.fitting.interval_bb import Box2D, IntervalBB
from pwlopt.fitting.triangulation import Triangulation
from pwlopt.utils import write_trg


def resolve_function(name: str):
    function = getattr(functions, name, None)
    gradient = getattr(functions, f"iv{name}_gradnorm", None)
    if not callable(function) or not callable(gradient):
        available = sorted(
            value
            for value in vars(functions)
            if value.startswith("fun") and callable(getattr(functions, value))
        )
        raise ValueError(  # noqa: TRY004
            f"Function {name!r} must have a matching iv{name}_gradnorm; "
            f"available functions: {', '.join(available)}"
        )
    return function, gradient


def estimate_lipschitz(gradient, ftol: float, dtol: float) -> float:
    """Estimate a safe Lipschitz constant from an interval gradient bound."""
    box = Box2D(Interval(0.0, 1.0), Interval(0.0, 1.0))

    def interval_gradient(intervals):
        try:
            return gradient(*intervals)
        except TypeError:
            return gradient(intervals)

    _, upper_bound = IntervalBB(interval_gradient, box).maximize(ftol, dtol)
    if not np.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError(f"Could not estimate a positive Lipschitz constant: {upper_bound}")
    return float(upper_bound)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("function", help="Function name from pwlopt.fitting.functions, for example fun2")
    parser.add_argument("--eps", type=float, required=True, help="Required maximum fitting error")
    parser.add_argument("--min-angle", type=float, required=True, help="Minimum triangle angle in degrees")
    parser.add_argument("--output", type=Path, help="Write the resulting triangulation to this file")
    parser.add_argument("--plot", type=Path, help="Write function, triangulation, and error-history plots to this file")
    parser.add_argument("--lipschitz-ftol", type=float, default=0.01, help="Interval bound tolerance")
    parser.add_argument("--lipschitz-dtol", type=float, default=0.001, help="Smallest interval diameter to split")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    args = parser.parse_args()

    if args.eps <= 0:
        parser.error("--eps must be positive")
    if args.min_angle <= 0 or args.min_angle >= 60:
        parser.error("--min-angle must be between 0 and 60 degrees")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    function, gradient = resolve_function(args.function)
    lipschitz = estimate_lipschitz(gradient, args.lipschitz_ftol, args.lipschitz_dtol)
    print(f"Estimated Lipschitz constant: {lipschitz:.8g}")

    initial_points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    triangulation = Triangulation.from_points(initial_points)
    approximation = PWLApproximation(triangulation, function, lipschitz, args.min_angle)
    approximation.approximate(args.eps)

    print(
        f"Finished with {len(triangulation.pts)} points, "
        f"{len(triangulation.trg)} triangles, "
        f"estimated error {approximation.global_max_err:.8g}"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_trg(
            [triangle.idxs for triangle in triangulation.trg],
            triangulation.pts,
            args.output,
        )
        print(f"Wrote triangulation to {args.output}")
    if args.plot:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        approximation.plot_function(figure, axes[0])
        triangulation.plot_triangulation(figure, axes[0])
        figure.suptitle(f"PWL fitting: {args.function}")
        axes[0].set_title(
            f"Function and final triangulation\n"
            f"minimum angle = {np.degrees(triangulation.min_angle):.2f} degrees"
        )

        error_history = approximation.err_hist #or [approximation.global_max_err]
        axes[1].plot([2*err for err in error_history], color="red", linewidth=2, label="Guaranteed maximal error")
        axes[1].plot([1*err for err in error_history], color="red", linestyle="-.",linewidth=.5, label="Measured maximal error")
        axes[1].grid()

        axes[1].axhline(args.eps, color="black", linestyle="--", label="Required error")
        axes[1].set_title(
            f"Error history\n"
            f"required eps = {args.eps:.6g}"
        )
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Interpolation error")
        axes[1].legend()

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=150)
        plt.close(figure)
        print(f"Wrote plot to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())