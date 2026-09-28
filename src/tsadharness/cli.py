"""Command line entry point.

``python -m tsadharness.cli run --out results/`` regenerates every CSV, figure
and Markdown table in this repository from scratch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .detectors import RAMP_BLINDNESS_CONSTANT, STUMPY_AVAILABLE, available_detector_names
from .protocol import DEFAULT_MAGNITUDES, DEFAULT_REPEATS
from .report import run_all
from .signals import GENERATORS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsadharness",
        description=(
            "Test harness for time-series anomaly detectors. Generates signals with "
            "known ground truth, reports which failure geometries each detector "
            "cannot see, and flags findings that are artifacts of a training window."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full sweep and write results")
    run.add_argument("--out", default="results", help="output directory (default: results)")
    run.add_argument("--n", type=int, default=1000, help="series length (default: 1000)")
    run.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"seeds per sweep cell (default: {DEFAULT_REPEATS})",
    )
    run.add_argument(
        "--noise", type=float, default=0.02, help="noise standard deviation (default: 0.02)"
    )
    run.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=list(DEFAULT_MAGNITUDES),
        help="magnitude grid",
    )
    run.add_argument(
        "--no-matrix-profile",
        action="store_true",
        help="skip the stumpy-backed detector even when stumpy is installed",
    )
    run.add_argument("--quiet", action="store_true", help="do not print the summary")

    sub.add_parser("list", help="list available generators and detectors")
    sub.add_parser("constant", help="print the analytic ramp-blindness constant")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector; ``sys.argv[1:]`` when None.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.command == "constant":
        print(f"{RAMP_BLINDNESS_CONSTANT:.10f}")
        print(
            "max|z| for a self-referencing robust z-score on a uniform ramp, "
            "for every ramp magnitude and every baseline fraction <= 1/4."
        )
        return 0

    if args.command == "list":
        print("generators:")
        for name in GENERATORS:
            print(f"  {name}")
        print("\ndetectors:")
        for name in available_detector_names():
            print(f"  {name}")
        if not STUMPY_AVAILABLE:
            print("\n  (MatrixProfile unavailable: stumpy is not installed)")
        return 0

    if args.command == "run":
        magnitudes: list[float] = [float(m) for m in args.magnitudes]
        result = run_all(
            out_dir=args.out,
            magnitudes=magnitudes,
            repeats=int(args.repeats),
            n=int(args.n),
            noise=float(args.noise),
            include_matrix_profile=not args.no_matrix_profile,
        )
        if not args.quiet:
            print(result["summary"])
        paths = result["paths"]
        print("\nwrote:", file=sys.stderr)
        for key in sorted(paths):
            print(f"  {paths[key]}", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
