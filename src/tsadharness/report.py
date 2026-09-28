"""CSV, figure and Markdown generation for a full harness run.

Everything in ``results/`` is produced by :func:`run_all`. Nothing is written
by hand, so a number quoted in the README or in the paper can always be traced
to a rerun.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless; must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from .controls import artifact_demo, corroborated_demo, onset_stability  # noqa: E402
from .detectors import STUMPY_AVAILABLE, default_detectors  # noqa: E402
from .protocol import (  # noqa: E402
    DEFAULT_MAGNITUDES,
    DEFAULT_REPEATS,
    false_positive_table,
    magnitude_sweep,
    ramp_ceiling_table,
)
from .signals import linear_ramp, step_change  # noqa: E402

Row = dict[str, object]

CONTROL_COLUMNS: Sequence[str] = (
    "control",
    "case",
    "expected",
    "error_a",
    "error_b",
    "ratio",
    "is_artifact",
    "onset_spread",
    "onset_spread_fraction",
    "classification",
    "suppress_lead_time_claim",
    "detail",
)


def write_csv(rows: Sequence[Row], path: str, columns: Sequence[str] | None = None) -> str:
    """Write rows to ``path`` as CSV and return the path.

    Args:
        rows: Records to write. All keys across all rows become columns unless
            ``columns`` is given.
        path: Destination path; parent directories are created.
        columns: Explicit column order.

    Raises:
        ValueError: If ``rows`` is empty and no columns are supplied.
    """
    if not rows and columns is None:
        raise ValueError(f"refusing to write an empty CSV with no schema to {path}")
    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        if value == float("inf"):
            return "inf"
        if abs(value) >= 1000 or (value != 0 and abs(value) < 0.001):
            return f"{value:.3g}"
        return f"{value:.3f}"
    return str(value)


def markdown_table(
    rows: Sequence[Row], columns: Sequence[str], headers: Sequence[str] | None = None
) -> str:
    """Render rows as a GitHub-flavoured Markdown table."""
    head = list(headers) if headers is not None else list(columns)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def _pivot_detection_rate(rows: Sequence[Row], generator: str) -> list[Row]:
    """Return one row per detector with detection rate per magnitude."""
    mags = sorted({float(r["magnitude"]) for r in rows if r["generator"] == generator})
    out: list[Row] = []
    for det in dict.fromkeys(str(r["detector"]) for r in rows):
        row: Row = {"detector": det}
        for mag in mags:
            match = [
                r
                for r in rows
                if r["generator"] == generator
                and str(r["detector"]) == det
                and float(r["magnitude"]) == mag
            ]
            row[f"{mag:g}x"] = float(match[0]["detection_rate"]) if match else float("nan")
        out.append(row)
    return out


def build_control_rows() -> list[Row]:
    """Run both controls on their constructed cases and return CSV rows."""
    rows: list[Row] = []

    art = artifact_demo()
    rows.append(
        {
            "control": "disjoint_window",
            "case": "artifact_demo (healthy two-mode asset, no anomaly exists)",
            "expected": "is_artifact=yes",
            "error_a": art.error_a,
            "error_b": art.error_b,
            "ratio": art.ratio,
            "is_artifact": art.is_artifact,
            "detail": art.verdict,
        }
    )

    ok = corroborated_demo()
    rows.append(
        {
            "control": "disjoint_window",
            "case": "corroborated_demo (real step change)",
            "expected": "is_artifact=no",
            "error_a": ok.error_a,
            "error_b": ok.error_b,
            "ratio": ok.ratio,
            "is_artifact": ok.is_artifact,
            "detail": ok.verdict,
        }
    )

    for label, signal, expected in (
        ("linear_ramp", linear_ramp(n=1000, magnitude=3.0, noise=0.02, seed=0), "ramp_type"),
        ("step_change", step_change(n=1000, magnitude=3.0, noise=0.02, seed=0), "step_type"),
    ):
        stab = onset_stability(signal.values)
        rows.append(
            {
                "control": "onset_stability",
                "case": label,
                "expected": f"classification={expected}",
                "onset_spread": stab.spread,
                "onset_spread_fraction": stab.spread_fraction,
                "classification": stab.classification,
                "suppress_lead_time_claim": stab.suppress_lead_time_claim,
                "detail": stab.reason,
            }
        )
    return rows


def plot_magnitude_sweep(rows: Sequence[Row], path: str) -> str:
    """Draw the headline figure: detection rate vs magnitude, ramp vs step.

    The MADZScore line sits flat at zero across the entire ramp panel while
    MADReference rises to one. That contrast is the result this repository
    exists to show.
    """
    panels = [
        ("linear_ramp", "Ramp (linear_ramp)"),
        ("step_change", "Step (step_change)"),
    ]
    detectors = list(dict.fromkeys(str(r["detector"]) for r in rows))
    styles = {
        "MADZScore": {"color": "#c0392b", "lw": 3.0, "marker": "o", "zorder": 5},
        "MADReference": {"color": "#1f77b4", "lw": 2.5, "marker": "s", "zorder": 4},
        "AEReconstruction": {"color": "#2ca02c", "lw": 1.8, "marker": "^", "zorder": 3},
        "FixedThreshold": {"color": "#ff7f0e", "lw": 1.8, "marker": "d", "zorder": 3},
        "MatrixProfile": {"color": "#9467bd", "lw": 1.8, "marker": "v", "zorder": 3},
        "RandomScore": {"color": "#7f7f7f", "lw": 1.6, "marker": "x", "zorder": 2, "ls": "--"},
    }

    # Several detectors sit at exactly 0.00 or exactly 1.00, which would hide
    # every line but the last one drawn. Each series is nudged vertically by a
    # small fixed amount so that coincident lines stay individually visible.
    # The offset is cosmetic and is stated on the figure.
    n_det = max(len(detectors), 1)
    offsets = {det: (i - (n_det - 1) / 2.0) * 0.018 for i, det in enumerate(detectors)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, (generator, title) in zip(axes, panels):
        for det in detectors:
            pts = sorted(
                (
                    (float(r["magnitude"]), float(r["detection_rate"]))
                    for r in rows
                    if r["generator"] == generator and str(r["detector"]) == det
                ),
                key=lambda p: p[0],
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] + offsets[det] for p in pts]
            style = dict(styles.get(det, {"color": "#333333", "lw": 1.5, "marker": "o"}))
            ls = style.pop("ls", "-")
            ax.plot(xs, ys, ls, label=det, markersize=5, alpha=0.9, **style)
        ax.set_xscale("log")
        ax.set_xlabel("anomaly magnitude (x baseline)")
        ax.set_title(title)
        ax.set_ylim(-0.12, 1.14)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("detection rate")
    axes[0].annotate(
        "MADZScore is flat at zero:\nmax|z| = 1.349 regardless of magnitude,\n"
        "so a 3-sigma threshold is unreachable",
        xy=(0.30, 0.55),
        xycoords="axes fraction",
        fontsize=9.5,
        color="#c0392b",
        va="center",
    )
    axes[1].annotate(
        "same detector, same threshold,\ndifferent geometry: steps are found",
        xy=(0.30, 0.55),
        xycoords="axes fraction",
        fontsize=9.5,
        color="#555555",
        va="center",
    )
    axes[1].annotate(
        "MatrixProfile never fires on a step:\nz-normalised subsequences discard level",
        xy=(0.30, 0.24),
        xycoords="axes fraction",
        fontsize=8.5,
        color="#9467bd",
        va="center",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)
    fig.suptitle(
        "Detection rate vs anomaly magnitude: a self-referencing robust threshold "
        "cannot see a ramp at any size",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.105,
        "Lines are offset vertically by up to +/-0.05 so that coincident series stay "
        "visible; true values are 0.00 or 1.00.",
        ha="center",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_false_positives(rows: Sequence[Row], path: str) -> str:
    """Draw per-detector false-positive rates on the two no-anomaly geometries."""
    detectors = list(dict.fromkeys(str(r["detector"]) for r in rows))
    generators = list(dict.fromkeys(str(r["generator"]) for r in rows))
    width = 0.8 / max(len(generators), 1)
    fig, ax = plt.subplots(figsize=(10, 5.0))
    for i, gen in enumerate(generators):
        vals = []
        for det in detectors:
            match = [r for r in rows if str(r["detector"]) == det and str(r["generator"]) == gen]
            vals.append(float(match[0]["series_false_positive_rate"]) if match else 0.0)
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(detectors))]
        ax.bar(xs, vals, width=width, label=gen)
    ax.set_xticks(range(len(detectors)))
    ax.set_xticklabels(detectors, rotation=20, ha="right")
    ax.set_ylabel("fraction of series with at least one false alarm")
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(
        "False alarms with nothing to find\n"
        "(ground truth is all-False on both geometries; any bar above zero is error)",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_summary(
    sweep: Sequence[Row],
    fps: Sequence[Row],
    controls: Sequence[Row],
    ceiling: Sequence[Row],
) -> str:
    """Assemble the Markdown summary printed to stdout and written to SUMMARY.md."""
    parts: list[str] = []
    parts.append("# tsad-harness results\n")
    parts.append(
        "All numbers below are regenerated by `python -m tsadharness.cli run --out "
        "results/`. Nothing here is hand-entered.\n"
    )
    if not STUMPY_AVAILABLE:
        parts.append(
            "> Note: stumpy is not installed in this environment, so the "
            "`MatrixProfile` detector is absent from these tables.\n"
        )

    parts.append("\n## 1. The ramp ceiling\n")
    parts.append(
        "A self-referencing robust z-score computes its median and MAD from the same "
        "series it scores. On a ramp of range R the deviations from the median are "
        "uniform on [-R/2, +R/2], so the scaled MAD is 0.37065*R while the largest "
        "deviation is 0.5*R. R cancels exactly:\n"
    )
    parts.append(
        markdown_table(
            ceiling,
            [
                "magnitude",
                "ramp_range",
                "max_abs_z",
                "analytic_constant",
                "abs_residual",
                "fires_at_threshold_3",
            ],
            [
                "magnitude",
                "ramp range R",
                "measured max|z|",
                "analytic",
                "residual",
                "fires at 3-sigma?",
            ],
        )
    )
    parts.append(
        "\nThe measured ceiling does not move across three orders of magnitude of R. "
        "A 1000x ramp is exactly as invisible as a 2x ramp.\n"
    )

    parts.append("\n## 2. Detection rate vs magnitude\n")
    parts.append("![magnitude sweep](magnitude_sweep.png)\n")
    for generator, title in (
        ("linear_ramp", "Ramp geometry (`linear_ramp`)"),
        ("step_change", "Step geometry (`step_change`) - the positive control"),
    ):
        pivot = _pivot_detection_rate(sweep, generator)
        cols = list(pivot[0].keys()) if pivot else ["detector"]
        parts.append(f"\n### {title}\n")
        parts.append(markdown_table(pivot, cols))

    parts.append("\n## 3. False positives with nothing to find\n")
    parts.append("![false positive rates](false_positive_rates.png)\n")
    parts.append(
        "Ground truth is all-False on both geometries below, so every non-zero entry "
        "is an error. Read this table before reading any recall number above it.\n"
    )
    parts.append(
        markdown_table(
            fps,
            [
                "detector",
                "generator",
                "series_false_positive_rate",
                "point_false_positive_rate",
                "mean_false_events",
            ],
            ["detector", "geometry", "series FP rate", "point FP rate", "mean false events"],
        )
    )

    parts.append("\n## 4. A detection rate is meaningless on its own\n")
    parts.append(
        "Detection rate asks only whether a detector fired *somewhere* inside the "
        "event. The column beside it asks what fraction of everything it fired on "
        "was healthy data. A detector can score 1.00 on the first while being wrong "
        "almost every time it speaks.\n"
    )
    ramp_rows = [
        r for r in sweep if r["generator"] == "linear_ramp" and float(r["magnitude"]) == 10.0
    ]
    parts.append(
        markdown_table(
            ramp_rows,
            ["detector", "detection_rate", "outside_gt_fire_fraction", "point_precision"],
            [
                "detector",
                "detection rate",
                "fraction of firings on healthy data",
                "point precision",
            ],
        )
    )
    parts.append(
        "\nThe `MatrixProfile` row is the cautionary one. Its detection rate on ramps "
        "reads 1.00, but almost every point it flags lies in the flat baseline before "
        "the ramp begins; it is credited with a detection only because a subsequence "
        "starting in the baseline extends a window's length into the labelled region. "
        "Reported alone, that 1.00 would be a straightforwardly false claim.\n"
    )

    parts.append("\n## 5. Metric inflation: point-adjusted F1\n")
    parts.append(
        "The same predictions scored three ways on the step geometry at 10x. "
        "Point-adjusted F1 credits an entire ground-truth event to any detector that "
        "lands a single point inside it, which is why `RandomScore` is competitive "
        "under it and worthless under the other two.\n"
    )
    infl = [r for r in sweep if r["generator"] == "step_change" and float(r["magnitude"]) == 10.0]
    parts.append(
        markdown_table(
            infl,
            ["detector", "point_f1", "event_f1", "point_adjusted_f1"],
            ["detector", "point F1", "event F1", "point-adjusted F1 (inflated)"],
        )
    )

    parts.append("\n## 6. Controls\n")
    parts.append(
        markdown_table(
            controls,
            [
                "control",
                "case",
                "expected",
                "is_artifact",
                "classification",
                "suppress_lead_time_claim",
                "ratio",
                "onset_spread",
            ],
            [
                "control",
                "case",
                "expected",
                "is_artifact",
                "classification",
                "suppress lead time",
                "A/B ratio",
                "onset spread",
            ],
        )
    )
    parts.append("")
    for row in controls:
        parts.append(f"- **{row.get('case')}** - {row.get('detail')}")
    parts.append("")
    return "\n".join(parts)


def run_all(
    out_dir: str = "results",
    magnitudes: Iterable[float] = DEFAULT_MAGNITUDES,
    repeats: int = DEFAULT_REPEATS,
    n: int = 1000,
    noise: float = 0.02,
    include_matrix_profile: bool = True,
) -> dict[str, object]:
    """Run the full harness and write every artifact into ``out_dir``.

    Args:
        out_dir: Destination directory for CSVs, figures and SUMMARY.md.
        magnitudes: Magnitude grid for the sweep.
        repeats: Seeds per sweep cell.
        n: Series length.
        noise: Noise standard deviation.
        include_matrix_profile: Include the stumpy detector when available.

    Returns:
        A dictionary with the produced rows, the summary text and the written
        file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    detectors = default_detectors(include_matrix_profile=include_matrix_profile)

    sweep = magnitude_sweep(
        detectors=detectors, magnitudes=list(magnitudes), repeats=repeats, n=n, noise=noise
    )
    fps = false_positive_table(detectors=detectors, repeats=max(repeats * 4, 20), n=n, noise=noise)
    controls = build_control_rows()
    ceiling = ramp_ceiling_table()

    paths = {
        "sweep": write_csv(sweep, os.path.join(out_dir, "sweep.csv")),
        "false_positives": write_csv(fps, os.path.join(out_dir, "false_positives.csv")),
        "controls": write_csv(
            controls, os.path.join(out_dir, "controls.csv"), columns=CONTROL_COLUMNS
        ),
        "ramp_ceiling": write_csv(ceiling, os.path.join(out_dir, "ramp_ceiling.csv")),
        "magnitude_sweep_png": plot_magnitude_sweep(
            sweep, os.path.join(out_dir, "magnitude_sweep.png")
        ),
        "false_positive_png": plot_false_positives(
            fps, os.path.join(out_dir, "false_positive_rates.png")
        ),
    }

    summary = build_summary(sweep, fps, controls, ceiling)
    summary_path = os.path.join(out_dir, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary)
    paths["summary"] = summary_path

    return {
        "sweep": sweep,
        "false_positives": fps,
        "controls": controls,
        "ramp_ceiling": ceiling,
        "summary": summary,
        "paths": paths,
    }
