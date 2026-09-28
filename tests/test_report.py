"""Tests for report generation and the CLI."""

from __future__ import annotations

import csv
import os

import pytest

from tsadharness.cli import main
from tsadharness.report import build_control_rows, markdown_table, run_all, write_csv

FAST = {"magnitudes": [2.0, 10.0], "repeats": 2, "n": 400}


def test_write_csv_round_trips(tmp_path) -> None:
    """Rows written are rows read back."""
    path = str(tmp_path / "x.csv")
    write_csv([{"a": 1, "b": "two"}, {"a": 3, "b": "four"}], path)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"a": "1", "b": "two"}, {"a": "3", "b": "four"}]


def test_write_csv_unions_keys_across_rows(tmp_path) -> None:
    """Ragged rows produce a union header with blanks, not a crash."""
    path = str(tmp_path / "y.csv")
    write_csv([{"a": 1}, {"b": 2}], path)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["b"] == ""
    assert rows[1]["a"] == ""


def test_write_csv_refuses_an_empty_schemaless_write(tmp_path) -> None:
    """Silently writing an empty file would hide a broken run."""
    with pytest.raises(ValueError, match="empty CSV"):
        write_csv([], str(tmp_path / "z.csv"))


def test_write_csv_accepts_explicit_columns(tmp_path) -> None:
    """An explicit schema allows an empty but well-formed table."""
    path = str(tmp_path / "w.csv")
    write_csv([], path, columns=["a", "b"])
    with open(path, encoding="utf-8") as handle:
        assert handle.readline().strip() == "a,b"


def test_markdown_table_has_a_header_and_a_rule() -> None:
    """The Markdown emitted is well formed."""
    table = markdown_table([{"a": 1.5, "b": True}], ["a", "b"])
    lines = table.splitlines()
    assert lines[0] == "| a | b |"
    assert set(lines[1]) <= set("|-")
    assert "1.500" in lines[2] and "yes" in lines[2]


def test_control_rows_report_both_controls_and_meet_expectations() -> None:
    """The four constructed cases each produce the outcome they advertise."""
    rows = build_control_rows()
    assert len(rows) == 4
    by_case = {str(r["case"]): r for r in rows}

    artifact = next(r for k, r in by_case.items() if "artifact_demo" in k)
    assert artifact["is_artifact"] is True

    corroborated = next(r for k, r in by_case.items() if "corroborated_demo" in k)
    assert corroborated["is_artifact"] is False

    assert by_case["linear_ramp"]["classification"] == "ramp_type"
    assert by_case["linear_ramp"]["suppress_lead_time_claim"] is True
    assert by_case["step_change"]["classification"] == "step_type"
    assert by_case["step_change"]["suppress_lead_time_claim"] is False


def test_run_all_writes_every_advertised_artifact(tmp_path) -> None:
    """The one command produces the full set of outputs."""
    out = str(tmp_path / "results")
    result = run_all(out_dir=out, **FAST)
    for filename in (
        "sweep.csv",
        "false_positives.csv",
        "controls.csv",
        "ramp_ceiling.csv",
        "magnitude_sweep.png",
        "false_positive_rates.png",
        "SUMMARY.md",
    ):
        path = os.path.join(out, filename)
        assert os.path.exists(path), f"missing {filename}"
        assert os.path.getsize(path) > 0, f"empty {filename}"
    assert result["sweep"] and result["false_positives"]


def test_run_all_summary_contains_the_headline_result(tmp_path) -> None:
    """The generated summary states the constant and both figures."""
    result = run_all(out_dir=str(tmp_path / "r"), **FAST)
    summary = str(result["summary"])
    assert "1.349" in summary
    assert "magnitude_sweep.png" in summary
    assert "false_positive_rates.png" in summary
    assert "RandomScore" in summary


def test_run_all_png_files_are_real_pngs(tmp_path) -> None:
    """Guards against a figure that silently wrote a stub."""
    out = str(tmp_path / "results")
    run_all(out_dir=out, **FAST)
    for name in ("magnitude_sweep.png", "false_positive_rates.png"):
        with open(os.path.join(out, name), "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
        assert os.path.getsize(os.path.join(out, name)) > 10_000


def test_sweep_csv_has_the_expected_columns(tmp_path) -> None:
    """The published schema is stable enough to be depended on."""
    out = str(tmp_path / "results")
    run_all(out_dir=out, **FAST)
    with open(os.path.join(out, "sweep.csv"), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        for column in (
            "detector",
            "generator",
            "magnitude",
            "detection_rate",
            "mean_max_score",
            "point_f1",
            "event_f1",
            "point_adjusted_f1",
        ):
            assert column in reader.fieldnames


def test_run_all_can_skip_the_optional_detector(tmp_path) -> None:
    """The run works identically without stumpy."""
    result = run_all(out_dir=str(tmp_path / "r"), include_matrix_profile=False, **FAST)
    assert "MatrixProfile" not in {str(r["detector"]) for r in result["sweep"]}


def test_cli_run_writes_results(tmp_path, capsys) -> None:
    """The advertised command works end to end."""
    out = str(tmp_path / "cli")
    code = main(["run", "--out", out, "--repeats", "2", "--n", "400", "--magnitudes", "2", "10"])
    assert code == 0
    assert os.path.exists(os.path.join(out, "SUMMARY.md"))
    assert "tsad-harness results" in capsys.readouterr().out


def test_cli_quiet_suppresses_the_summary(tmp_path, capsys) -> None:
    """--quiet writes files without printing the tables."""
    out = str(tmp_path / "cli")
    main(["run", "--out", out, "--repeats", "2", "--n", "400", "--magnitudes", "2", "--quiet"])
    assert "tsad-harness results" not in capsys.readouterr().out


def test_cli_constant_prints_the_derived_value(capsys) -> None:
    """A one-liner for checking the central number."""
    assert main(["constant"]) == 0
    assert "1.3489815" in capsys.readouterr().out


def test_cli_list_names_generators_and_detectors(capsys) -> None:
    """Discoverability without reading the source."""
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "linear_ramp" in out and "pure_noise" in out
    assert "MADZScore" in out and "RandomScore" in out


def test_cli_rejects_an_unknown_command() -> None:
    """argparse exits rather than silently doing nothing."""
    with pytest.raises(SystemExit):
        main(["nope"])
