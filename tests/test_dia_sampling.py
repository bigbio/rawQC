"""Numerical oracles for observed DIA sampling and isolation-window coverage."""

import json

import numpy as np
import pyopenms as oms
import pytest

from rawQC.dia import compute_dia_metrics


def _scan(rt, lower=400, upper=420, mobility=None, unit=None, cv=None):
    spectrum = oms.MSSpectrum()
    spectrum.setMSLevel(2)
    spectrum.setRT(float(rt))
    spectrum.set_peaks((np.asarray([100.0]), np.asarray([10.0])))
    precursor = oms.Precursor()
    precursor.setMZ((lower + upper) / 2)
    precursor.setIsolationWindowLowerOffset((upper - lower) / 2)
    precursor.setIsolationWindowUpperOffset((upper - lower) / 2)
    spectrum.setPrecursors([precursor])
    if unit is not None:
        spectrum.setDriftTimeUnit(unit)
    if mobility is not None:
        spectrum.setMetaValue("ion mobility lower limit", mobility[0])
        spectrum.setMetaValue("ion mobility upper limit", mobility[1])
    if cv is not None:
        spectrum.setDriftTime(float(cv))
    return spectrum


def _metrics(spectra):
    experiment = oms.MSExperiment()
    for spectrum in spectra:
        experiment.addSpectrum(spectrum)
    result = compute_dia_metrics(experiment)
    json.dumps(result, allow_nan=False)
    return result


def _rows(table):
    return [dict(zip(table, values)) for values in zip(*table.values())]


def test_revisit_uses_window_identity_and_keeps_simultaneous_windows_separate():
    spectra = []
    for rt in (0, 2, 4, 9):
        spectra.extend([_scan(rt, mobility=(0.7, 0.9), unit=oms.DriftTimeUnit.VSSC),
                        _scan(rt, mobility=(1.0, 1.2), unit=oms.DriftTimeUnit.VSSC)])
    metrics = _metrics(spectra)
    table = metrics["DIA_IsolationWindow_RevisitSummary"]
    assert table["window_index"] == [0, 1]
    assert table["observation_count"] == [4, 4]
    assert table["positive_interval_count"] == [3, 3]
    assert table["interval_median_seconds"] == [2, 2]
    assert table["interval_cv"] == pytest.approx([2 ** 0.5 / 3, 2 ** 0.5 / 3])
    assert table["equal_rt_interval_count"] == [0, 0]
    assert table["long_interval_count"] == [1, 1]
    assert metrics["DIA_MS1BoundedCycle_Count"] == 0


def test_revisit_does_not_sort_or_bridge_invalid_retention_times():
    table = _metrics([_scan(rt) for rt in (0, 0, 2, 1, float("nan"), 10, 12)])["DIA_IsolationWindow_RevisitSummary"]
    assert table["nonfinite_rt_count"] == [1]
    assert table["equal_rt_interval_count"] == [1]
    assert table["decreasing_rt_interval_count"] == [1]
    assert table["invalid_interval_count"] == [2]
    assert table["positive_interval_count"] == [2]
    assert table["interval_min_seconds"] == [2]
    assert table["interval_max_seconds"] == [2]
    assert table["long_interval_count"] == [None]


def test_revisit_long_interval_threshold_is_strict_and_needs_three_intervals():
    table = _metrics([_scan(rt) for rt in (0, 2, 4, 7)])["DIA_IsolationWindow_RevisitSummary"]
    assert table["long_interval_count"] == [0]  # 3 seconds equals 1.5 * median
    table = _metrics([_scan(rt) for rt in (0, 1, 100)])["DIA_IsolationWindow_RevisitSummary"]
    assert table["long_interval_count"] == [None]
    table = _metrics([_scan(0)])["DIA_IsolationWindow_RevisitSummary"]
    assert table["positive_interval_count"] == [0]
    assert table["interval_median_seconds"] == [None]
    assert table["interval_cv"] == [None]


def test_revisit_intervals_are_partitioned_by_faims_voltage():
    table = _metrics([
        _scan(rt, unit=oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE, cv=cv)
        for rt, cv in ((0, -45), (1, -65), (2, -45), (4, -65), (6, -45))
    ])["DIA_IsolationWindow_RevisitSummary"]
    assert table["observation_count"] == [2, 3]  # sorted CV -65, -45
    assert table["interval_median_seconds"] == [3, 3]
    assert table["positive_interval_count"] == [1, 2]


def test_mz_union_gap_and_overlap_have_exact_widths():
    rows = _rows(_metrics([
        _scan(0, 400, 420), _scan(1, 410, 430), _scan(2, 440, 450), _scan(3, 400, 420)
    ])["DIA_IsolationWindow_MzCoverage"])
    assert rows == [{
        "mobility_mode": "none", "drift_time_unit": None, "faims_cv": None,
        "window_count": 3, "distinct_mz_interval_count": 3,
        "mz_lower": 400, "mz_upper": 450, "envelope_width_mz": 50,
        "covered_width_mz": 40, "gap_width_mz": 10, "overlap_width_mz": 10,
        "coverage_fraction": 0.8, "gap_count": 1,
    }]


def test_touching_mz_intervals_have_no_gap_or_overlap():
    row, = _rows(_metrics([_scan(0, 400, 420), _scan(1, 420, 440)])[
        "DIA_IsolationWindow_MzCoverage"])
    assert row["covered_width_mz"] == 40
    assert row["gap_width_mz"] == 0
    assert row["gap_count"] == 0
    assert row["overlap_width_mz"] == 0
    assert row["coverage_fraction"] == 1


def test_mz_projection_deduplicates_identical_intervals_across_im_selections():
    metrics = _metrics([
        _scan(0, mobility=(0.7, 0.9), unit=oms.DriftTimeUnit.VSSC),
        _scan(0, mobility=(1.0, 1.2), unit=oms.DriftTimeUnit.VSSC),
    ])
    row, = _rows(metrics["DIA_IsolationWindow_MzCoverage"])
    assert row["window_count"] == 2
    assert row["distinct_mz_interval_count"] == 1
    assert row["covered_width_mz"] == 20
    assert row["overlap_width_mz"] == 0
    assert row["coverage_fraction"] == 1
    row, = _rows(metrics["DIA_IsolationWindow_MzIMCoverage"])
    assert row["covered_area"] == pytest.approx(8)
    assert row["uncovered_area"] == pytest.approx(2)
    assert row["coverage_fraction"] == pytest.approx(0.8)


def test_rectangle_union_overlap_and_uncovered_bounding_area():
    metrics = _metrics([
        _scan(0, 400, 420, (0.7, 1.0), oms.DriftTimeUnit.VSSC),
        _scan(0, 410, 430, (0.9, 1.2), oms.DriftTimeUnit.VSSC),
    ])
    row, = _rows(metrics["DIA_IsolationWindow_MzIMCoverage"])
    assert row["rectangle_count"] == 2
    assert row["excluded_window_count"] == 0
    assert row["envelope_area"] == pytest.approx(15)
    assert row["covered_area"] == pytest.approx(11)
    assert row["uncovered_area"] == pytest.approx(4)
    assert row["overlap_area"] == pytest.approx(1)
    assert row["coverage_fraction"] == pytest.approx(11 / 15)


def test_overlap_is_region_covered_twice_or_more_not_excess_multiplicity():
    metrics = _metrics([_scan(i, low, high, (1, 2), oms.DriftTimeUnit.VSSC)
                        for i, low, high in ((0, 400, 430), (1, 410, 440), (2, 420, 450))])
    mz, = _rows(metrics["DIA_IsolationWindow_MzCoverage"])
    im, = _rows(metrics["DIA_IsolationWindow_MzIMCoverage"])
    assert mz["covered_width_mz"] == 50
    assert mz["overlap_width_mz"] == 30
    assert im["covered_area"] == 50
    assert im["overlap_area"] == 30


def test_coverage_keeps_faims_units_and_unannotated_windows_separate():
    metrics = _metrics([
        _scan(0, 400, 410),
        _scan(1, 500, 510, (1, 2)),  # recorded bounds, unknown unit
        _scan(2, 600, 610, (1, 2), oms.DriftTimeUnit.VSSC),
        _scan(3, 700, 710, (1, 2), oms.DriftTimeUnit.MILLISECOND),
        _scan(4, 800, 810, unit=oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE, cv=-45),
        _scan(5, 900, 910, unit=oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE, cv=-65),
    ])
    mz_rows = _rows(metrics["DIA_IsolationWindow_MzCoverage"])
    im_rows = _rows(metrics["DIA_IsolationWindow_MzIMCoverage"])
    assert len(mz_rows) == 6
    assert all(row["gap_width_mz"] == 0 for row in mz_rows)
    assert sorted(row["faims_cv"] for row in mz_rows if row["mobility_mode"] == "faims") == [-65, -45]
    assert len(im_rows) == 2
    assert {row["drift_time_unit"] for row in im_rows} == {
        oms.DriftTimeUnit.VSSC.value, oms.DriftTimeUnit.MILLISECOND.value}
    assert all(row["covered_area"] == 10 for row in im_rows)


@pytest.mark.parametrize("bounds", [(1, 1), (2, 1), (-1, 1), (float("nan"), 1)])
def test_invalid_mobility_bounds_are_excluded_without_a_fabricated_envelope(bounds):
    row, = _rows(_metrics([_scan(0, mobility=bounds, unit=oms.DriftTimeUnit.VSSC)])[
        "DIA_IsolationWindow_MzIMCoverage"])
    assert row["window_count"] == 1
    assert row["excluded_window_count"] == 1
    assert row["rectangle_count"] == 0
    assert row["envelope_area"] is None
    assert row["covered_area"] is None


def test_precursor_offsets_define_coverage_with_known_unit_when_reader_bounds_absent():
    spectrum = _scan(0, unit=oms.DriftTimeUnit.VSSC)
    precursor = spectrum.getPrecursors()[0]
    precursor.setDriftTime(1.0)
    precursor.setDriftTimeWindowLowerOffset(0.1)
    precursor.setDriftTimeWindowUpperOffset(0.2)
    spectrum.setPrecursors([precursor])
    row, = _rows(_metrics([spectrum])["DIA_IsolationWindow_MzIMCoverage"])
    assert row["ion_mobility_lower"] == pytest.approx(0.9)
    assert row["ion_mobility_upper"] == pytest.approx(1.2)
    assert row["covered_area"] == pytest.approx(6)


def test_duplicate_rectangles_do_not_create_geometric_overlap():
    spectra = [_scan(i, mobility=(1, 2), unit=oms.DriftTimeUnit.VSSC) for i in (0, 1)]
    precursor = spectra[1].getPrecursors()[0]
    precursor.setDriftTime(1.5)  # distinct window identity, same explicit rectangle
    spectra[1].setPrecursors([precursor])
    row, = _rows(_metrics(spectra)["DIA_IsolationWindow_MzIMCoverage"])
    assert row["window_count"] == 2
    assert row["rectangle_count"] == 1
    assert row["covered_area"] == 20
    assert row["overlap_area"] == 0


def test_explicit_invalid_reader_bounds_do_not_fall_back_to_precursor_selection():
    spectrum = _scan(0, mobility=(float("nan"), float("nan")), unit=oms.DriftTimeUnit.VSSC)
    precursor = spectrum.getPrecursors()[0]
    precursor.setDriftTime(1.0)
    precursor.setDriftTimeWindowLowerOffset(0.1)
    precursor.setDriftTimeWindowUpperOffset(0.2)
    spectrum.setPrecursors([precursor])
    row, = _rows(_metrics([spectrum])["DIA_IsolationWindow_MzIMCoverage"])
    assert row["excluded_window_count"] == 1
    assert row["covered_area"] is None


def test_empty_run_has_empty_sampling_tables():
    metrics = _metrics([])
    for key in ("DIA_IsolationWindow_RevisitSummary", "DIA_IsolationWindow_MzCoverage",
                "DIA_IsolationWindow_MzIMCoverage"):
        assert all(values == [] for values in metrics[key].values())
