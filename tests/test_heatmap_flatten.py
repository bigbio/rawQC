"""Tests for the heatmap flattening helper.

n-tuple and table metrics (charge fractions, RT quantiles, ranges, analyzer/
activation tables) must be expanded into per-element rows so they appear in the
heatmap instead of being dropped (a list/dict cannot fill a single cell).
"""
import math

from rawQC.calculate_metrics import _flatten_metric_for_heatmap


def test_scalar():
    assert _flatten_metric_for_heatmap("A", 5.0) == [("A", 5.0)]
    assert _flatten_metric_for_heatmap("A", 3) == [("A", 3.0)]


def test_none_and_string_and_bool_skipped():
    assert _flatten_metric_for_heatmap("A", None) == []
    assert _flatten_metric_for_heatmap("A", "centroid") == []
    assert _flatten_metric_for_heatmap("A", True) == []


def test_detailed_scan_records_do_not_expand_into_heatmap_rows():
    table = {"native_id": ["scan=1", "scan=2"], "ion_injection_time_ms": [10, 20]}
    assert _flatten_metric_for_heatmap("Acquisition_ScanMetadata", table) == []
    assert _flatten_metric_for_heatmap("IonInjectionTime_RTSummary", table) == []
    assert _flatten_metric_for_heatmap("IonInjectionTime_Summary", table)


def test_ntuple_expanded():
    assert _flatten_metric_for_heatmap("RT", [0.25, 0.25, 0.25, 0.25]) == [
        ("RT[0]", 0.25), ("RT[1]", 0.25), ("RT[2]", 0.25), ("RT[3]", 0.25)]


def test_ntuple_drops_non_numeric_entries():
    rows = dict(_flatten_metric_for_heatmap("X", [1.0, None, 3.0]))
    assert rows == {"X[0]": 1.0, "X[2]": 3.0}


def test_table_expanded_by_label_column():
    tbl = {"charge_state": ["1", "2", ">=6"], "count": [0, 5, 1],
           "fraction": [0.0, 0.833, 0.167]}
    rows = dict(_flatten_metric_for_heatmap("Z", tbl))
    assert rows["Z.count[2]"] == 5.0
    assert rows["Z.fraction[>=6]"] == 0.167
    # the label column itself is not emitted as a numeric row
    assert not any(k.startswith("Z.charge_state") for k in rows)


def test_table_with_missing_numeric_skipped():
    tbl = {"index": [0, 1], "type": ["ORBITRAP", "IT"], "resolution": [60000.0, None]}
    rows = dict(_flatten_metric_for_heatmap("MA", tbl))
    assert rows.get("MA.resolution[ORBITRAP]") == 60000.0
    assert "MA.resolution[IT]" not in rows          # None dropped
    assert not any(".index[" in k for k in rows)    # positional index suppressed


def test_filling_group_labels_remain_distinct_and_stable_across_runs():
    table = {"ms_level": [2, 2], "faims_cv_volts": [-50, -50],
             "isolation_metadata_status": ["recorded", "recorded"],
             "isolation_lower_mz": [400, 500], "isolation_upper_mz": [425, 525],
             "injection_time_median_ms": [10, 50]}
    rows = _flatten_metric_for_heatmap("IonInjectionTime_Summary", table)
    assert len(rows) == len(dict(rows)) == 2
    assert sorted(value for _, value in rows) == [10, 50]
    reversed_rows = _flatten_metric_for_heatmap("IonInjectionTime_Summary",
                                               {key: values[::-1] for key, values in table.items()})
    assert dict(rows) == dict(reversed_rows)


def test_coverage_heatmap_labels_keep_faims_voltages_separate():
    table = {"mobility_mode": ["faims", "faims"], "drift_time_unit": [3, 3],
             "faims_cv": [-50, -70], "coverage_fraction": [0.5, 1.0]}
    rows = _flatten_metric_for_heatmap("DIA_IsolationWindow_MzCoverage", table)
    assert len(rows) == len(dict(rows)) == 2
