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
