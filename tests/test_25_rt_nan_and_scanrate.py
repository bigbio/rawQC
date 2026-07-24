"""Regression tests for issue #25.

* chromatography_duration must ignore non-finite retention times (a single NaN
  RT otherwise poisons the duration and every dependent metric).
* Level-specific scan rates must use each level's own acquisition span, not one
  combined MS1/MS2 run duration.
"""
import math

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    chromatography_duration,
    scan_rate,
    compute_qc_metrics,
)


def _spec(rt, level):
    sp = oms.MSSpectrum()
    sp.setRT(float(rt))
    sp.setMSLevel(level)
    sp.set_peaks((np.array([500.0]), np.array([100.0])))
    if level >= 2:
        prec = oms.Precursor(); prec.setMZ(500.0); prec.setCharge(2)
        sp.setPrecursors([prec])
    return sp


def _exp(specs):
    exp = oms.MSExperiment()
    for s in specs:
        exp.addSpectrum(s)
    exp.updateRanges()
    return exp


def test_duration_ignores_nan_rt():
    exp = _exp([_spec(0.0, 1), _spec(float("nan"), 1), _spec(120.0, 1)])
    assert chromatography_duration(exp) == 120.0


def test_scan_rate_is_level_specific():
    # MS1 at 0, 60, 120 s -> span 2 min, 3 scans -> 1.5 /min
    # MS2 at 30, 90 s      -> span 1 min, 2 scans -> 2.0 /min (old combined-
    #                          duration code would have said 1.0 /min)
    exp = _exp([_spec(0, 1), _spec(60, 1), _spec(120, 1), _spec(30, 2), _spec(90, 2)])
    assert scan_rate(exp, 1) == 1.5
    assert scan_rate(exp, 2) == 2.0
    m = compute_qc_metrics(exp)
    assert m["ScanRate_MS1"] == 1.5
    assert m["ScanRate_MS2"] == 2.0


def test_scan_rate_edge_cases():
    assert math.isnan(scan_rate(_exp([_spec(5.0, 1)]), 1))          # one scan
    assert math.isnan(scan_rate(_exp([]), 1))                        # none
    # duplicate RTs -> zero span -> undefined
    assert math.isnan(scan_rate(_exp([_spec(5.0, 1), _spec(5.0, 1)]), 1))
