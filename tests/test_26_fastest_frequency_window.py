"""Regression tests for issue #26.

The fastest acquisition frequency (MS:4000065/MS:4000066) must be the maximum
rate sustained over a one-minute window (QuaMeter/macproqc), not the inverse of
the smallest gap between two scans. A single very short gap embedded in an
otherwise slow run must not dominate the metric.
"""
import math

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import fastest_ms_frequency, compute_qc_metrics


def _run(rts, level=1):
    exp = oms.MSExperiment()
    for rt in rts:
        sp = oms.MSSpectrum()
        sp.setRT(float(rt))
        sp.setMSLevel(level)
        sp.set_peaks((np.array([500.0]), np.array([100.0])))
        if level >= 2:
            prec = oms.Precursor(); prec.setMZ(500.0); prec.setCharge(2)
            sp.setPrecursors([prec])
        exp.addSpectrum(sp)
    exp.updateRanges()
    return exp


def test_single_fast_gap_does_not_dominate():
    # ~1 Hz run for 100 s, with one extra scan 0.01 s after t=50 (a 100 Hz pair).
    rts = list(range(100)) + [50.01]
    exp = _run(rts)
    freq = fastest_ms_frequency(exp, 1)
    # Old code (1/min-gap) would report 100 Hz. Window count: 61 integer scans
    # in [t, t+60] plus the extra -> 62; 62/60.
    assert freq == 62 / 60.0
    assert freq < 2.0  # the 100 Hz pair does not dominate


def test_uniform_two_hz():
    rts = list(np.arange(0.0, 120.0001, 0.5))  # 2 Hz for 120 s
    exp = _run(rts)
    # [0, 60] contains 0, 0.5, ..., 60 -> 121 scans; 121/60.
    assert fastest_ms_frequency(exp, 1) == 121 / 60.0


def test_short_run_uses_full_window_denominator():
    rts = list(range(10))  # 10 scans over 9 s (< 60 s)
    exp = _run(rts)
    assert fastest_ms_frequency(exp, 1) == 10 / 60.0


def test_no_scans_is_nan():
    exp = _run([], level=1)
    assert math.isnan(fastest_ms_frequency(exp, 1))


def test_compute_wiring():
    exp = _run(list(range(120)))
    m = compute_qc_metrics(exp)
    assert m["FastestFrequency_MS1"] == 61 / 60.0
