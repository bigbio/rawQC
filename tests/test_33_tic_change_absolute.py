"""Regression tests for issue #33.

MS:4000186 (MS1 TIC-change quartile ratios) is described by PSI-MS as "the
original QuaMeter metrics", and QuaMeter computes the scan-to-scan change with
fabs(). Using signed diff() produced negative change quartiles whose ratios/logs
are undefined (NaN). rawQC now uses absolute changes, so:
  * a rising run and its exact reversal give identical change quartiles, and
  * an alternating run yields finite results instead of NaN.
"""
import math

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    tic_quartile_to_quartile_log_ratio,
    compute_qc_metrics,
)


def _ms1_run(tic_values):
    exp = oms.MSExperiment()
    for i, tic in enumerate(tic_values):
        sp = oms.MSSpectrum()
        sp.setRT(float(i))
        sp.setMSLevel(1)
        # single peak whose intensity is the desired TIC
        sp.set_peaks((np.array([500.0]), np.array([float(tic)])))
        exp.addSpectrum(sp)
    exp.updateRanges()
    return exp


def test_rising_equals_falling_magnitude():
    rising = _ms1_run([1.0, 10.0, 100.0, 1000.0])
    falling = _ms1_run([1000.0, 100.0, 10.0, 1.0])
    r = tic_quartile_to_quartile_log_ratio(rising, 1, mode="TIC_change")
    f = tic_quartile_to_quartile_log_ratio(falling, 1, mode="TIC_change")
    assert all(math.isfinite(x) for x in r)
    assert r == f  # absolute change is symmetric under time reversal


def test_alternating_is_finite_not_nan():
    exp = _ms1_run([10.0, 1.0, 10.0, 1.0, 10.0])
    got = tic_quartile_to_quartile_log_ratio(exp, 1, mode="TIC_change")
    # |diff| = [9, 9, 9, 9] -> all quartiles equal -> log(1) == 0, never NaN
    assert got == [0.0, 0.0, 0.0]


def test_compute_wires_change_metric_finite():
    exp = _ms1_run([1.0, 5.0, 25.0, 125.0, 625.0])
    m = compute_qc_metrics(exp)
    assert math.isfinite(m["TIC_MS1_Change_Q2"])
    assert math.isfinite(m["TIC_MS1_Change_Q3"])
    assert math.isfinite(m["TIC_MS1_Change_Q4"])
