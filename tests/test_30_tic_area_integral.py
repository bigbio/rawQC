"""Regression tests for issue #30.

MS:4000029/030/155/156 are "area under the total ion chromatogram" terms. On
irregularly sampled data a per-spectrum sum is not an area. rawQC now computes a
true trapezoidal time integral of the TIC against retention time, which is
distinguishable from a sum on an irregular-RT fixture.
"""
import math

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    area_under_tic,
    area_under_tic_rt_quantiles,
    compute_qc_metrics,
)


def _ms1_run(rt_tic):
    exp = oms.MSExperiment()
    for rt, tic in rt_tic:
        sp = oms.MSSpectrum()
        sp.setRT(float(rt))
        sp.setMSLevel(1)
        sp.set_peaks((np.array([500.0]), np.array([float(tic)])))
        exp.addSpectrum(sp)
    exp.updateRanges()
    return exp


def test_integral_differs_from_sum_on_irregular_rt():
    # RTs 0, 1, 10 with constant TIC 10.
    # sum = 30; trapezoidal integral = 10*1 + 10*9 = 100.
    exp = _ms1_run([(0, 10), (1, 10), (10, 10)])
    assert area_under_tic(exp, 1) == 100.0
    assert area_under_tic(exp, 1) != 30.0


def test_regular_sampling_integral():
    # RTs 0,1,2,3 TIC 10,20,30,40 -> trapz = 15+25+35 = 75
    exp = _ms1_run([(0, 10), (1, 20), (2, 30), (3, 40)])
    assert area_under_tic(exp, 1) == 75.0


def test_single_scan_is_nan():
    exp = _ms1_run([(5, 100)])
    assert math.isnan(area_under_tic(exp, 1))


def test_nan_rt_does_not_flip_sign():
    # A stray non-finite RT must be dropped before ordering; otherwise the
    # remaining scans stay unsorted and the trapezoid integral goes negative.
    exp = _ms1_run([(2, 2), (float("nan"), 0), (1, 1)])
    assert area_under_tic(exp, 1) == 1.5  # trapz over RT 1..2 of TIC 1..2
    areas = area_under_tic_rt_quantiles(exp, 1)
    assert all(a >= 0 for a in areas)
    assert abs(sum(areas) - 1.5) < 1e-9


def test_rt_quantile_areas_conserve_and_dont_collapse():
    # 8 evenly spaced scans, constant TIC 10. Whole-run trapezoidal integral is
    # 10 * (rt span 7) = 70. The four quartile integrals must sum to 70 and NONE
    # may collapse to 0 (the earlier per-bin "<2 scans -> 0" rule regressed this,
    # e.g. RTQ1 -> 0 for the sparse first quartile).
    exp = _ms1_run([(float(i), 10.0) for i in range(8)])
    areas = area_under_tic_rt_quantiles(exp, 1)
    assert abs(sum(areas) - area_under_tic(exp, 1)) < 1e-9
    assert abs(sum(areas) - 70.0) < 1e-9
    assert all(a > 0 for a in areas), areas


def test_rt_quantile_areas_on_sparse_first_quartile():
    # The irregular fixture the issue targets: first quartile has a single scan.
    # It must not zero the whole tuple.
    exp = _ms1_run([(0, 10), (1, 10), (10, 10)])
    areas = area_under_tic_rt_quantiles(exp, 1)
    assert abs(sum(areas) - area_under_tic(exp, 1)) < 1e-9
    assert sum(areas) > 0


def test_wired_into_compute():
    exp = _ms1_run([(0, 10), (1, 10), (10, 10)])
    m = compute_qc_metrics(exp)
    assert m["TIC_MS1_Area"] == 100.0
