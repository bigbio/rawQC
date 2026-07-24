"""Regression tests for issue #39.

_precursor_values used truthiness when reading the precursor intensity, so a
legitimate 0.0 was turned into NaN and silently dropped. rawQC now distinguishes
an absent precursor (NaN) from a present numeric zero, and follows QuaMeter's
explicit fallback: a zero/unrecorded precursor intensity is replaced by the
spectrum MS2 TIC, with the substitution count reported.
"""
import math

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    precursor_intensities,
    precursor_intensity_stats,
    compute_qc_metrics,
)


def _ms2(rt, prec_intensity, tic, has_precursor=True):
    sp = oms.MSSpectrum()
    sp.setRT(float(rt))
    sp.setMSLevel(2)
    # single peak whose intensity is the MS2 TIC
    sp.set_peaks((np.array([100.0]), np.array([float(tic)])))
    if has_precursor:
        prec = oms.Precursor()
        prec.setMZ(500.0)
        prec.setCharge(2)
        prec.setIntensity(float(prec_intensity))
        sp.setPrecursors([prec])
    return sp


def _exp(specs):
    exp = oms.MSExperiment()
    for s in specs:
        exp.addSpectrum(s)
    exp.updateRanges()
    return exp


def test_absent_vs_zero_vs_positive_with_fallback():
    specs = [
        _ms2(1, 1000.0, 30.0),                 # positive -> 1000
        _ms2(2, 0.0, 50.0),                    # present zero -> fallback to TIC 50
        _ms2(3, 0.0, 0.0, has_precursor=False),  # absent -> NaN
        _ms2(4, 2000.0, 40.0),                 # positive -> 2000
    ]
    vals, n_fb, n_missing = precursor_intensities(specs)
    assert vals[0] == 1000.0
    assert vals[1] == 50.0          # zero preserved via MS2-TIC fallback, not dropped
    assert math.isnan(vals[2])
    assert vals[3] == 2000.0
    assert n_fb == 1
    assert n_missing == 1


def test_zero_preserved_without_fallback():
    specs = [_ms2(1, 0.0, 77.0)]
    vals, n_fb, _ = precursor_intensities(specs, fallback_to_ms2_tic=False)
    assert vals[0] == 0.0           # present zero preserved as zero, not NaN
    assert n_fb == 0


def test_stats_include_fallback_value_and_count_wired():
    exp = _exp([
        _ms2(1, 1000.0, 30.0),
        _ms2(2, 0.0, 50.0),
        _ms2(4, 2000.0, 40.0),
    ])
    stats = precursor_intensity_stats(exp, 2)
    # distribution is [1000, 50, 2000] -> median 1000
    assert stats["PrecursorIntensity_Q2"] == 1000.0
    m = compute_qc_metrics(exp)
    assert m["PrecursorIntensity_FallbackCount"] == 1
