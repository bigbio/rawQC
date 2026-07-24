"""Regression tests for issue #38.

MS:4000202 ("maximum base peak intensity of all spectra in a single run") must
consider every MS level. The old code only looked at the separately collected
MS1 and MS2 base peaks, so a run whose largest base peak lives in an MS3 scan
reported a too-small maximum.
"""
import math

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import compute_qc_metrics, max_base_peak_intensity


def _spec(rt, level, intens):
    sp = oms.MSSpectrum()
    sp.setRT(float(rt))
    sp.setMSLevel(level)
    mzs = np.arange(100.0, 100.0 + len(intens))
    sp.set_peaks((mzs, np.asarray(intens, dtype=float)))
    if level >= 2:
        prec = oms.Precursor()
        prec.setMZ(rt + 400.0)
        prec.setCharge(2)
        sp.setPrecursors([prec])
    return sp


def test_ms3_base_peak_is_the_max():
    exp = oms.MSExperiment()
    exp.addSpectrum(_spec(1.0, 1, [10.0, 100.0, 30.0]))   # MS1 max 100
    exp.addSpectrum(_spec(2.0, 2, [50.0, 200.0]))          # MS2 max 200
    exp.addSpectrum(_spec(3.0, 3, [500.0, 20.0]))          # MS3 max 500 <- largest
    exp.updateRanges()
    assert max_base_peak_intensity(exp) == 500.0
    metrics = compute_qc_metrics(exp)
    assert metrics["BasePeak_All_Max"] == 500.0
    # MS1/MS2 means unaffected.
    assert metrics["BasePeak_MS1_Mean"] == 100.0
    assert metrics["BasePeak_MS2_Mean"] == 200.0


def test_empty_run_returns_nan():
    exp = oms.MSExperiment()
    assert math.isnan(max_base_peak_intensity(exp))


def test_empty_scans_skipped():
    exp = oms.MSExperiment()
    empty = oms.MSSpectrum(); empty.setMSLevel(1); empty.setRT(1.0)
    exp.addSpectrum(empty)
    exp.addSpectrum(_spec(2.0, 1, [42.0, 7.0]))
    exp.updateRanges()
    assert max_base_peak_intensity(exp) == 42.0
