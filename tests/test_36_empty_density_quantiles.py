"""Regression tests for issue #36.

QuaMeter-compatible peak-density quantiles (MS:4000061/MS:4000062) must exclude
zero-length spectra, which QuaMeter skips (defaultArrayLength == 0). Empty scans
are already counted separately by MS:4000099/MS:4000100, so counting them again
as artificial 0-peak scans conflates two concepts and biases the quantiles low.
"""
import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    peak_density_quantiles,
    number_empty_scans,
    compute_qc_metrics,
)


def _spec_with_n_peaks(rt, n):
    sp = oms.MSSpectrum()
    sp.setRT(float(rt))
    sp.setMSLevel(1)
    if n > 0:
        mzs = np.arange(100.0, 100.0 + n)
        sp.set_peaks((mzs, np.full(n, 5.0)))
    return sp


def test_empty_scans_excluded_from_density():
    exp = oms.MSExperiment()
    for i, n in enumerate([10, 20, 30, 40]):
        exp.addSpectrum(_spec_with_n_peaks(i, n))
    # two zero-length scans that must not enter the density distribution
    exp.addSpectrum(_spec_with_n_peaks(4, 0))
    exp.addSpectrum(_spec_with_n_peaks(5, 0))
    exp.updateRanges()

    got = peak_density_quantiles(exp, ms_level=1)
    expected = list(np.quantile([10, 20, 30, 40], [0.25, 0.50, 0.75]))
    assert got == [float(x) for x in expected]

    # empty scans remain counted by their own metric
    assert number_empty_scans(exp, 1) == 2


def test_all_empty_returns_nan():
    exp = oms.MSExperiment()
    exp.addSpectrum(_spec_with_n_peaks(0, 0))
    exp.addSpectrum(_spec_with_n_peaks(1, 0))
    exp.updateRanges()
    got = peak_density_quantiles(exp, ms_level=1)
    assert all(np.isnan(x) for x in got)


def test_compute_wires_density_and_empty_counts():
    exp = oms.MSExperiment()
    for i, n in enumerate([10, 20, 30, 40]):
        exp.addSpectrum(_spec_with_n_peaks(i, n))
    exp.addSpectrum(_spec_with_n_peaks(4, 0))
    exp.updateRanges()
    m = compute_qc_metrics(exp)
    assert m["PeakDensity_MS1_Q2"] == 25.0
    assert m["EmptyScans_MS1"] == 1
