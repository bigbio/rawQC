"""Regression tests for issue #37.

peak_type_statistics recorded the first annotated spectrum type (and the first
estimatable spectrum) for each MS level, so a mixed profile/centroid run, or a
misleading first spectrum, was reported as homogeneous. It now aggregates over
all spectra of a level and reports centroid/profile/mixed/unknown plus a profile
fraction.
"""
import math

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import peak_type_statistics, compute_qc_metrics

ST = oms.SpectrumSettings.SpectrumType


def _spec(level, stype, n_peaks=3):
    sp = oms.MSSpectrum()
    sp.setRT(1.0)
    sp.setMSLevel(level)
    sp.setType(stype)
    mzs = np.arange(100.0, 100.0 + n_peaks)
    sp.set_peaks((mzs, np.full(n_peaks, 5.0)))
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


def test_mixed_run_reported_as_mixed():
    # first spectrum centroid, the rest profile -> old code would say "centroid"
    exp = _exp([
        _spec(1, ST.CENTROID),
        _spec(1, ST.PROFILE),
        _spec(1, ST.PROFILE),
        _spec(1, ST.PROFILE),
    ])
    stats = peak_type_statistics(exp)
    assert stats["MS1_PeakType_Annotated"] == "mixed"
    assert stats["MS1_PeakType_Annotated_ProfileFraction"] == 0.75


def test_homogeneous_and_unknown():
    exp = _exp([_spec(1, ST.PROFILE), _spec(1, ST.PROFILE)])
    stats = peak_type_statistics(exp)
    assert stats["MS1_PeakType_Annotated"] == "profile"
    assert stats["MS1_PeakType_Annotated_ProfileFraction"] == 1.0

    exp2 = _exp([_spec(1, ST.UNKNOWN), _spec(1, ST.UNKNOWN)])
    stats2 = peak_type_statistics(exp2)
    assert stats2["MS1_PeakType_Annotated"] == "unknown"
    assert math.isnan(stats2["MS1_PeakType_Annotated_ProfileFraction"])


def test_wired_into_compute():
    exp = _exp([_spec(1, ST.CENTROID), _spec(1, ST.PROFILE),
                _spec(2, ST.CENTROID), _spec(2, ST.CENTROID)])
    m = compute_qc_metrics(exp)
    assert m["MS1_PeakType_Annotated"] == "mixed"
    assert m["MS2_PeakType_Annotated"] == "centroid"
    assert m["MS2_PeakType_Annotated_ProfileFraction"] == 0.0
