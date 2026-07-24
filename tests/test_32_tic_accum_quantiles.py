"""Regression tests for issue #32.

MS:4000183 is an n-tuple of four RT *interval* widths (original QuaMeter),
normalized by the applicable MS-level duration. rawQC previously emitted five
cumulative RT positions normalized by the whole-experiment duration across all
MS levels. The four intervals now sum to 1.0 and use the level's own span.
"""
import json

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    tic_quantile_rt_fraction,
    compute_qc_metrics,
    build_mzqc,
)


def _spec(rt, level, tic):
    sp = oms.MSSpectrum()
    sp.setRT(float(rt))
    sp.setMSLevel(level)
    sp.set_peaks((np.array([500.0]), np.array([float(tic)])))
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


def test_uniform_tic_even_quarters():
    exp = _exp([_spec(rt, 1, 1.0) for rt in range(101)])  # 0..100 s, uniform TIC
    vals = tic_quantile_rt_fraction(exp, 1)
    assert vals == [0.25, 0.25, 0.25, 0.25]
    assert sum(vals) == 1.0


def test_level_duration_not_whole_experiment():
    # MS1 spans 0..40 s; MS2 spans 0..100 s. The MS1 metric must normalize by the
    # MS1 span (40 s) -> intervals sum to 1.0. Using the whole-experiment
    # duration (100 s) would have made them sum to 0.4.
    ms1 = [_spec(rt, 1, 1.0) for rt in range(41)]         # 0..40
    ms2 = [_spec(rt, 2, 1.0) for rt in range(0, 101, 10)]  # 0..100
    exp = _exp(ms1 + ms2)
    vals = tic_quantile_rt_fraction(exp, 1)
    assert vals == [0.25, 0.25, 0.25, 0.25]
    assert sum(vals) == 1.0


def test_front_loaded_tic():
    # All TIC in the first scan -> cumulative reaches 25/50/75% immediately at t=0
    # -> first three intervals 0, last interval spans the whole run.
    specs = [_spec(0, 1, 100.0)] + [_spec(rt, 1, 0.0) for rt in range(1, 11)]
    exp = _exp(specs)
    vals = tic_quantile_rt_fraction(exp, 1)
    assert vals == [0.0, 0.0, 0.0, 1.0]


def test_emitted_as_single_ntuple():
    exp = _exp([_spec(rt, 1, 1.0) for rt in range(101)])
    m = compute_qc_metrics(exp)
    assert m["RT_TIC_Quantiles"] == [0.25, 0.25, 0.25, 0.25]
    js = build_mzqc([{"filename": "r.mzML", "metrics": m, "instrument_metadata": {}}])
    qms = json.loads(js)["mzQC"]["runQualities"][0]["qualityMetrics"]
    nt = [q for q in qms if q.get("accession") == "MS:4000183"]
    assert len(nt) == 1 and nt[0]["value"] == [0.25, 0.25, 0.25, 0.25]
