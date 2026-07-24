"""Regression tests for issue #31.

area_under_tic_rt_quantiles excluded the minimum-RT spectrum (first bin used
`rts > q0`), and compute_qc_metrics exposed only Q1-Q3, discarding Q4. rawQC now
assigns every finite-RT spectrum to exactly one quartile bin (so the four values
conserve the whole-run TIC) and emits all four as one MS:4000156 n-tuple.
"""
import json

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    area_under_tic_rt_quantiles,
    area_under_tic,
    compute_qc_metrics,
    build_mzqc,
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


def test_minimum_rt_spectrum_included_and_all_four_returned():
    exp = _ms1_run([(0, 10), (1, 20), (2, 30), (3, 40)])
    vals = area_under_tic_rt_quantiles(exp, 1)
    assert vals == [10.0, 20.0, 30.0, 40.0]  # min-RT scan (10) no longer lost


def test_conservation_against_total_tic():
    exp = _ms1_run([(0, 10), (1, 20), (2, 30), (3, 40), (4, 50)])
    vals = area_under_tic_rt_quantiles(exp, 1)
    assert sum(vals) == area_under_tic(exp, 1)


def test_tied_boundaries_each_spectrum_once():
    exp = _ms1_run([(0, 10), (1, 20), (1, 30), (2, 40)])
    vals = area_under_tic_rt_quantiles(exp, 1)
    assert sum(vals) == 100.0  # every spectrum counted exactly once


def test_emitted_as_single_ntuple():
    exp = _ms1_run([(0, 10), (1, 20), (2, 30), (3, 40)])
    metrics = compute_qc_metrics(exp)
    assert metrics["TIC_MS1_Area_RTQuantiles"] == [10.0, 20.0, 30.0, 40.0]
    js = build_mzqc([{
        "filename": "r.mzML",
        "metrics": metrics,
        "instrument_metadata": {},
    }])
    data = json.loads(js)
    qms = data["mzQC"]["runQualities"][0]["qualityMetrics"]
    ntuple = [m for m in qms if m.get("accession") == "MS:4000156"]
    assert len(ntuple) == 1
    assert ntuple[0]["value"] == [10.0, 20.0, 30.0, 40.0]
