"""Regression tests for issue #31.

area_under_tic_rt_quantiles excluded the minimum-RT spectrum (first bin used
`rts > q0`), and compute_qc_metrics exposed only Q1-Q3, discarding Q4. rawQC now
assigns every finite-RT spectrum to exactly one quartile bin (so the four values
conserve the whole-run TIC) and emits all four as one MS:4000156 n-tuple.
"""
import math
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
    # Under the integrated area-under-curve contract (#30 integral + #31
    # completeness) the four quartile areas are trapezoidal integrals, the
    # minimum-RT scan contributes to the first quartile (no longer lost), and all
    # four values are returned and conserve the whole-run integral.
    exp = _ms1_run([(0, 10), (1, 20), (2, 30), (3, 40)])
    vals = area_under_tic_rt_quantiles(exp, 1)
    assert len(vals) == 4 and all(math.isfinite(v) for v in vals)
    assert vals[0] > 0  # minimum-RT region carries signal (not dropped)
    assert abs(sum(vals) - area_under_tic(exp, 1)) < 1e-9


def test_conservation_against_total_tic():
    exp = _ms1_run([(0, 10), (1, 20), (2, 30), (3, 40), (4, 50)])
    vals = area_under_tic_rt_quantiles(exp, 1)
    assert sum(vals) == area_under_tic(exp, 1)


def test_tied_boundaries_conserve_integral():
    exp = _ms1_run([(0, 10), (1, 20), (1, 30), (2, 40)])
    vals = area_under_tic_rt_quantiles(exp, 1)
    assert len(vals) == 4
    assert abs(sum(vals) - area_under_tic(exp, 1)) < 1e-9  # conserved, no double count


def test_emitted_as_single_ntuple():
    exp = _ms1_run([(0, 10), (1, 20), (2, 30), (3, 40)])
    metrics = compute_qc_metrics(exp)
    expected = area_under_tic_rt_quantiles(exp, 1)
    assert len(expected) == 4
    assert metrics["TIC_MS1_Area_RTQuantiles"] == expected
    js = build_mzqc([{
        "filename": "r.mzML",
        "metrics": metrics,
        "instrument_metadata": {},
    }])
    data = json.loads(js)
    qms = data["mzQC"]["runQualities"][0]["qualityMetrics"]
    ntuple = [m for m in qms if m.get("accession") == "MS:4000156"]
    assert len(ntuple) == 1
    assert ntuple[0]["value"] == expected
