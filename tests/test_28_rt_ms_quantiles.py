"""Regression tests for issue #28.

MS:4000184/MS:4000185 are n-tuples of four RT *interval* widths (the original
QuaMeter RT-MS-Q1..Q4), normalized by the MS level's own acquisition duration.
rawQC previously returned four cumulative endpoints normalized by the whole-
experiment duration with a non-reference index partition. The four intervals now
sum to 1.0 and are emitted as a single n-tuple.
"""
import json

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    rt_over_ms_quantiles,
    compute_qc_metrics,
    build_mzqc,
)


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


def test_uniform_intervals_are_quarters():
    # evenly spaced RTs -> percentiles at 25/50/75% of the span -> four equal
    # intervals of 0.25 each.
    exp = _run(list(np.linspace(0.0, 100.0, 101)))
    vals = rt_over_ms_quantiles(exp, 1)
    assert vals == [0.25, 0.25, 0.25, 0.25]
    assert sum(vals) == 1.0


def test_intervals_sum_to_one_general():
    exp = _run([0, 2, 5, 9, 14, 20, 27, 35])
    vals = rt_over_ms_quantiles(exp, 1)
    assert abs(sum(vals) - 1.0) < 1e-12


def test_non_multiple_of_four_counts():
    for n in (5, 6, 7, 9, 11):
        exp = _run(list(range(n)))
        vals = rt_over_ms_quantiles(exp, 1)
        assert len(vals) == 4
        assert abs(sum(vals) - 1.0) < 1e-12


def test_emitted_as_single_ntuple():
    exp = _run(list(np.linspace(0.0, 100.0, 101)))
    m = compute_qc_metrics(exp)
    assert m["RT_MS1_Quantiles"] == [0.25, 0.25, 0.25, 0.25]
    js = build_mzqc([{"filename": "r.mzML", "metrics": m, "instrument_metadata": {}}])
    qms = json.loads(js)["mzQC"]["runQualities"][0]["qualityMetrics"]
    nt = [q for q in qms if q.get("accession") == "MS:4000184"]
    assert len(nt) == 1 and nt[0]["value"] == [0.25, 0.25, 0.25, 0.25]
