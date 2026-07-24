"""Regression tests for issue #41.

MS:4000063 precursor-charge fractions were wrong: unknown/missing/zero charges
were dropped, each fraction was divided by the number of *known* charges (not all
MS2 scans), six repeated scalar metrics were emitted instead of one table, and
the terminal bin was labelled 5+ while the code binned >=6. rawQC now emits one
table over all MS2 scans with an explicit unknown bin and a >=6 terminal bin.
"""
import json

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import charge_metrics, compute_qc_metrics, build_mzqc


def _ms2(charge):
    sp = oms.MSSpectrum()
    sp.setRT(1.0)
    sp.setMSLevel(2)
    sp.set_peaks((np.array([100.0]), np.array([10.0])))
    prec = oms.Precursor()
    prec.setMZ(500.0)
    prec.setCharge(int(charge))  # 0 -> unknown
    sp.setPrecursors([prec])
    return sp


def _exp(charges):
    exp = oms.MSExperiment()
    for z in charges:
        exp.addSpectrum(_ms2(z))
    exp.updateRanges()
    return exp


def test_denominator_is_all_ms2_scans_and_unknown_kept():
    # 10 MS2 scans: 2x z=2, 1x z=3, 1x z=7 (>=6 bin), 6x unknown (z=0)
    exp = _exp([2, 2, 3, 7, 0, 0, 0, 0, 0, 0])
    tbl = charge_metrics(exp, 2)["MS2_PrecursorCharge_Fractions"]
    assert tbl["charge_state"] == ["1", "2", "3", "4", "5", ">=6", "unknown"]
    assert tbl["count"] == [0, 2, 1, 0, 0, 1, 6]
    # denominator is 10 (all MS2 scans), not 4 (known charges)
    assert tbl["fraction"] == [0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.6]
    assert abs(sum(tbl["fraction"]) - 1.0) < 1e-12


def test_charge_6_goes_to_terminal_bin():
    exp = _exp([6, 6, 2])
    tbl = charge_metrics(exp, 2)["MS2_PrecursorCharge_Fractions"]
    assert tbl["count"][5] == 2   # >=6 bin holds the two z=6
    assert tbl["count"][1] == 1   # z=2


def test_emitted_as_single_table_metric():
    exp = _exp([2, 2, 3, 0])
    m = compute_qc_metrics(exp)
    # No repeated scalar charge metrics anymore.
    assert "MS2-PrecZ-1" not in m
    qms = json.loads(build_mzqc([{"filename": "r.mzML", "metrics": m,
                                  "instrument_metadata": {}}]))["mzQC"]["runQualities"][0]["qualityMetrics"]
    tables = [q for q in qms if q.get("accession") == "MS:4000063"]
    assert len(tables) == 1
    val = tables[0]["value"]
    assert val["charge_state"] == ["1", "2", "3", "4", "5", ">=6", "unknown"]
    assert val["fraction"] == [0.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.25]


def test_negative_charge_goes_to_unknown_and_sum_is_one():
    # A non-physical negative charge must not silently vanish: it lands in the
    # unknown bin so the fractions still sum to 1.0.
    exp = _exp([2, -3, 0])  # one valid (2), one negative, one zero(unknown)
    tbl = charge_metrics(exp, 2)["MS2_PrecursorCharge_Fractions"]
    assert tbl["count"][1] == 1                    # z=2
    assert tbl["count"][-1] == 2                    # negative + zero -> unknown
    assert abs(sum(tbl["fraction"]) - 1.0) < 1e-12
