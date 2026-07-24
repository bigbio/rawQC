"""Regression tests for issue #27.

The acquisition-range accessions were swapped (MzRange used MS:4000070, the RT
term; RtRange used MS:4000069, the precursor m/z term). An MS1 precursor m/z
range was also emitted even though MS1 spectra have no precursor. Ranges are now
emitted as two-value [min, max] n-tuples with the correct accessions and no MS1
precursor range.
"""
import json

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import compute_qc_metrics, build_mzqc, METRIC_METADATA


def _spec(rt, level, prec_mz=None):
    sp = oms.MSSpectrum()
    sp.setRT(float(rt))
    sp.setMSLevel(level)
    sp.set_peaks((np.array([100.0, 200.0]), np.array([5.0, 6.0])))
    if level >= 2 and prec_mz is not None:
        prec = oms.Precursor(); prec.setMZ(float(prec_mz)); prec.setCharge(2)
        sp.setPrecursors([prec])
    return sp


def _exp():
    exp = oms.MSExperiment()
    exp.addSpectrum(_spec(0.0, 1))
    exp.addSpectrum(_spec(1.0, 2, prec_mz=400.0))
    exp.addSpectrum(_spec(2.0, 1))
    exp.addSpectrum(_spec(3.0, 2, prec_mz=800.0))
    exp.updateRanges()
    return exp


def test_accessions_are_correct_and_not_swapped():
    assert METRIC_METADATA["MzRange_MS2"]["accession"] == "MS:4000069"
    assert METRIC_METADATA["RtRange_MS1"]["accession"] == "MS:4000070"
    assert METRIC_METADATA["RtRange_MS2"]["accession"] == "MS:4000070"


def test_no_ms1_precursor_range_emitted():
    m = compute_qc_metrics(_exp())
    assert "MzRange_MS1" not in m
    assert "MzRange_MS1_Min" not in m


def test_ranges_are_two_value_ntuples():
    m = compute_qc_metrics(_exp())
    assert m["MzRange_MS2"] == [400.0, 800.0]
    assert m["RtRange_MS1"] == [0.0, 2.0]
    assert m["RtRange_MS2"] == [1.0, 3.0]


def test_emitted_ntuple_shapes_and_accessions():
    m = compute_qc_metrics(_exp())
    qms = json.loads(build_mzqc([{"filename": "r.mzML", "metrics": m,
                                  "instrument_metadata": {}}]))["mzQC"]["runQualities"][0]["qualityMetrics"]
    by_name = {q["name"]: q for q in qms}
    assert by_name["MzRange_MS2"]["accession"] == "MS:4000069"
    assert by_name["MzRange_MS2"]["value"] == [400.0, 800.0]
    assert by_name["RtRange_MS1"]["accession"] == "MS:4000070"
    assert by_name["RtRange_MS1"]["value"] == [0.0, 2.0]
