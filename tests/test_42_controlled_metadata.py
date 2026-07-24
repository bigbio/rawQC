"""Regression tests for issue #42.

Instrument/acquisition metadata (polarity, activation, analyzer, FAIMS) was
mixed into QC metrics with invalid semantics: scan-polarity terms MS:1000129/130
and the FAIMS voltage term MS:1001581 were reused as QC-metric accessions,
analyzer/activation keys were hard-coded or dynamic without metadata. rawQC now
represents them as valid custom metrics (tables / n-tuples) with no invalid
accession reuse and no fixed key slots.
"""
import json

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    activation_method_statistics,
    mass_analyzer_info,
    faims_compensation_voltages,
    compute_qc_metrics,
    build_mzqc,
    METRIC_METADATA,
)


def _ms1(rt, faims=None):
    sp = oms.MSSpectrum(); sp.setRT(float(rt)); sp.setMSLevel(1)
    sp.set_peaks((np.array([100.0, 200.0]), np.array([5.0, 6.0])))
    ins = sp.getInstrumentSettings(); ins.setPolarity(oms.IonSource.Polarity.POSITIVE)
    sp.setInstrumentSettings(ins)
    if faims is not None:
        # OpenMS stores a FAIMS CV as drift time with the FAIMS unit (see #23).
        sp.setDriftTime(float(faims))
        sp.setDriftTimeUnit(oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE)
    return sp


def _ms2(rt):
    sp = oms.MSSpectrum(); sp.setRT(float(rt)); sp.setMSLevel(2)
    sp.set_peaks((np.array([100.0]), np.array([5.0])))
    prec = oms.Precursor(); prec.setMZ(500.0); prec.setCharge(2)
    prec.setActivationMethods({oms.Precursor.ActivationMethod.HCD})
    sp.setPrecursors([prec])
    return sp


def _exp():
    exp = oms.MSExperiment()
    faims_cycle = [-45.0, -50.0, -55.0]
    for i in range(6):
        exp.addSpectrum(_ms1(10 + i, faims=faims_cycle[i % 3]))
        exp.addSpectrum(_ms2(10.5 + i))
    inst = exp.getInstrument()
    a0 = oms.MassAnalyzer(); a0.setType(oms.MassAnalyzer.AnalyzerType.ORBITRAP); a0.setResolution(60000.0)
    a1 = oms.MassAnalyzer(); a1.setType(oms.MassAnalyzer.AnalyzerType.IT); a1.setResolution(0.0)
    inst.setMassAnalyzers([a0, a1])
    exp.setInstrument(inst)
    exp.updateRanges()
    return exp


def test_analyzer_table():
    tbl = mass_analyzer_info(_exp())
    assert tbl["index"] == [0, 1]
    assert tbl["type"] == ["ORBITRAP", "IT"]
    assert tbl["resolution"] == [60000.0, None]


def test_activation_table():
    tbl = activation_method_statistics(_exp())
    assert tbl["ms_level"] == [2]
    assert tbl["method"] == ["HCD"]
    assert tbl["count"] == [6]


def test_faims_values_range_count():
    f = faims_compensation_voltages(_exp())
    assert f["FAIMS_CV_Count"] == 3
    # distinct sorted values differ from the [min, max] range (3 values)
    assert f["FAIMS_CV_Values"] == [-55.0, -50.0, -45.0]
    assert f["FAIMS_CV_Range"] == [-55.0, -45.0]
    assert f["FAIMS_CV_Values"] != f["FAIMS_CV_Range"]


def test_polarity_accessions_not_reused():
    for key in ("Polarity_MS1_positive", "Polarity_MS1_negative",
                "Polarity_MS2_positive", "Polarity_MS2_negative"):
        assert METRIC_METADATA[key]["accession"] is None


def test_no_invalid_accession_reuse_in_emitted_mzqc():
    m = compute_qc_metrics(_exp())
    # tables/lists surfaced as single metrics; no fixed key slots
    assert "MassAnalyzers" in m and "MassAnalyzer_0_Type" not in m
    assert "ActivationMethods" in m and "MS2_ActivationMethod_HCD" not in m
    qms = json.loads(build_mzqc([{"filename": "r.mzML", "metrics": m,
                                  "instrument_metadata": {}}]))["mzQC"]["runQualities"][0]["qualityMetrics"]
    bad = {"MS:1000129", "MS:1000130", "MS:1001581"}
    assert not [q for q in qms if q.get("accession") in bad]
    by_name = {q["name"]: q for q in qms}
    assert by_name["MassAnalyzers"]["value"]["type"] == ["ORBITRAP", "IT"]
    assert by_name["ActivationMethods"]["value"]["method"] == ["HCD"]
    assert by_name["FAIMS_CV_Range"]["value"] == [-55.0, -45.0]
