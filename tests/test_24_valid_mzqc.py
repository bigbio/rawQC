"""Regression tests for issue #24.

build_mzqc must emit schema- and semantically-valid mzQC:
* every quality metric / cvParameter has a valid accession (^[A-Z]+:[A-Z0-9]+$)
  and a name -- custom metrics get a deterministic LOCAL:* accession backed by a
  declared local CV, never a null accession;
* AnalysisSoftware carries an accession;
* instrument/input metadata is in the metadata structure (inputFile
  fileProperties), not appended as quality metrics;
* the PSI-MS CV version is recorded (not a stale hard-coded 4.1.7).

The generated file is validated against the official mzQC JSON Schema (vendored
in tests/data) and by semantic checks.
"""
import json
import os
import re

import numpy as np
import pyopenms as oms
import pytest

from rawQC.calculate_metrics import (
    compute_qc_metrics,
    build_mzqc,
    PSI_MS_CV_VERSION,
)

ACCESSION_RE = re.compile(r"^[A-Z]+:[A-Z0-9]+$")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "data", "mzqc_schema.json")


def _exp():
    exp = oms.MSExperiment()
    for i in range(6):
        sp = oms.MSSpectrum(); sp.setRT(float(i)); sp.setMSLevel(1)
        sp.set_peaks((np.array([100.0, 200.0]), np.array([5.0, 6.0])))
        exp.addSpectrum(sp)
        s2 = oms.MSSpectrum(); s2.setRT(i + 0.5); s2.setMSLevel(2)
        s2.set_peaks((np.array([100.0]), np.array([5.0])))
        prec = oms.Precursor(); prec.setMZ(500.0); prec.setCharge(2)
        s2.setPrecursors([prec])
        exp.addSpectrum(s2)
    inst = exp.getInstrument(); inst.setName("Q Exactive")
    exp.setInstrument(inst)
    exp.updateRanges()
    return exp


def _mzqc():
    exp = _exp()
    metrics = compute_qc_metrics(exp)
    instrument_meta = {"Instrument model name": "Q Exactive", "Manufacturer": "Thermo"}
    return json.loads(build_mzqc([{
        "filename": "sample.mzML",
        "metrics": metrics,
        "instrument_metadata": instrument_meta,
    }]))


def test_every_metric_has_valid_accession_and_name():
    data = _mzqc()
    for run in data["mzQC"]["runQualities"]:
        for qm in run["qualityMetrics"]:
            assert qm.get("name"), "metric missing name"
            acc = qm.get("accession")
            assert acc and ACCESSION_RE.match(acc), f"bad accession {acc!r} for {qm.get('name')}"


def test_analysis_software_has_accession():
    data = _mzqc()
    sw = data["mzQC"]["runQualities"][0]["metadata"]["analysisSoftware"][0]
    assert ACCESSION_RE.match(sw["accession"])


def test_instrument_metadata_in_metadata_not_metrics():
    data = _mzqc()
    run = data["mzQC"]["runQualities"][0]
    # Not appended as quality metrics.
    assert not [q for q in run["qualityMetrics"] if q["name"].startswith("Instrument ")]
    props = run["metadata"]["inputFiles"][0].get("fileProperties", [])
    names = {p["name"] for p in props}
    assert "instrument model" in names  # mapped to MS:1000031
    for p in props:
        assert ACCESSION_RE.match(p["accession"])


def test_cv_version_recorded_and_local_cv_present():
    data = _mzqc()
    cvs = data["mzQC"]["controlledVocabularies"]
    versions = {cv.get("version") for cv in cvs}
    assert PSI_MS_CV_VERSION in versions
    assert PSI_MS_CV_VERSION != "4.1.7"
    # local CV declared for the LOCAL:* accessions
    uris = {cv.get("uri") for cv in cvs}
    assert any("rawQC" in (u or "") for u in uris)


def test_validates_against_official_json_schema():
    import jsonschema
    with open(SCHEMA_PATH) as fh:
        schema = json.load(fh)
    jsonschema.validate(instance=_mzqc(), schema=schema)
