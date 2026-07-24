"""Cross-language / mzQC conformance suite (issue #45).

This is the shared conformance harness. It runs the whole deterministic corpus
(tests/synthetic.py) through compute_qc_metrics + build_mzqc on the declared
minimum pyOpenMS version and validates the generated mzQC against the official
JSON Schema and semantic CV/value-shape rules.

Per-metric numeric oracles (the exact expected values for each audited metric,
with their source/algorithm version) live in the dedicated per-issue test files
(test_23..test_44). This file guarantees the corpus exists, builds, runs on the
minimum dependency, and yields schema- and semantically-valid output.
"""
import json
import os
import re

import numpy as np
import pyopenms as oms
import pytest

from rawQC.calculate_metrics import compute_qc_metrics, build_mzqc
from synthetic import CORPUS, REFERENCE_SOURCES, MIN_PYOPENMS

ACCESSION_RE = re.compile(r"^[A-Z]+:[A-Z0-9]+$")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "data", "mzqc_schema.json")


def _version_tuple(v):
    parts = re.findall(r"\d+", v)
    return tuple(int(p) for p in parts[:3])


def test_declared_minimum_pyopenms_is_met():
    got = _version_tuple(oms.__version__)
    assert got >= MIN_PYOPENMS, f"pyopenms {oms.__version__} < declared minimum {MIN_PYOPENMS}"


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_corpus_computes_without_error(name):
    exp = CORPUS[name]()
    metrics = compute_qc_metrics(exp)
    assert isinstance(metrics, dict) and metrics


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_corpus_emits_schema_valid_mzqc(name):
    import jsonschema
    with open(SCHEMA_PATH) as fh:
        schema = json.load(fh)
    exp = CORPUS[name]()
    js = build_mzqc([{
        "filename": f"{name}.mzML",
        "metrics": compute_qc_metrics(exp),
        "instrument_metadata": {"Instrument model name": "synthetic"},
    }])
    jsonschema.validate(instance=json.loads(js), schema=schema)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_corpus_metrics_are_semantically_valid(name):
    exp = CORPUS[name]()
    data = json.loads(build_mzqc([{
        "filename": f"{name}.mzML",
        "metrics": compute_qc_metrics(exp),
        "instrument_metadata": {},
    }]))
    for run in data["mzQC"]["runQualities"]:
        for qm in run["qualityMetrics"]:
            assert qm.get("name"), "metric without a name"
            acc = qm.get("accession")
            assert acc and ACCESSION_RE.match(acc), f"invalid accession {acc!r}"
            # value shapes: scalar, array (n-tuple) or object (table); JSON-safe
            val = qm.get("value")
            assert val is None or isinstance(val, (int, float, str, list, dict))


def test_faims_fixture_actually_exercises_faims():
    # The corpus claims to cover FAIMS; make sure the fixture is not inert (the
    # metric must actually be emitted, not silently absent).
    metrics = compute_qc_metrics(CORPUS["faims_run"]())
    assert metrics.get("FAIMS_CV_Count", 0) >= 1
    assert metrics.get("FAIMS_CV_Values")  # the voltages are actually collected


def test_reference_manifest_is_explicit():
    # Every audited reference source is pinned to a commit/version so expected
    # values are never silently mixed across conflicting implementations.
    for key in ("quameter_cpp", "msquality_r", "macproqc_python", "smaqc_csharp", "psi_ms_cv"):
        assert key in REFERENCE_SOURCES
    assert REFERENCE_SOURCES["psi_ms_cv"]["version"]
    for k in ("quameter_cpp", "msquality_r", "macproqc_python", "smaqc_csharp"):
        assert len(REFERENCE_SOURCES[k]["commit"]) == 40
