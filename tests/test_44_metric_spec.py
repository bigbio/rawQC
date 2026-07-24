"""Regression tests for issue #44.

The metadata registry, computed keys, and presentation order had drifted:
a stale MS2_ActivationMethod_0 order entry, an MS_Run_Duration metric registered
but never computed, and dynamic keys only partially represented. rawQC now has a
single authoritative spec (METRIC_METADATA + METRIC_ORDER + dynamic patterns) and
validate_metric_registry() that fails when a computed metric lacks metadata or a
registered/ordered metric cannot be produced.
"""
import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    METRIC_METADATA,
    METRIC_ORDER,
    compute_qc_metrics,
    validate_metric_registry,
)


def _rich_exp():
    exp = oms.MSExperiment()
    for i in range(6):
        m1 = oms.MSSpectrum(); m1.setRT(float(i)); m1.setMSLevel(1)
        m1.set_peaks((np.arange(100.0, 115.0), np.full(15, 5.0)))
        m1.setMetaValue("FAIMS_CV", -45.0 if i % 2 == 0 else -55.0)
        exp.addSpectrum(m1)
        m2 = oms.MSSpectrum(); m2.setRT(i + 0.3); m2.setMSLevel(2)
        m2.set_peaks((np.array([100.0]), np.array([5.0])))
        prec = oms.Precursor(); prec.setMZ(500.0); prec.setCharge(2)
        prec.setActivationMethods({oms.Precursor.ActivationMethod.HCD})
        m2.setPrecursors([prec])
        exp.addSpectrum(m2)
        m3 = oms.MSSpectrum(); m3.setRT(i + 0.6); m3.setMSLevel(3)
        m3.set_peaks((np.array([50.0]), np.array([3.0])))
        exp.addSpectrum(m3)
    # chromatogram + analyzers
    ch = oms.MSChromatogram()
    ch.setChromatogramType(oms.ChromatogramSettings.ChromatogramType.TOTAL_ION_CURRENT_CHROMATOGRAM)
    ch.set_peaks((np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])))
    exp.setChromatograms([ch])
    inst = exp.getInstrument()
    a0 = oms.MassAnalyzer(); a0.setType(oms.MassAnalyzer.AnalyzerType.ORBITRAP); a0.setResolution(60000.0)
    inst.setMassAnalyzers([a0])
    exp.setInstrument(inst)
    exp.updateRanges()
    return exp


def test_no_dead_entries():
    assert "MS_Run_Duration" not in METRIC_METADATA
    assert "MS2_ActivationMethod_0" not in METRIC_ORDER


def test_order_is_subset_of_metadata_and_unique():
    missing = [k for k in METRIC_ORDER if k not in METRIC_METADATA]
    assert missing == []
    assert len(METRIC_ORDER) == len(set(METRIC_ORDER))


def test_registry_validation_has_no_violations():
    computed = compute_qc_metrics(_rich_exp())
    problems = validate_metric_registry(computed)
    assert problems["uncovered"] == [], f"computed metrics without metadata: {problems['uncovered']}"
    assert problems["ordered_missing_metadata"] == []
    assert problems["ordered_not_computed"] == []
    assert problems["registered_orphans"] == [], f"dead registrations: {problems['registered_orphans']}"


def test_validation_flags_an_uncovered_metric():
    # A metric name that is neither registered nor a known dynamic family must be
    # reported, proving the guard actually catches drift.
    problems = validate_metric_registry({"TotallyBogusMetric_xyz": 1})
    assert "TotallyBogusMetric_xyz" in problems["uncovered"]


def test_validation_flags_registered_but_not_produced():
    # A registered/ordered metric that a run fails to produce must be flagged in
    # BOTH the ordered-not-computed and registered-orphans categories.
    computed = compute_qc_metrics(_rich_exp())
    del computed["ChromatographyDuration"]
    problems = validate_metric_registry(computed)
    assert "ChromatographyDuration" in problems["ordered_not_computed"]
    assert "ChromatographyDuration" in problems["registered_orphans"]


def test_validation_flags_stale_dynamic_looking_order_entry(monkeypatch):
    # Reintroducing the stale "MS2_ActivationMethod_0" order entry (which matches
    # a dynamic pattern) must still be caught, not silently excluded.
    import importlib
    cm = importlib.import_module("rawQC.calculate_metrics")
    monkeypatch.setattr(cm, "METRIC_ORDER", cm.METRIC_ORDER + ["MS2_ActivationMethod_0"])
    problems = cm.validate_metric_registry(compute_qc_metrics(_rich_exp()))
    assert "MS2_ActivationMethod_0" in problems["ordered_not_computed"]
