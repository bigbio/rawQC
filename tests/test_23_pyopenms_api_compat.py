"""Regression tests for issue #23.

rawQC declares ``pyopenms>=3.4.0`` but previously called ``*ToString`` binding
helpers and ``FAIMSHelper`` that do not exist in that version, so polarity,
peak-type, activation, analyzer, and FAIMS handling either silently degraded or
aborted ``compute_qc_metrics``. These tests build a synthetic experiment that
exercises every one of those paths and asserts the metrics are computed without
raising and without silently turning binding gaps into valid-looking values.
"""
import math

import numpy as np
import pyopenms as oms
import pytest

from rawQC.calculate_metrics import (
    _activation_method_to_str,
    _analyzer_type_to_str,
    _faims_compensation_voltages,
    _polarity_to_str,
    _spectrum_type_to_str,
    compute_qc_metrics,
    peak_type_statistics,
    activation_method_statistics,
    mass_analyzer_info,
)


def _ms1(rt, mzs, intens, polarity=oms.IonSource.Polarity.POSITIVE,
         stype=oms.SpectrumSettings.SpectrumType.PROFILE, faims_cv=None):
    sp = oms.MSSpectrum()
    sp.setRT(float(rt))
    sp.setMSLevel(1)
    sp.set_peaks((np.asarray(mzs, dtype=float), np.asarray(intens, dtype=float)))
    ins = sp.getInstrumentSettings()
    ins.setPolarity(polarity)
    sp.setInstrumentSettings(ins)
    sp.setType(stype)
    if faims_cv is not None:
        sp.setMetaValue("FAIMS_CV", float(faims_cv))
    return sp


def _ms2(rt, prec_mz, prec_int, charge, activation, polarity=oms.IonSource.Polarity.POSITIVE):
    sp = oms.MSSpectrum()
    sp.setRT(float(rt))
    sp.setMSLevel(2)
    sp.set_peaks((np.array([100.0, 110.0, 120.0]), np.array([5.0, 6.0, 7.0])))
    ins = sp.getInstrumentSettings()
    ins.setPolarity(polarity)
    sp.setInstrumentSettings(ins)
    prec = oms.Precursor()
    prec.setMZ(float(prec_mz))
    prec.setIntensity(float(prec_int))
    prec.setCharge(int(charge))
    prec.setActivationMethods({activation})
    sp.setPrecursors([prec])
    return sp


def _experiment_with_two_analyzers():
    exp = oms.MSExperiment()
    for i in range(6):
        exp.addSpectrum(_ms1(10.0 + i, [100.0 + i, 200.0 + i, 300.0], [10.0, 20.0, 5.0],
                             faims_cv=-45.0 if i % 2 == 0 else -55.0))
        exp.addSpectrum(_ms2(10.5 + i, 500.0 + i, 1000.0, 2, oms.Precursor.ActivationMethod.HCD))
    # Attach two mass analyzers to the instrument.
    inst = exp.getInstrument()
    a0 = oms.MassAnalyzer(); a0.setType(oms.MassAnalyzer.AnalyzerType.ORBITRAP); a0.setResolution(60000.0)
    a1 = oms.MassAnalyzer(); a1.setType(oms.MassAnalyzer.AnalyzerType.IT); a1.setResolution(0.0)
    inst.setMassAnalyzers([a0, a1])
    exp.setInstrument(inst)
    exp.updateRanges()
    return exp


def test_enum_string_converters():
    assert _polarity_to_str(oms.IonSource.Polarity.POSITIVE) == "positive"
    assert _polarity_to_str(oms.IonSource.Polarity.NEGATIVE) == "negative"
    assert _polarity_to_str(oms.IonSource.Polarity.POLNULL) == "unknown"

    assert _spectrum_type_to_str(oms.SpectrumSettings.SpectrumType.CENTROID) == "centroid"
    assert _spectrum_type_to_str(oms.SpectrumSettings.SpectrumType.PROFILE) == "profile"
    assert _spectrum_type_to_str(oms.SpectrumSettings.SpectrumType.UNKNOWN) == "unknown"

    assert _activation_method_to_str(oms.Precursor.ActivationMethod.HCD) == "HCD"
    assert _activation_method_to_str(oms.Precursor.ActivationMethod.CID) == "CID"

    assert _analyzer_type_to_str(oms.MassAnalyzer.AnalyzerType.ORBITRAP) == "ORBITRAP"
    assert _analyzer_type_to_str(oms.MassAnalyzer.AnalyzerType.TOF) == "TOF"


def test_faims_voltages_extracted():
    exp = _experiment_with_two_analyzers()
    cvs = _faims_compensation_voltages(exp)
    assert cvs == [-55.0, -45.0]


def test_peak_type_and_activation_and_analyzer_paths():
    exp = _experiment_with_two_analyzers()
    pt = peak_type_statistics(exp)
    assert pt["MS1_PeakType_Annotated"] == "profile"
    am = activation_method_statistics(exp)
    assert am.get("MS2_ActivationMethod_HCD") == 6
    ma = mass_analyzer_info(exp)
    assert ma["MassAnalyzer_0_Type"] == "ORBITRAP"
    assert ma["MassAnalyzer_1_Type"] == "IT"
    assert ma["MassAnalyzer_0_Resolution"] == 60000.0


def test_compute_qc_metrics_runs_end_to_end():
    """The whole pipeline must run on pyopenms 3.4.0 without raising."""
    exp = _experiment_with_two_analyzers()
    metrics = compute_qc_metrics(exp)
    # Polarity resolved (not silently 'unknown').
    assert metrics["Polarity_MS1_positive"] == 6
    assert metrics["Polarity_MS1_unknown"] == 0
    # FAIMS collected.
    assert metrics["FAIMS_CV_Count"] == 2
    assert metrics["FAIMS_CV_Min"] == -55.0
    assert metrics["FAIMS_CV_Max"] == -45.0
    # Analyzer info present.
    assert metrics["MassAnalyzer_0_Type"] == "ORBITRAP"
    # Sanity: a core numeric metric is finite.
    assert math.isfinite(metrics["ChromatographyDuration"])
