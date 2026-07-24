"""Deterministic synthetic mzML fixtures and reference provenance for the rawQC
conformance suite (issue #45).

Every fixture is built entirely in memory (no network, no external mzML files)
so the suite is reproducible and fast. The corpus deliberately covers the edge
cases the metric audit found: empty scans, missing precursors, unknown charge,
MS3, mixed peak types, irregular retention times, and non-multiple-of-four scan
counts.

Expected metric values (per algorithm) live in each issue's dedicated test file
(test_<issue>_*.py); those files are the numeric oracles. This module provides
the shared corpus and the provenance manifest so those oracles reference an
explicit source and algorithm version rather than silently mixing conflicting
C++/R/CV/newer-port semantics.
"""
from typing import List

import numpy as np
import pyopenms as oms


# ---------------------------------------------------------------------------
# Reference provenance: the exact upstream sources and versions rawQC's metric
# semantics are pinned to. Do not mix these silently.
# ---------------------------------------------------------------------------
REFERENCE_SOURCES = {
    "quameter_cpp": {
        "repo": "ProteoWizard/pwiz",
        "commit": "a09eea91209131f6aa487f7316647fc536188c19",
        "path": "pwiz_tools/Bumbershoot/quameter/quameter.cpp",
        "used_for": ["MS:4000063", "MS:4000065", "MS:4000066", "MS:4000156",
                     "MS:4000183", "MS:4000184", "MS:4000185", "MS:4000186"],
    },
    "msquality_r": {
        "repo": "tnaake/MsQuality",
        "commit": "7a883367e4bb61b2154693177096fdd45a7caa43",
        "path": "R/function_Spectra_metrics.R",
        "used_for": ["MS:4000158 (recycling partition)"],
    },
    "macproqc_python": {
        "repo": "mpc-bioinformatics/macproqc-helpers",
        "commit": "ffdb14fba8ea9e7bc081ad2bbe70e8f07315c00f",
        "path": "src/macproqc_helpers/helpers/collect_metrics_from_mzml.py",
        "used_for": ["MS:4000065", "MS:4000066", "MS:4000202"],
    },
    "smaqc_csharp": {
        "repo": "PNNL-Comp-Mass-Spec/SMAQC",
        "commit": "b1f0d1b72594bee80fe3d83b8e19ab21a33e9c0a",
        "path": "SMAQC/Measurement.cs",
        "used_for": ["MS:4000152", "MS:4000153", "MS:4000154", "MS:4000157",
                     "MS:4000158", "MS:4000159"],
    },
    "psi_ms_cv": {
        "repo": "HUPO-PSI/psi-ms-CV",
        "version": "4.1.257",
    },
}

# Declared minimum supported pyOpenMS version (see pyproject.toml).
MIN_PYOPENMS = (3, 5, 0)


def _spectrum(rt, level, mzs, intens, *, prec_mz=None, prec_int=None,
              charge=None, stype=None, faims=None, native_id=None):
    sp = oms.MSSpectrum()
    sp.setRT(float(rt))
    sp.setMSLevel(int(level))
    if native_id is not None:
        sp.setNativeID(str(native_id))
    if len(mzs):
        sp.set_peaks((np.asarray(mzs, dtype=float), np.asarray(intens, dtype=float)))
    if stype is not None:
        sp.setType(stype)
    if faims is not None:
        sp.setMetaValue("FAIMS_CV", float(faims))
    if level >= 2 and (prec_mz is not None or charge is not None or prec_int is not None):
        prec = oms.Precursor()
        if prec_mz is not None:
            prec.setMZ(float(prec_mz))
        if prec_int is not None:
            prec.setIntensity(float(prec_int))
        if charge is not None:
            prec.setCharge(int(charge))
        sp.setPrecursors([prec])
    return sp


def _exp(spectra: List[oms.MSSpectrum]) -> oms.MSExperiment:
    exp = oms.MSExperiment()
    for sp in spectra:
        exp.addSpectrum(sp)
    exp.updateRanges()
    return exp


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def empty_scans() -> oms.MSExperiment:
    """MS1 run containing zero-length (empty) scans among normal scans."""
    specs = [
        _spectrum(0, 1, [100, 200, 300], [10, 20, 30]),
        _spectrum(1, 1, [], []),                       # empty
        _spectrum(2, 1, [100, 200], [5, 5]),
        _spectrum(3, 1, [], []),                       # empty
    ]
    return _exp(specs)


def missing_precursors() -> oms.MSExperiment:
    """MS2 scans, some without any precursor set."""
    specs = [
        _spectrum(0, 1, [100], [10]),
        _spectrum(0.5, 2, [50], [5], prec_mz=400, charge=2),
        _spectrum(1.0, 2, [50], [5]),                  # no precursor
        _spectrum(1.5, 2, [50], [5], prec_mz=600, charge=3),
    ]
    return _exp(specs)


def unknown_charge() -> oms.MSExperiment:
    """MS2 scans including unknown (zero) charge states."""
    specs = [
        _spectrum(0, 1, [100], [10]),
        _spectrum(0.5, 2, [50], [5], prec_mz=400, charge=2),
        _spectrum(1.0, 2, [50], [5], prec_mz=500, charge=0),   # unknown
        _spectrum(1.5, 2, [50], [5], prec_mz=600, charge=7),   # >=6
    ]
    return _exp(specs)


def with_ms3() -> oms.MSExperiment:
    """Run with MS1, MS2 and MS3 spectra (MS3 holds the largest base peak)."""
    specs = [
        _spectrum(0, 1, [100, 200], [10, 100]),
        _spectrum(0.3, 2, [50], [200], prec_mz=400, charge=2),
        _spectrum(0.6, 3, [30], [500]),                # largest base peak
    ]
    return _exp(specs)


def mixed_peak_types() -> oms.MSExperiment:
    """MS1 run mixing profile and centroid annotated spectra."""
    ST = oms.SpectrumSettings.SpectrumType
    specs = [
        _spectrum(0, 1, [100, 101, 102], [1, 2, 1], stype=ST.CENTROID),
        _spectrum(1, 1, [100, 101, 102], [1, 2, 1], stype=ST.PROFILE),
        _spectrum(2, 1, [100, 101, 102], [1, 2, 1], stype=ST.PROFILE),
    ]
    return _exp(specs)


def irregular_rts() -> oms.MSExperiment:
    """MS1 run with irregular retention-time spacing and constant TIC."""
    specs = [_spectrum(rt, 1, [100], [10]) for rt in (0.0, 1.0, 10.0, 12.0, 40.0)]
    return _exp(specs)


def non_multiple_of_four(n: int = 5) -> oms.MSExperiment:
    """MS1 run whose scan count is not a multiple of four."""
    specs = [_spectrum(float(i), 1, [100], [float(i + 1)], native_id=f"scan={i}")
             for i in range(n)]
    return _exp(specs)


def faims_run() -> oms.MSExperiment:
    """MS1 run carrying FAIMS compensation voltages."""
    specs = [_spectrum(i, 1, [100], [10], faims=-45.0 if i % 2 == 0 else -55.0)
             for i in range(4)]
    return _exp(specs)


CORPUS = {
    "empty_scans": empty_scans,
    "missing_precursors": missing_precursors,
    "unknown_charge": unknown_charge,
    "with_ms3": with_ms3,
    "mixed_peak_types": mixed_peak_types,
    "irregular_rts": irregular_rts,
    "non_multiple_of_four_5": lambda: non_multiple_of_four(5),
    "non_multiple_of_four_7": lambda: non_multiple_of_four(7),
    "faims_run": faims_run,
}
