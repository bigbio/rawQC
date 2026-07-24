"""Regression tests for issue #40.

median_precursor_mz (MS:4000152) and extent_identified_precursor_intensity
(MS:4000157) are ID-based terms but were documented as such while computing over
all MS2 precursors. rawQC exposes them as ID-free proxies (no ID accession
emitted) and accepts optional accepted identifications.
"""
import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    median_precursor_mz,
    extent_identified_precursor_intensity,
    METRIC_ACCESSIONS,
)


def _ms2_run(mz_int_pairs):
    exp = oms.MSExperiment()
    for i, (mz, inten) in enumerate(mz_int_pairs):
        sp = oms.MSSpectrum()
        sp.setRT(float(i))
        sp.setMSLevel(2)
        sp.setNativeID(f"scan={i}")
        sp.set_peaks((np.array([100.0]), np.array([10.0])))
        prec = oms.Precursor()
        prec.setMZ(float(mz))
        prec.setIntensity(float(inten))
        prec.setCharge(2)
        sp.setPrecursors([prec])
        exp.addSpectrum(sp)
    exp.updateRanges()
    return exp


def test_no_id_accession_emitted():
    for key in ("PrecursorMz_MS2_Median", "ExtentPrecursorIntensity_95over5_MS2"):
        assert key not in METRIC_ACCESSIONS


def test_median_precursor_mz_all_vs_accepted():
    exp = _ms2_run([(400, 100), (500, 200), (600, 300), (700, 400)])
    assert median_precursor_mz(exp, 2) == 550.0
    accepted = {"scan=0", "scan=1"}  # mz 400, 500
    assert median_precursor_mz(exp, 2, accepted_native_ids=accepted) == 450.0


def test_extent_all_vs_accepted():
    exp = _ms2_run([(400, 100), (500, 200), (600, 300), (700, 400)])
    all_extent = extent_identified_precursor_intensity(exp, 2)
    q5, q95 = np.quantile([100, 200, 300, 400], [0.05, 0.95])
    assert all_extent == float(q95 / q5)
    accepted = {"scan=2", "scan=3"}  # intensities 300, 400
    sub = extent_identified_precursor_intensity(exp, 2, accepted_native_ids=accepted)
    assert sub != all_extent
