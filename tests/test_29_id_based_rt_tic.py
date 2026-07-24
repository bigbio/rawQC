"""Regression tests for issue #29.

rt_iqr / rt_iqr_rate (MS:4000153/154, C-2A/C-2B) and median_tic_rt_iqr /
median_tic_of_rt_range (MS:4000158/159) are ID-based PSI-MS terms but were
documented as such while operating on all raw spectra. rawQC exposes them as
ID-free proxies (no ID accession emitted) and accepts optional accepted
identifications to reproduce the ID-based definition.
"""
import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    rt_iqr,
    rt_iqr_rate,
    median_tic_rt_iqr,
    median_tic_of_rt_range,
    METRIC_ACCESSIONS,
)


def _ms1_run(rts):
    exp = oms.MSExperiment()
    for i, rt in enumerate(rts):
        sp = oms.MSSpectrum()
        sp.setRT(float(rt))
        sp.setMSLevel(1)
        sp.setNativeID(f"scan={i}")
        sp.set_peaks((np.array([500.0]), np.array([100.0])))
        exp.addSpectrum(sp)
    exp.updateRanges()
    return exp


def test_id_free_proxy_not_emitted_with_id_accession():
    # The ID-based accessions must NOT be attached to the mzML-only proxies.
    for key in ("RT_MS1_IQR", "RT_MS1_IQRRate",
                "MedianTIC_in_RT_MS1_IQR", "TIC_MS1_MedianInHalfRange"):
        assert key not in METRIC_ACCESSIONS


def test_no_identifications_uses_all_spectra():
    exp = _ms1_run([0, 10, 20, 30, 40, 50])
    # IQR of all six RTs
    q75, q25 = np.percentile([0, 10, 20, 30, 40, 50], [75, 25])
    assert rt_iqr(exp, 1) == float(q75 - q25)


def test_accepted_ids_restrict_computation():
    exp = _ms1_run([0, 10, 20, 30, 40, 50])
    accepted = {"scan=1", "scan=2", "scan=3"}  # RTs 10, 20, 30
    q75, q25 = np.percentile([10, 20, 30], [75, 25])
    assert rt_iqr(exp, 1, accepted_native_ids=accepted) == float(q75 - q25)
    # different from the all-spectra proxy
    assert rt_iqr(exp, 1, accepted_native_ids=accepted) != rt_iqr(exp, 1)


def test_tic_proxies_accept_identifications():
    exp = _ms1_run([0, 10, 20, 30, 40, 50])
    accepted = {"scan=2", "scan=3"}
    # Both must run and honour the filter without raising.
    assert not np.isnan(median_tic_rt_iqr(exp, 1))
    assert not np.isnan(median_tic_of_rt_range(exp, 1, accepted_native_ids=accepted))
    assert not np.isnan(rt_iqr_rate(exp, 1, accepted_native_ids=accepted))
