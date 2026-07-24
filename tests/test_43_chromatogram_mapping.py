"""Regression tests for issue #43.

* chromatogram_point_count counts data points, not resolved peaks, and is named
  accordingly (NumberOfChromatogramDataPoints).
* Chromatogram enum types map to STABLE short keys (TIC/BPC/SRM/SIM/XIC/SIC/
  Unknown) so their counts keep their metadata, instead of dynamic keys like
  "Chromatograms_TOTAL ION CURRENT CHROMATOGRAM".
* Empty chromatograms / experiments have a defined RT-range policy.
"""
import math

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import (
    chromatogram_statistics,
    chromatogram_point_count,
    compute_qc_metrics,
    METRIC_METADATA,
)

CT = oms.ChromatogramSettings.ChromatogramType


def _chrom(ctype, rts, intens):
    ch = oms.MSChromatogram()
    ch.setChromatogramType(ctype)
    if rts:
        ch.set_peaks((np.asarray(rts, dtype=float), np.asarray(intens, dtype=float)))
    return ch


def _exp_with_chroms(chroms, spectra=True):
    exp = oms.MSExperiment()
    if spectra:
        for rt in (1.0, 2.0):
            sp = oms.MSSpectrum(); sp.setRT(rt); sp.setMSLevel(1)
            sp.set_peaks((np.array([100.0]), np.array([5.0])))
            exp.addSpectrum(sp)
    exp.setChromatograms(chroms)
    exp.updateRanges()
    return exp


def test_type_mapping_to_stable_keys():
    exp = _exp_with_chroms([
        _chrom(CT.TOTAL_ION_CURRENT_CHROMATOGRAM, [10, 20, 30], [1, 2, 3]),
        _chrom(CT.BASEPEAK_CHROMATOGRAM, [5, 40], [1, 1]),
        _chrom(CT.SELECTED_REACTION_MONITORING_CHROMATOGRAM, [1, 2], [1, 1]),
        _chrom(CT.SELECTED_ION_MONITORING_CHROMATOGRAM, [1], [1]),
        _chrom(CT.MASS_CHROMATOGRAM, [1, 2], [1, 1]),
        _chrom(CT.SELECTED_ION_CURRENT_CHROMATOGRAM, [1], [1]),
        _chrom(CT.EMISSION_CHROMATOGRAM, [1], [1]),  # -> Unknown
    ])
    stats = chromatogram_statistics(exp)
    assert stats["counts_by_type"] == {
        "TIC": 1, "BPC": 1, "SRM": 1, "SIM": 1, "XIC": 1, "SIC": 1, "Unknown": 1,
    }
    # RT range spans all finite chromatogram bounds (min 1 from SRM/SIM/...,
    # max 40 from BPC).
    assert stats["rt_range_min"] == 1.0
    assert stats["rt_range_max"] == 40.0


def test_point_count_is_data_points():
    exp = _exp_with_chroms([
        _chrom(CT.TOTAL_ION_CURRENT_CHROMATOGRAM, [10, 20, 30], [1, 2, 3]),
        _chrom(CT.BASEPEAK_CHROMATOGRAM, [5, 40], [1, 1]),
    ])
    assert chromatogram_point_count(exp) == 5  # 3 + 2 data points


def test_empty_experiment_rt_range_nan():
    exp = _exp_with_chroms([])
    stats = chromatogram_statistics(exp)
    assert stats["total_chromatograms"] == 0
    assert math.isnan(stats["rt_range_min"])
    assert math.isnan(stats["rt_range_max"])


def test_empty_chromatogram_does_not_corrupt_rt_range():
    exp = _exp_with_chroms([
        _chrom(CT.TOTAL_ION_CURRENT_CHROMATOGRAM, [10, 20], [1, 2]),
        _chrom(CT.BASEPEAK_CHROMATOGRAM, [], []),  # empty
    ])
    stats = chromatogram_statistics(exp)
    assert stats["rt_range_min"] == 10.0
    assert stats["rt_range_max"] == 20.0


def test_wired_into_compute_with_metadata():
    exp = _exp_with_chroms([
        _chrom(CT.TOTAL_ION_CURRENT_CHROMATOGRAM, [10, 20, 30], [1, 2, 3]),
    ])
    m = compute_qc_metrics(exp)
    assert m["Chromatograms_TIC"] == 1
    assert m["NumberOfChromatogramDataPoints"] == 3
    # the emitted key has registered metadata (no metadata loss)
    assert "Chromatograms_TIC" in METRIC_METADATA
    assert "NumberOfChromatogramDataPoints" in METRIC_METADATA
    assert "NumberOfChromatographicPeaks" not in METRIC_METADATA
