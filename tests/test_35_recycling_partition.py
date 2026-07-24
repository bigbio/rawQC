"""Regression tests for issue #35.

median_tic_rt_iqr claims to match MsQuality's R partition
    ind <- sort(rep(seq_len(4), length.out = n))
    Q1ToQ3 <- spectra[ind %in% c(2, 3), ]
but used contiguous ceil(n/4) blocks, which diverge whenever n % 4 != 0.

The expected selected indices below are the R partition worked out by hand for
n = 1..12; each spectrum's TIC equals its RT-ordered index, so the expected
median is simply the median of the selected indices.
"""
import math

import numpy as np
import pyopenms as oms

from rawQC.calculate_metrics import median_tic_rt_iqr

# 0-based indices selected by R's ind %in% c(2,3), derived manually.
EXPECTED_SELECTED = {
    1: [],
    2: [1],
    3: [1, 2],
    4: [1, 2],
    5: [2, 3],
    6: [2, 3, 4],
    7: [2, 3, 4, 5],
    8: [2, 3, 4, 5],
    9: [3, 4, 5, 6],
    10: [3, 4, 5, 6, 7],
    11: [3, 4, 5, 6, 7, 8],
    12: [3, 4, 5, 6, 7, 8],
}


def _run_with_tic_equal_index(n):
    exp = oms.MSExperiment()
    for i in range(n):
        sp = oms.MSSpectrum()
        sp.setRT(float(i))
        sp.setMSLevel(1)
        # TIC == i (use i+1 peaks? no: single peak intensity i; i=0 -> empty ok)
        sp.set_peaks((np.array([500.0]), np.array([float(i)])))
        exp.addSpectrum(sp)
    exp.updateRanges()
    return exp


def test_recycling_partition_matches_R_for_n_1_to_12():
    for n, sel in EXPECTED_SELECTED.items():
        exp = _run_with_tic_equal_index(n)
        got = median_tic_rt_iqr(exp, 1)
        if not sel:
            assert math.isnan(got), f"n={n} expected NaN, got {got}"
        else:
            expected = float(np.median(sel))  # tic value == index
            assert got == expected, f"n={n}: got {got}, expected {expected}"


def test_n5_differs_from_old_contiguous_partition():
    # R selects indices [2,3] (median 2.5); the old ceil-block code selected
    # [2,3,4] (median 3.0). Assert the corrected value.
    exp = _run_with_tic_equal_index(5)
    assert median_tic_rt_iqr(exp, 1) == 2.5
