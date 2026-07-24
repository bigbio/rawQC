"""Regression tests for issue #34.

Two implicit numeric contracts are made explicit:
1. ms_signal_10x_change validates `change`, returns an integer count (0, not
   NaN, for < 2 spectra), and has a defined policy for zero/non-finite TIC.
2. TIC_MS1_CV / TIC_MS2_CV use the sample standard deviation (ddof=1),
   consistent with the precursor-intensity SD.
"""
import numpy as np
import pyopenms as oms
import pytest

from rawQC.calculate_metrics import ms_signal_10x_change, compute_qc_metrics


def _ms1_run(tic_values):
    exp = oms.MSExperiment()
    for i, tic in enumerate(tic_values):
        sp = oms.MSSpectrum()
        sp.setRT(float(i))
        sp.setMSLevel(1)
        if tic > 0:
            sp.set_peaks((np.array([500.0]), np.array([float(tic)])))
        # tic == 0 -> leave empty (defaultArrayLength 0 -> TIC 0)
        exp.addSpectrum(sp)
    exp.updateRanges()
    return exp


def test_change_argument_validated():
    exp = _ms1_run([1.0, 10.0])
    with pytest.raises(ValueError):
        ms_signal_10x_change(exp, change="up", ms_level=1)


def test_fewer_than_two_spectra_is_zero_int():
    exp = _ms1_run([5.0])
    r = ms_signal_10x_change(exp, "jump", 1)
    assert r == 0 and isinstance(r, int)
    r2 = ms_signal_10x_change(_ms1_run([]), "fall", 1)
    assert r2 == 0 and isinstance(r2, int)


def test_jump_counts():
    exp = _ms1_run([1.0, 10.0, 100.0])  # two >=10x jumps
    assert ms_signal_10x_change(exp, "jump", 1) == 2
    assert ms_signal_10x_change(exp, "fall", 1) == 0


def test_zero_tic_policy():
    # values: 10 -> 1 (fall .1), 1 -> 0 (fall to zero, counts), 0 -> 5 (from
    # zero, undefined, excluded)
    exp = _ms1_run([10.0, 1.0, 0.0, 5.0])
    assert ms_signal_10x_change(exp, "fall", 1) == 2
    assert ms_signal_10x_change(exp, "jump", 1) == 0


def test_tic_cv_uses_sample_sd():
    exp = _ms1_run([10.0, 20.0, 30.0])
    m = compute_qc_metrics(exp)
    # sample sd of [10,20,30] = 10; mean = 20; CV = 0.5
    assert m["TIC_MS1_CV"] == pytest.approx(0.5)
