"""Numeric oracles for DIA isolation, fragment signal and survey cycles (#69)."""

import json

import numpy as np
import pyopenms as oms
import pytest

from rawQC.dia import DIA_METRIC_METADATA, compute_dia_metrics


def _precursor(center=410, lower=10, upper=10):
    precursor = oms.Precursor()
    precursor.setMZ(float(center))
    precursor.setIsolationWindowLowerOffset(float(lower))
    precursor.setIsolationWindowUpperOffset(float(upper))
    precursor.setIntensity(1e9)  # must never enter fragment-signal summaries
    return precursor


def _scan(rt, level=2, center=410, lower=10, upper=10, intensities=(10,)):
    spectrum = oms.MSSpectrum()
    spectrum.setMSLevel(level)
    spectrum.setRT(float(rt))
    spectrum.set_peaks((np.arange(len(intensities), dtype=float) + 100,
                        np.asarray(intensities, dtype=float)))
    if level == 2 and center is not None:
        spectrum.setPrecursors([_precursor(center, lower, upper)])
    return spectrum


def _experiment(spectra):
    experiment = oms.MSExperiment()
    for spectrum in spectra:
        experiment.addSpectrum(spectrum)
    return experiment


def _variable_window_run():
    return _experiment([
        _scan(0, level=1),
        _scan(0.1, center=405, lower=5, upper=15, intensities=(10, 20)),
        _scan(0.2, center=430, lower=12, upper=18, intensities=(0,)),
        _scan(2, level=1),
        _scan(2.1, center=405, lower=5, upper=15, intensities=(30, 60)),
        _scan(2.2, center=430, lower=12, upper=18, intensities=()),
        _scan(6, level=1),
        _scan(6.1, center=405, lower=5, upper=15, intensities=(300,)),
    ])


def test_asymmetric_overlapping_variable_windows_and_fragment_tic():
    metrics = compute_dia_metrics(_variable_window_run())
    assert metrics["DIA_MS2_Count"] == 5
    assert metrics["DIA_IsolationWindow_Count"] == 2
    assert metrics["DIA_IsolationWindow_MzRange"] == [400, 448]
    assert metrics["DIA_IsolationWindow_WidthRange"] == [20, 30]
    table = metrics["DIA_IsolationWindow_Summary"]
    assert table["lower_mz"] == [400, 418]
    assert table["upper_mz"] == [420, 448]
    assert table["scan_count"] == [3, 2]
    assert table["empty_scan_count"] == [0, 2]
    assert table["tic_sum"] == [420, 0]
    assert table["tic_median"] == [90, 0]
    assert table["drift_time"] == [None, None]


def test_ms1_bounded_cycles_exclude_unbounded_tail():
    metrics = compute_dia_metrics(_variable_window_run())
    assert metrics["DIA_MS1BoundedCycle_Count"] == 2
    assert metrics["DIA_MS1BoundedCycle_DurationMedian"] == 3
    assert metrics["DIA_MS1BoundedCycle_DurationCV"] == pytest.approx(1 / 3)
    assert metrics["DIA_MS1BoundedCycle_MS2CountMedian"] == 2
    assert metrics["DIA_MS1BoundedCycle_MS2CountRange"] == [2, 2]
    assert metrics["DIA_MS1BoundedCycle_WindowCountMedian"] == 2
    assert metrics["DIA_MS1BoundedCycle_WindowCountRange"] == [2, 2]


def test_missing_invalid_and_multi_precursor_metadata_are_separate():
    ambiguous = _scan(3)
    ambiguous.setPrecursors([_precursor(), _precursor(500)])
    spectra = [
        _scan(0, center=None),
        _scan(1, lower=0, upper=0),
        _scan(2, center=float("nan")),
        ambiguous,
        _scan(4, center=10, lower=20),
        _scan(5, upper=float("inf")),
        _scan(6, level=3),
        _scan(7),
    ]
    metrics = compute_dia_metrics(_experiment(spectra))
    assert metrics["DIA_MS2_Count"] == 7
    assert metrics["DIA_IsolationMetadata_MissingCount"] == 2
    assert metrics["DIA_IsolationMetadata_InvalidCount"] == 3
    assert metrics["DIA_IsolationMetadata_AmbiguousCount"] == 1
    assert metrics["DIA_IsolationWindow_Count"] == 1
    assert metrics["DIA_IsolationWindow_Summary"]["scan_count"] == [1]
    assert metrics["DIA_IsolationWindow_Summary"]["tic_sum"] == [10]


def test_equal_rt_diapasef_preserves_mobility_windows_without_inventing_cycles():
    first, second = _scan(0), _scan(0)
    for spectrum, low, high in [(first, 0.7, 0.9), (second, 1.0, 1.2)]:
        spectrum.setDriftTimeUnit(oms.DriftTimeUnit.VSSC)
        spectrum.setMetaValue("ion mobility lower limit", low)
        spectrum.setMetaValue("ion mobility upper limit", high)
    unbounded = compute_dia_metrics(_experiment([first, second]))
    assert unbounded["DIA_IsolationWindow_Count"] == 2
    assert unbounded["DIA_MS1BoundedCycle_Count"] == 0
    assert unbounded["DIA_MS1BoundedCycle_DurationMedian"] is None
    table = unbounded["DIA_IsolationWindow_Summary"]
    assert table["ion_mobility_lower"] == [0.7, 1.0]
    assert table["ion_mobility_upper"] == [0.9, 1.2]
    assert table["drift_time"] == [None, None]
    assert table["drift_time_unit"] == [int(oms.DriftTimeUnit.VSSC)] * 2
    bounded = compute_dia_metrics(_experiment([
        _scan(0, level=1), first, second, _scan(1, level=1),
    ]))
    assert bounded["DIA_MS1BoundedCycle_Count"] == 1
    assert bounded["DIA_MS1BoundedCycle_DurationMedian"] == 1
    assert bounded["DIA_MS1BoundedCycle_MS2CountMedian"] == 2
    assert bounded["DIA_MS1BoundedCycle_DurationCV"] is None


def test_faims_voltage_is_part_of_window_identity():
    spectra = [_scan(0), _scan(1), _scan(2)]
    for spectrum, cv in zip(spectra, [-45, -65, -45]):
        spectrum.setDriftTime(float(cv))
        spectrum.setDriftTimeUnit(oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE)
    metrics = compute_dia_metrics(_experiment(spectra))
    table = metrics["DIA_IsolationWindow_Summary"]
    assert metrics["DIA_IsolationWindow_Count"] == 2
    assert table["drift_time"] == [-65, -45]
    assert table["scan_count"] == [1, 2]


def test_precursor_mobility_selection_is_part_of_window_identity():
    spectra = [_scan(0), _scan(1)]
    for spectrum, drift in zip(spectra, [0.8, 1.2]):
        spectrum.setDriftTimeUnit(oms.DriftTimeUnit.VSSC)
        precursor = spectrum.getPrecursors()[0]
        precursor.setDriftTime(drift)
        precursor.setDriftTimeWindowLowerOffset(0.1)
        precursor.setDriftTimeWindowUpperOffset(0.2)
        spectrum.setPrecursors([precursor])
    metrics = compute_dia_metrics(_experiment(spectra))
    assert metrics["DIA_IsolationWindow_Count"] == 2
    table = metrics["DIA_IsolationWindow_Summary"]
    assert table["precursor_drift_time"] == [0.8, 1.2]
    assert table["precursor_drift_lower_offset"] == [0.1, 0.1]
    assert table["precursor_drift_upper_offset"] == [0.2, 0.2]


def test_nonfinite_fragment_intensities_are_counted_without_fabricating_tic():
    metrics = compute_dia_metrics(_experiment([
        _scan(0, intensities=(float("nan"),)),
        _scan(1, intensities=(20,)),
        _scan(2, center=500, intensities=(float("inf"),)),
    ]))
    table = metrics["DIA_IsolationWindow_Summary"]
    assert table["scan_count"] == [2, 1]
    assert table["invalid_tic_scan_count"] == [1, 1]
    assert table["empty_scan_count"] == [0, 0]
    assert table["tic_sum"] == [20, None]
    assert table["tic_median"] == [20, None]
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("start, middle, end", [
    (1, 1, 1), (2, 1.5, 1), (0, float("nan"), 1),
    (0, 2, 1), (float("nan"), 0.5, 1),
])
def test_unreliable_rt_boundaries_do_not_define_cycles(start, middle, end):
    metrics = compute_dia_metrics(_experiment([
        _scan(start, level=1), _scan(middle), _scan(end, level=1),
    ]))
    assert metrics["DIA_MS1BoundedCycle_Count"] == 0
    assert metrics["DIA_IsolationWindow_Summary"]["scan_count"] == [1]


def test_empty_run_has_counts_but_no_fabricated_ranges():
    metrics = compute_dia_metrics(oms.MSExperiment())
    assert metrics["DIA_MS2_Count"] == 0
    assert metrics["DIA_IsolationWindow_Count"] == 0
    assert metrics["DIA_MS1BoundedCycle_Count"] == 0
    assert metrics["DIA_IsolationWindow_MzRange"] is None
    assert all(column == [] for column in metrics["DIA_IsolationWindow_Summary"].values())


def test_all_dia_metrics_have_custom_metadata():
    metrics = compute_dia_metrics(_variable_window_run())
    assert set(metrics) == set(DIA_METRIC_METADATA)
    assert all(meta["accession"] is None and meta["description"]
               for meta in DIA_METRIC_METADATA.values())


def test_bruker_metadata_distinguishes_reader_omissions_from_loaded_empty_scans():
    experiment = _variable_window_run()
    experiment.setMetaValue("rawqc_bruker_expected_ms1_spectra", 4)
    experiment.setMetaValue("rawqc_bruker_expected_ms2_spectra", 8)
    metrics = compute_dia_metrics(experiment)
    assert metrics["DIA_Bruker_ExpectedMS1SpectrumCount"] == 4
    assert metrics["DIA_Bruker_ExpectedMS2SpectrumCount"] == 8
    assert metrics["DIA_Bruker_ReaderOmittedMS1SpectrumCount"] == 1
    assert metrics["DIA_Bruker_ReaderOmittedMS2SpectrumCount"] == 3
    assert metrics["DIA_IsolationWindow_Summary"]["empty_scan_count"] == [0, 2]


def test_absent_or_inconsistent_bruker_metadata_does_not_invent_omissions():
    experiment = _variable_window_run()
    metrics = compute_dia_metrics(experiment)
    assert metrics["DIA_Bruker_ExpectedMS2SpectrumCount"] is None
    assert metrics["DIA_Bruker_ReaderOmittedMS2SpectrumCount"] is None
    experiment.setMetaValue("rawqc_bruker_expected_ms2_spectra", 2)
    metrics = compute_dia_metrics(experiment)
    assert metrics["DIA_Bruker_ExpectedMS2SpectrumCount"] == 2
    assert metrics["DIA_Bruker_ReaderOmittedMS2SpectrumCount"] is None


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (1, 0, 3, 2), (3, 0, 2, 1)])
def test_ms2_tic_area_sums_equal_rt_windows_before_integration(order):
    spectra = [
        _scan(0, intensities=(1,)), _scan(0, center=500, intensities=(100,)),
        _scan(1, intensities=(1,)), _scan(1, center=500, intensities=(100,)),
    ]
    metrics = compute_dia_metrics(_experiment([spectra[i] for i in order]))
    assert metrics["DIA_MS2_TIC_Area"] == 101


def test_ms2_tic_area_needs_two_distinct_valid_times():
    metrics = compute_dia_metrics(_experiment([
        _scan(0, intensities=(1,)), _scan(0, intensities=(100,)),
        _scan(float("nan"), intensities=(100,)),
        _scan(1, intensities=(float("nan"),)),
        _scan(2, intensities=(float("inf"),)),
    ]))
    assert metrics["DIA_MS2_TIC_Area"] is None


def test_ms2_tic_area_includes_missing_window_metadata_and_excludes_invalid_scans():
    metrics = compute_dia_metrics(_experiment([
        _scan(0, center=None, intensities=(2,)),
        _scan(1, center=None, intensities=(4,)),
        _scan(float("inf"), intensities=(100,)),
        _scan(1, intensities=(1, float("nan"))),
        _scan(2, intensities=(float("inf"),)),
        _scan(0, level=1, intensities=(1000,)),
    ]))
    assert metrics["DIA_MS2_TIC_Area"] == 3
