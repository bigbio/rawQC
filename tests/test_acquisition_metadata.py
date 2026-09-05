"""Recorded units, missing-data denominators and native frame accounting."""

import json
from pathlib import Path

import pyopenms as oms
import pytest

from rawQC.acquisition import ACQUISITION_METRIC_METADATA, compute_acquisition_metrics


def scan(rt=0, level=1, actual=None, maximum=None, native_id=""):
    spectrum = oms.MSSpectrum()
    spectrum.setRT(float(rt))
    spectrum.setMSLevel(level)
    spectrum.setNativeID(native_id)
    spectrum.set_peaks(([100.], [10.]))
    if actual is not None:
        acquisition = oms.Acquisition()
        acquisition.setMetaValue("MS:1000927", actual)
        info = oms.AcquisitionInfo()
        info.push_back(acquisition)
        spectrum.setAcquisitionInfo(info)
    if maximum is not None:
        spectrum.setMetaValue("Max. Ion Time (ms)", maximum)
    return spectrum


def experiment(*spectra):
    exp = oms.MSExperiment()
    for spectrum in spectra:
        exp.addSpectrum(spectrum)
    return exp


def set_window(spectrum, center=410, low=10, high=10, cv=None, im=None):
    precursor = oms.Precursor()
    precursor.setMZ(center)
    precursor.setIsolationWindowLowerOffset(low)
    precursor.setIsolationWindowUpperOffset(high)
    spectrum.setPrecursors([precursor])
    if cv is not None:
        spectrum.setDriftTime(cv)
        spectrum.setDriftTimeUnit(oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE)
    if im is not None:
        spectrum.setMetaValue("ion mobility lower limit", im[0])
        spectrum.setMetaValue("ion mobility upper limit", im[1])
        spectrum.setDriftTimeUnit(oms.DriftTimeUnit.VSSC)
    return spectrum


def test_missing_values_are_null_not_defaults():
    metrics = compute_acquisition_metrics(experiment(scan()))
    assert set(metrics) == set(ACQUISITION_METRIC_METADATA)
    table = metrics["Acquisition_ScanMetadata"]
    for key in ("ion_injection_time_ms", "max_ion_injection_time_ms", "agc_target",
                "activation_energy_ev", "collision_energy_ev", "normalized_collision_energy_percent",
                "reported_mass_resolving_power", "reported_mass_resolution"):
        assert table[key] == [None]
        assert table[f"{key}_status"] == ["missing"]
    assert table["faims_cv_volts"] == [None]
    assert metrics["IonInjectionTime_Summary"]["fraction_at_max_time"] == [None]
    assert metrics["Acquisition_BrukerFrameMetadata"] is None
    json.dumps(metrics, allow_nan=False)
    default_spectrum = oms.MSSpectrum()
    default_table = compute_acquisition_metrics(experiment(default_spectrum))["Acquisition_ScanMetadata"]
    assert default_table["retention_time_seconds"] == [None]


def test_actual_maximum_and_paired_denominators_and_linear_quantiles():
    exp = experiment(scan(actual=10, maximum=10), scan(actual=20, maximum=100),
                     scan(actual=40), scan(maximum=50), scan(actual=0, maximum=0),
                     scan(actual=-5, maximum=10), scan(actual=float("nan")),
                     scan(actual=60, maximum=50))
    metrics = compute_acquisition_metrics(exp)
    summary = metrics["IonInjectionTime_Summary"]
    assert summary["spectrum_count"] == [8]
    assert summary["actual_time_count"] == [5]
    assert summary["positive_max_time_count"] == [5]
    assert summary["paired_time_count"] == [3]
    assert summary["at_max_time_count"] == [2]
    assert summary["above_max_time_count"] == [1]
    assert summary["fraction_at_max_time"] == [pytest.approx(2 / 3)]
    assert summary["injection_time_median_ms"] == [20]
    assert summary["injection_time_q25_ms"] == [10]
    assert summary["injection_time_p95_ms"] == [pytest.approx(56)]
    json.dumps(metrics, allow_nan=False)


def test_explicit_time_units_are_converted_and_missing_xml_units_are_not_guessed():
    seconds = scan(actual=.01, maximum=.02)
    seconds.setMetaValue("rawqc_metadata_units_json", json.dumps({
        "acquisition[0]:MS:1000927": "UO:0000010",
        "spectrum:Max. Ion Time (ms)": "UO:0000010",
    }))
    missing = scan(actual=10, maximum=20)
    missing.setMetaValue("rawqc_metadata_units_json", json.dumps({
        "acquisition[0]:MS:1000927": None,
    }))
    unsupported = scan(actual=10)
    unsupported.setMetaValue("rawqc_metadata_units_json", json.dumps({
        "acquisition[0]:MS:1000927": "UO:0000266",
    }))
    metrics = compute_acquisition_metrics(experiment(seconds, missing, unsupported))
    table = metrics["Acquisition_ScanMetadata"]
    assert table["ion_injection_time_ms"] == [10, None, None]
    assert table["max_ion_injection_time_ms"] == [20, 20, None]
    assert table["ion_injection_time_ms_status"] == ["recorded", "unit_unknown", "unit_unsupported"]
    sources = metrics["Acquisition_MetadataSources"]
    assert sources["raw_unit"] == ["UO:0000010", "UO:0000010", "ms", None, "UO:0000266"]


def test_unlabeled_maximum_requires_units_and_agc_keeps_unknown_unit():
    unknown = scan(actual=10)
    unknown.setMetaValue("maximum injection time", 10)
    unknown.setMetaValue("AGC Target", 1e6)
    known = scan(actual=10)
    known.setMetaValue("maximum injection time", .01)
    known.setMetaValue("maximum injection time unit", "s")
    known.setMetaValue("Normalized AGC Target (%)", 200)
    table = compute_acquisition_metrics(experiment(unknown, known))["Acquisition_ScanMetadata"]
    assert table["max_ion_injection_time_ms"] == [None, 10]
    assert table["agc_target"] == [1e6, 200]
    assert table["agc_target_unit"] == [None, "%"]


def test_collision_energy_and_resolving_power_use_recorded_metadata_only():
    first = set_window(scan(level=2))
    precursor = first.getPrecursors()[0]
    precursor.setActivationEnergy(50)  # ActivationEnergy is not collision energy.
    first.setPrecursors([precursor])
    second = set_window(scan(level=2))
    precursor = second.getPrecursors()[0]
    precursor.setMetaValue("collision energy", 25)
    precursor.setMetaValue("percent collision energy", 30)
    second.setPrecursors([precursor])
    second.setMetaValue("mass resolving power", 60000)
    second.setMetaValue("mass resolution", .002)
    metrics = compute_acquisition_metrics(experiment(first, second))
    table = metrics["Acquisition_ScanMetadata"]
    assert table["collision_energy_ev"] == [None, 25]
    assert table["activation_energy_ev"] == [50, None]
    assert table["normalized_collision_energy_percent"] == [None, 30]
    assert table["reported_mass_resolving_power"] == [None, 60000]
    assert table["reported_mass_resolution"] == [None, .002]
    assert table["reported_mass_resolution_unit"] == [None, None]


def test_combined_acquisitions_and_conflicting_actual_times_are_ambiguous():
    combined = scan(actual=10)
    info = combined.getAcquisitionInfo()
    another = oms.Acquisition()
    another.setMetaValue("MS:1000927", 10)
    info.push_back(another)
    combined.setAcquisitionInfo(info)
    conflicting = scan(actual=10)
    conflicting.setMetaValue("Ion Injection Time (ms)", 20)
    table = compute_acquisition_metrics(experiment(combined, conflicting))["Acquisition_ScanMetadata"]
    assert table["ion_injection_time_ms"] == [None, None]
    assert table["ion_injection_time_ms_status"] == ["ambiguous", "ambiguous"]


def test_grouping_separates_ms_level_dia_windows_faims_and_mobility():
    scans = [scan(actual=1), set_window(scan(level=2, actual=2), cv=-45),
             set_window(scan(level=2, actual=3), cv=-65),
             set_window(scan(level=2, actual=4), center=500, cv=-45),
             set_window(scan(level=2, actual=5), im=(.7, .9)),
             set_window(scan(level=2, actual=6), im=(1., 1.2))]
    dia = compute_acquisition_metrics(experiment(*scans), "dia")["IonInjectionTime_Summary"]
    assert dia["spectrum_count"] == [1] * 6
    assert dia["isolation_lower_mz"] == [None, 400, 400, 490, 400, 400]
    dda = compute_acquisition_metrics(experiment(*scans), "dda")["IonInjectionTime_Summary"]
    assert dda["spectrum_count"] == [1, 2, 1, 2]
    assert dda["isolation_lower_mz"] == [None] * 4


def test_half_open_rt_bins_and_invalid_rt_exclusion():
    metrics = compute_acquisition_metrics(experiment(
        scan(0, actual=1), scan(59.99, actual=3), scan(60, actual=5),
        scan(120, actual=7), scan(-1, actual=9), scan(float("nan"), actual=11)))
    binned = metrics["IonInjectionTime_RTSummary"]
    assert binned["rt_bin_start_seconds"] == [0, 60, 120]
    assert binned["rt_bin_end_seconds"] == [60, 120, 180]
    assert binned["spectrum_count"] == [2, 1, 1]
    assert binned["injection_time_median_ms"] == [2, 5, 7]
    assert metrics["IonInjectionTime_Summary"]["spectrum_count"] == [6]


def test_fraction_at_maximum_tolerance():
    summary = compute_acquisition_metrics(experiment(
        scan(actual=99.99995, maximum=100), scan(actual=99.9, maximum=100),
        scan(actual=100.00005, maximum=100)))["IonInjectionTime_Summary"]
    assert summary["at_max_time_count"] == [2]
    assert summary["above_max_time_count"] == [0]


def attach_frames(exp, frames):
    exp.setMetaValue("rawqc_bruker_frames_json", json.dumps(frames))
    exp.setMetaValue("rawqc_bruker_frame_units_json", json.dumps({
        "Time": "s", "AccumulationTime": "ms", "RampTime": "ms"}))
    return exp


def test_frame_rows_not_duplicated_and_reader_omission_is_distinct_from_id_gap():
    exp = attach_frames(experiment(
        scan(native_id="frame=1 windowGroup=1 scan=0"),
        scan(native_id="frame=1 windowGroup=1 scan=1"),
        scan(native_id="frame=4 windowGroup=1 scan=0")), [
        {"Id": 1, "Time": 0, "AccumulationTime": 20, "RampTime": 30, "NumPeaks": 500},
        {"Id": 2, "Time": 1, "AccumulationTime": 25, "RampTime": 30, "NumPeaks": 0},
        {"Id": 4, "Time": 4, "AccumulationTime": 40, "RampTime": 30, "NumPeaks": 700},
    ])
    metrics = compute_acquisition_metrics(exp)
    frames = metrics["Acquisition_BrukerFrameMetadata"]
    assert frames["Id"] == [1, 2, 4]
    assert frames["NumPeaks"] == [500, 0, 700]
    assert frames["rawqc_loaded_spectrum_count"] == [2, 0, 1]
    summary = metrics["Acquisition_BrukerFrameSummary"]
    assert summary["acquisition_frame_count"] == [3]
    assert summary["loaded_frame_count"] == [2]
    assert summary["reader_omitted_frame_count"] == [1]
    assert summary["nonconsecutive_id_transition_count"] == [1]
    assert summary["positive_interval_median_seconds"] == [2]
    assert summary["positive_interval_cv"] == [pytest.approx(.5)]
    scans = metrics["Acquisition_ScanMetadata"]
    assert scans["accumulation_time_ms"] == [20, 20, 40]
    assert scans["ion_injection_time_ms"] == [None] * 3


@pytest.mark.parametrize("native_id", ["scan=1", "frame=99", "frame=1 frame=2"])
def test_incomplete_frame_mapping_prevents_omission_claims(native_id):
    exp = attach_frames(experiment(scan(native_id=native_id)), [
        {"Id": 1, "Time": 0}, {"Id": 2, "Time": 0}, {"Id": 3, "Time": -1}])
    metrics = compute_acquisition_metrics(exp)
    assert metrics["Acquisition_BrukerFrameMetadata"]["rawqc_loaded_spectrum_count"] == [None] * 3
    summary = metrics["Acquisition_BrukerFrameSummary"]
    assert summary["reader_omitted_frame_count"] == [None]
    assert summary["nonincreasing_rt_transition_count"] == [2]


def test_empty_input_has_stable_table_schemas():
    metrics = compute_acquisition_metrics(oms.MSExperiment())
    assert metrics["Acquisition_ScanMetadata"]["spectrum_index"] == []
    assert metrics["IonInjectionTime_Summary"]["spectrum_count"] == []
    json.dumps(metrics, allow_nan=False)


def test_exact_unit_bearing_user_label_supplies_unit_without_xml_unit_accession():
    spectrum = scan(actual=10, maximum=20)
    spectrum.setMetaValue("rawqc_metadata_units_json", json.dumps({
        "acquisition[0]:MS:1000927": None,
        "spectrum:Max. Ion Time (ms)": None,
    }))
    metrics = compute_acquisition_metrics(experiment(spectrum))
    table = metrics["Acquisition_ScanMetadata"]
    assert table["ion_injection_time_ms"] == [None]
    assert table["max_ion_injection_time_ms"] == [20]
    assert metrics["Acquisition_MetadataSources"]["unit_source"] == ["parameter_label", None]


def test_bruker_dda_aggregated_precursor_does_not_imply_single_frame_or_omission():
    exp = attach_frames(experiment(scan(native_id="frame=1 precursor=10")), [
        {"Id": 1, "Time": 0, "AccumulationTime": 20, "RampTime": 30},
        {"Id": 2, "Time": 1, "AccumulationTime": 25, "RampTime": 30},
    ])
    metrics = compute_acquisition_metrics(exp)
    table = metrics["Acquisition_ScanMetadata"]
    assert table["frame_id"] == [1]
    assert table["frame_association_status"] == ["first_contributing_frame"]
    assert table["accumulation_time_ms"] == [None]
    assert metrics["Acquisition_BrukerFrameSummary"]["reader_omitted_frame_count"] == [None]
    assert metrics["Acquisition_BrukerFrameMetadata"]["rawqc_loaded_spectrum_count"] == [None, None]


def test_structured_activation_energy_retains_explicit_zero_and_missing_units():
    spectra = [set_window(scan(level=2)) for _ in range(4)]
    spectra[1].setMetaValue("rawqc_metadata_units_json", json.dumps({
        "precursor[0]:MS:1000509": "UO:0000266"}))
    for spectrum in spectra[2:]:
        precursor = spectrum.getPrecursors()[0]
        precursor.setActivationEnergy(20)
        spectrum.setPrecursors([precursor])
    spectra[2].setMetaValue("rawqc_metadata_units_json", json.dumps({
        "precursor[0]:MS:1000509": None}))
    spectra[3].setMetaValue("rawqc_metadata_units_json", json.dumps({
        "precursor[0]:MS:1000509": "UO:0000187"}))
    table = compute_acquisition_metrics(experiment(*spectra))["Acquisition_ScanMetadata"]
    assert table["activation_energy_ev"] == [None, 0, None, None]
    assert table["activation_energy_ev_status"] == ["missing", "recorded", "unit_unknown", "unit_unsupported"]


def test_conflicting_precursor_activation_energies_are_ambiguous():
    spectrum = set_window(scan(level=2))
    precursors = spectrum.getPrecursors()
    precursors[0].setActivationEnergy(20)
    other = oms.Precursor()
    other.setActivationEnergy(30)
    spectrum.setPrecursors(precursors + [other])
    table = compute_acquisition_metrics(experiment(spectrum))["Acquisition_ScanMetadata"]
    assert table["activation_energy_ev"] == [None]
    assert table["activation_energy_ev_status"] == ["ambiguous"]


def test_mzml_structured_activation_energy_roundtrip(tmp_path):
    from rawQC.input import load_experiment

    spectrum = set_window(scan(level=2))
    precursor = spectrum.getPrecursors()[0]
    precursor.setActivationEnergy(35)
    spectrum.setPrecursors([precursor])
    path = tmp_path / "energy.mzML"
    oms.MzMLFile().store(str(path), experiment(spectrum))
    loaded = load_experiment(path)
    table = compute_acquisition_metrics(loaded)["Acquisition_ScanMetadata"]
    assert table["activation_energy_ev"] == [35]
    assert table["collision_energy_ev"] == [None]


def test_pinned_bruker_reader_does_not_invent_unexported_activation_energy():
    from rawQC.input import load_experiment

    loaded = load_experiment(Path(__file__).parent / "data" / "bruker_dia.d")
    table = compute_acquisition_metrics(loaded)["Acquisition_ScanMetadata"]
    # This nightly does not copy Bruker window collision energy into Precursors.
    assert table["activation_energy_ev"] == [None] * len(loaded)
