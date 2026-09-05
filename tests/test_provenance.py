import json

import pyopenms as oms

from rawQC.provenance import PROVENANCE_METRIC_METADATA, compute_provenance_metrics


def _rows(metrics):
    table = metrics["Provenance_RecordedMetadata"]
    return [dict(zip(table, row)) for row in zip(*table.values())]


def test_missing_defaults_are_null_and_payload_is_json_safe():
    metrics = compute_provenance_metrics(oms.MSExperiment())
    assert set(metrics) == set(PROVENANCE_METRIC_METADATA)
    assert all(meta["accession"] is None for meta in PROVENANCE_METRIC_METADATA.values())
    assert metrics["Provenance_Run"]["acquisition_datetime"] is None
    assert metrics["Provenance_Instrument"]["model"] is None
    assert all(value is None for value in metrics["Provenance_Sample"].values())
    assert metrics["Provenance_Processing"] == []
    json.dumps(metrics, allow_nan=False)


def test_records_instrument_sample_units_and_original_files():
    exp = oms.MSExperiment()
    instrument = oms.Instrument()
    instrument.setName("Orbitrap Exploris 480")
    instrument.setModel("Exploris")
    instrument.setVendor("Thermo Fisher Scientific")
    instrument.setMetaValue("instrument serial number", "SN123")
    software = oms.Software()
    software.setName("Acquisition suite")
    software.setVersion("4.2")
    instrument.setSoftware(software)
    exp.setInstrument(instrument)
    sample = oms.Sample()
    sample.setName("QC pool")
    sample.setNumber("QC_17")
    sample.setVolume(0.25)
    sample.setMass(0.001)
    sample.setConcentration(0.1)
    sample.setMetaValue("Vial", "A1")
    sample.setMetaValue("injection volume (uL)", 2.0)
    exp.setSample(sample)
    date = oms.DateTime()
    date.set("2026-09-05 12:13:14")
    exp.setDateTime(date)
    original = oms.SourceFile()
    original.setNameOfFile("run.raw")
    original.setPathToFile("file:///instrument/data")
    original.setChecksum("deadbeef", oms.SourceFile.ChecksumType.SHA1)
    exp.setSourceFiles([original])

    metrics = compute_provenance_metrics(exp)
    assert metrics["Provenance_Run"]["acquisition_datetime"] == "2026-09-05 12:13:14"
    assert metrics["Provenance_Instrument"]["serial_number"] == "SN123"
    assert metrics["Provenance_Instrument"]["software"] == {"name": "Acquisition suite", "version": "4.2"}
    assert metrics["Provenance_Sample"]["volume_mL"] == 0.25
    assert metrics["Provenance_Sample"]["injection_volume_uL"] == 2.0
    assert metrics["Provenance_Sample"]["vial"] == "A1"
    assert metrics["Provenance_SourceFiles"][0]["checksum_type"] == "SHA-1"
    assert exp.getSample().getVolume() == 0.25


def test_vendor_metadata_preserved_without_guessing_units_or_dumping_frames():
    exp = oms.MSExperiment()
    exp.setMetaValue("rawqc_bruker_global_metadata_json", json.dumps({
        "SampleName": "pool", "InstrumentSerialNumber": "tims123",
        "InjectionVolume": "2", "AcquisitionSoftware": "timsTOF",
        "InstrumentMethodFile": "dia.m", "NovelVendorField": "arbitrary value",
    }))
    exp.setMetaValue("rawqc_bruker_frames_json", '[{"Id": 1, "NumPeaks": 9000000}]')
    exp.setMetaValue("rawqc_bruker_expected_ms1_spectra", 1)
    exp.setMetaValue("custom zero", 0)
    exp.setMetaValue("nonfinite", [1.0, float("nan"), float("inf")])
    metrics = compute_provenance_metrics(exp)
    assert metrics["Provenance_Sample"]["name"] == "pool"
    assert metrics["Provenance_Sample"]["injection_volume_uL"] is None
    assert metrics["Provenance_Instrument"]["serial_number"] == "tims123"
    assert metrics["Provenance_Run"]["method_files"][0]["value"] == "dia.m"
    rows = _rows(metrics)
    assert not any(row["key"].startswith("rawqc_") for row in rows)
    assert next(row["value"] for row in rows if row["key"] == "custom zero") == 0
    assert next(row["value"] for row in rows if row["key"] == "nonfinite") == [1.0, None, None]
    assert next(row for row in rows if row["key"] == "NovelVendorField")["source"] == "bruker.GlobalMetadata"
    json.dumps(metrics, allow_nan=False)


def test_conflicting_serial_numbers_are_retained_without_arbitrary_choice():
    exp = oms.MSExperiment()
    instrument = oms.Instrument()
    instrument.setMetaValue("instrument serial number", "A")
    exp.setInstrument(instrument)
    exp.setMetaValue("SerialNumber", "B")
    metrics = compute_provenance_metrics(exp)
    assert metrics["Provenance_Instrument"]["serial_number"] is None
    assert {row["value"] for row in _rows(metrics)} == {"A", "B"}


def test_processing_is_deduplicated_and_distinct_metadata_remains_distinct():
    exp = oms.MSExperiment()
    processing = oms.DataProcessing()
    software = oms.Software()
    software.setName("ThermoRawFileParser")
    software.setVersion("2.0")
    processing.setSoftware(software)
    processing.setProcessingActions({oms.DataProcessing.ProcessingAction.CONVERSION_MZML})
    processing.setMetaValue("ProcessingMethodFile", "convert.xml")
    for _ in range(2):
        spectrum = oms.MSSpectrum()
        spectrum.setDataProcessing([processing])
        exp.addSpectrum(spectrum)
    chromatogram = oms.MSChromatogram()
    chromatogram.setDataProcessing([processing])
    exp.addChromatogram(chromatogram)
    different = oms.DataProcessing(processing)
    different.setMetaValue("ProcessingMethodFile", "alternate.xml")
    spectrum = oms.MSSpectrum()
    spectrum.setDataProcessing([different])
    exp.addSpectrum(spectrum)
    result = compute_provenance_metrics(exp)["Provenance_Processing"]
    assert len(result) == 2
    assert result[0]["spectrum_count"] == 2
    assert result[0]["chromatogram_count"] == 1
    assert result[0]["first_spectrum_index"] == 0
    assert result[0]["first_chromatogram_index"] == 0
    assert result[0]["actions"] == ["Conversion to mzML format"]
    assert result[1]["first_spectrum_index"] == 2
    assert result[1]["chromatogram_count"] == 0
    assert result[0]["metadata"][0]["value"] == "convert.xml"


def test_malformed_optional_native_metadata_does_not_break_provenance():
    for value in ("not json", "[]", "null"):
        exp = oms.MSExperiment()
        exp.setMetaValue("rawqc_bruker_global_metadata_json", value)
        assert compute_provenance_metrics(exp)["Provenance_RecordedMetadata"]["key"] == []


def test_cv_term_values_and_recorded_units_are_preserved():
    exp = oms.MSExperiment()
    term = oms.CVTerm()
    term.setAccession("LOCAL:source_volume")
    term.setName("source volume")
    term.setValue(2.0)
    unit = term.getUnit()
    unit.name = "microliter"
    unit.accession = "UO:0000101"
    unit.cv_ref = "UO"
    term.setUnit(unit)
    source = oms.SourceFile()
    source.setCVTerms([term])
    exp.setSourceFiles([source])
    row = _rows(compute_provenance_metrics(exp))[0]
    assert row["source"] == "source_files[0]"
    assert row["accession"] == "LOCAL:source_volume"
    assert row["value"] == 2.0
    assert row["unit"] == {"accession": "UO:0000101", "name": "microliter", "cv_ref": "UO"}


def test_injection_volume_converts_explicit_units_and_preserves_conflicts():
    exp = oms.MSExperiment()
    exp.setMetaValue("InjectionVolume_mL", 0.002)
    assert compute_provenance_metrics(exp)["Provenance_Sample"]["injection_volume_uL"] == 2.0
    exp.setMetaValue("InjectionVolume_uL", 2.0)
    assert compute_provenance_metrics(exp)["Provenance_Sample"]["injection_volume_uL"] == 2.0
    sample = oms.Sample()
    sample.setMetaValue("InjectionVolume_uL", 3.0)
    exp.setSample(sample)
    assert compute_provenance_metrics(exp)["Provenance_Sample"]["injection_volume_uL"] is None
