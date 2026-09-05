"""Native input dispatch and DDA/DIA report integration for issue #69."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pyopenms as oms
import pytest
from click.testing import CliRunner

from rawQC.calculate_metrics import build_mzqc, compute_qc_metrics, main, validate_metric_registry
from rawQC.input import load_experiment, resolve_acquisition_mode

readers = importlib.import_module("rawQC.input")


def _dia_experiment(native_ids=False):
    exp = oms.MSExperiment()
    for cycle in range(3):
        ms1 = oms.MSSpectrum()
        ms1.setMSLevel(1)
        ms1.setRT(float(cycle * 2))
        ms1.set_peaks((np.array([420.0]), np.array([100.0])))
        exp.addSpectrum(ms1)
        for index, center in enumerate((425.0, 475.0)):
            sp = oms.MSSpectrum()
            sp.setMSLevel(2)
            sp.setRT(cycle * 2 + 0.5 + index * 0.5)
            sp.set_peaks((np.array([200.0]), np.array([10.0 + index])))
            p = oms.Precursor()
            p.setMZ(center)
            p.setIsolationWindowLowerOffset(25.0)
            p.setIsolationWindowUpperOffset(25.0)
            sp.setPrecursors([p])
            if native_ids:
                sp.setNativeID(f"frame={cycle * 2 + 1} windowGroup=1 scan={index}")
            exp.addSpectrum(sp)
    return exp


def test_dia_report_omits_single_precursor_statistics():
    metrics = compute_qc_metrics(_dia_experiment(), "dia")
    assert metrics["DIA_IsolationWindow_Count"] == 2
    assert metrics["DIA_IsolationWindow_MzRange"] == [400.0, 500.0]
    assert metrics["NumberOfSpectra_MS2"] == 6
    assert "PrecursorIntensity_FallbackCount" not in metrics
    assert "ChargeMean" not in metrics
    assert "MzRange_MS2" not in metrics
    problems = validate_metric_registry(metrics)
    assert problems["uncovered"] == []
    assert problems["registered_orphans"] == []
    # Existing FAIMS registrations are only computed when FAIMS is present.
    assert set(problems["ordered_not_computed"]) <= {"FAIMS_CV_Count", "FAIMS_CV_Values", "FAIMS_CV_Range"}
    document = json.loads(build_mzqc([{
        "filename": "run.mzML", "metrics": metrics, "instrument_metadata": {},
    }]))
    schema = json.loads((Path(__file__).parent / "data/mzqc_schema.json").read_text())
    jsonschema.validate(document, schema)


def test_auto_does_not_infer_dia_from_wide_repeated_windows():
    exp = _dia_experiment()
    assert resolve_acquisition_mode(exp) == "dda"
    assert resolve_acquisition_mode(exp, "dia") == "dia"
    assert resolve_acquisition_mode(_dia_experiment(native_ids=True)) == "dia"
    exp.setMetaValue("MS:1003215", "")
    assert resolve_acquisition_mode(exp) == "dia"
    assert resolve_acquisition_mode(exp, "dda") == "dda"
    with pytest.raises(ValueError, match="acquisition_mode"):
        resolve_acquisition_mode(exp, "invalid")


def test_partial_dia_annotations_require_explicit_selection():
    exp = _dia_experiment()
    sp = exp[1]
    sp.setMetaValue("acquisition_mode", "dia")
    exp[1] = sp
    with pytest.raises(ValueError, match="Mixed DIA"):
        resolve_acquisition_mode(exp)


def test_bruker_return_api_and_metadata_counts(tmp_path, monkeypatch):
    path = tmp_path / "run.D"
    path.mkdir()
    for name in ("analysis.tdf", "analysis.tdf_bin"):
        (path / name).touch()
    exp = _dia_experiment(native_ids=True)
    class Reader:
        def load(self, filename):
            assert filename == str(path)
            return exp
        def readDIAMetadata(self, filename):
            return SimpleNamespace(nr_ms1_spectra=4, nr_ms2_spectra=[4, 4]), None
    monkeypatch.setattr(readers.oms, "BrukerTimsFile", Reader)
    monkeypatch.setattr(readers, "_bruker_is_dia", lambda path: True)
    loaded = load_experiment(path)
    metrics = compute_qc_metrics(loaded)
    assert metrics["DIA_Bruker_ExpectedMS2SpectrumCount"] == 8
    assert metrics["DIA_Bruker_ReaderOmittedMS2SpectrumCount"] == 2
    assert metrics["DIA_Bruker_ReaderOmittedMS1SpectrumCount"] == 1

    # If the reader omits every empty MS2 window, the acquisition remains DIA
    # and acquisition metadata still makes all reader omissions visible.
    exp.clear(True)
    loaded = load_experiment(path)
    assert resolve_acquisition_mode(loaded) == "dia"
    metrics = compute_qc_metrics(loaded)
    assert metrics["DIA_Bruker_ReaderOmittedMS2SpectrumCount"] == 8
    assert metrics["DIA_IsolationWindow_Count"] == 0


def test_rejects_tsf_and_unrelated_directories(tmp_path):
    path = tmp_path / "run.d"
    path.mkdir()
    (path / "analysis.tsf").touch()
    with pytest.raises(ValueError, match="TSF"):
        load_experiment(path)
    with pytest.raises(ValueError, match="Unsupported input"):
        load_experiment(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_experiment(tmp_path / "absent.mzML")


def test_missing_bruker_build_has_actionable_error(tmp_path, monkeypatch):
    path = tmp_path / "run.d"
    path.mkdir()
    for name in ("analysis.tdf", "analysis.tdf_bin"):
        (path / name).touch()
    monkeypatch.delattr(readers.oms, "BrukerTimsFile")
    with pytest.raises(RuntimeError, match="pinned nightly"):
        load_experiment(path)


def test_thermo_uses_filehandler_and_preserves_error(tmp_path, monkeypatch):
    path = tmp_path / "run.RAW"
    path.touch()
    class SupportedReader:
        def loadExperiment(self, filename, exp):
            assert filename == str(path)
            exp.addSpectrum(oms.MSSpectrum())
    monkeypatch.setattr(readers.oms, "FileHandler", SupportedReader)
    assert load_experiment(path).size() == 1
    class UnsupportedReader:
        def loadExperiment(self, filename, exp):
            raise RuntimeError("type is not supported for loading experiments")
    monkeypatch.setattr(readers.oms, "FileHandler", UnsupportedReader)
    with pytest.raises(RuntimeError, match="convert to mzML") as error:
        load_experiment(path)
    assert "type is not supported" in str(error.value.__cause__)


@pytest.mark.parametrize("filename,accession", [
    ("run.mzML", "MS:1000584"), ("run.d/", "MS:1002817"), ("run.RAW", "MS:1000563"),
])
def test_mzqc_records_real_input_format(filename, accession):
    doc = json.loads(build_mzqc([{"filename": filename, "metrics": {}, "instrument_metadata": {}}]))
    assert doc["mzQC"]["runQualities"][0]["metadata"]["inputFiles"][0]["fileFormat"]["accession"] == accession
    assert doc["mzQC"]["runQualities"][0]["metadata"]["inputFiles"][0]["name"] == "run"


def test_native_bruker_tdf_cli(tmp_path):
    """Exercise the real nightly reader, binary data and mzQC output together."""
    path = Path(__file__).parent / "data/bruker_dia.d"
    output = tmp_path / "bruker.mzQC"
    result = CliRunner().invoke(main, [str(path) + "/", "--no-plot", "-o", str(output)])
    assert result.exit_code == 0, result.output
    doc = json.loads(output.read_text())
    schema = json.loads((Path(__file__).parent / "data/mzqc_schema.json").read_text())
    jsonschema.validate(doc, schema)
    run = doc["mzQC"]["runQualities"][0]
    assert run["metadata"]["inputFiles"][0]["fileFormat"]["accession"] == "MS:1002817"
    metrics = {m["name"]: m.get("value") for m in run["qualityMetrics"]}
    assert metrics["NumberOfSpectra_MS1"] == 2
    assert metrics["NumberOfSpectra_MS2"] == 8
    assert metrics["DIA_IsolationWindow_Count"] == 4
    assert metrics["DIA_Bruker_ExpectedMS2SpectrumCount"] == 8
    assert metrics["DIA_Bruker_ReaderOmittedMS2SpectrumCount"] == 0
    assert metrics["DIA_IsolationWindow_WidthRange"] == [50.0, 50.0]
    assert all(v is not None for v in metrics["DIA_IsolationWindow_Summary"]["ion_mobility_lower"])
    assert "PrecursorIntensity_FallbackCount" not in metrics


def test_mzml_cli_and_continue_on_error(tmp_path):
    path = tmp_path / "dia.mzML"
    oms.MzMLFile().store(str(path), _dia_experiment())
    output = tmp_path / "result.mzQC"
    invalid = tmp_path / "other.d"
    invalid.mkdir()
    result = CliRunner().invoke(main, [str(invalid), str(path), "--acquisition-mode", "dia",
                                      "--continue-on-error", "--no-plot", "-o", str(output)])
    assert result.exit_code == 1, result.output
    doc = json.loads(output.read_text())
    assert len(doc["mzQC"]["runQualities"]) == 1
    metrics = {m["name"]: m.get("value") for m in doc["mzQC"]["runQualities"][0]["qualityMetrics"]}
    assert metrics["DIA_IsolationWindow_Count"] == 2
    assert output.with_suffix(".tsv").is_file()
