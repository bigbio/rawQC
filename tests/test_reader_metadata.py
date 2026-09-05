"""Source metadata survives native loading and pyOpenMS unit erasure."""

import json
import sqlite3
import xml.etree.ElementTree as ET

import pyopenms as oms
import pytest

from rawQC.input import load_experiment
from rawQC.reader_metadata import attach_bruker_metadata, attach_mzml_units


def _exp(*ids):
    exp = oms.MSExperiment()
    for native_id in ids:
        spectrum = oms.MSSpectrum()
        spectrum.setNativeID(native_id)
        exp.addSpectrum(spectrum)
    return exp


def test_xml_units_reference_groups_and_multiple_acquisitions(tmp_path):
    path = tmp_path / "units.mzML"
    path.write_text('''<mzML xmlns="http://psi.hupo.org/ms/mzml">
      <referenceableParamGroupList><referenceableParamGroup id="fill">
        <cvParam accession="MS:1000927" name="ion injection time" value="0.012"
                 unitAccession="UO:0000010"/>
      </referenceableParamGroup></referenceableParamGroupList>
      <run><spectrumList><spectrum id="scan=1">
        <userParam name="Max. Ion Time (ms)" value="20" unitAccession="UO:0000028"/>
        <scanList><scan><referenceableParamGroupRef ref="fill"/></scan>
          <scan><cvParam accession="MS:1000927" name="ion injection time" value="3"/></scan>
        </scanList><binaryDataArrayList><binaryDataArray><binary>ignored</binary>
        </binaryDataArray></binaryDataArrayList></spectrum>
      <spectrum id="scan=2"><scanList><scan><userParam name="AGC Target" value="100"/>
      </scan></scanList></spectrum></spectrumList></run></mzML>''')
    exp = _exp("scan=1", "scan=2")
    attach_mzml_units(path, exp)
    units = json.loads(exp[0].getMetaValue("rawqc_metadata_units_json"))
    assert units["acquisition[0]:MS:1000927"] == "UO:0000010"
    assert units["acquisition[0]:ion injection time"] == "UO:0000010"
    assert units["acquisition[1]:MS:1000927"] is None
    assert units["spectrum:Max. Ion Time (ms)"] == "UO:0000028"
    assert json.loads(exp[1].getMetaValue("rawqc_metadata_units_json")) == {
        "acquisition[0]:AGC Target": None}


def test_xml_unit_alignment_is_checked(tmp_path):
    path = tmp_path / "mismatch.mzML"
    path.write_text('<mzML><run><spectrumList><spectrum id="scan=2"/></spectrumList></run></mzML>')
    with pytest.raises(ValueError, match="native IDs"):
        attach_mzml_units(path, _exp("scan=1"))


def test_native_optional_tables_and_read_only_access(tmp_path):
    path = tmp_path / "with space.d"
    path.mkdir()
    database = path / "analysis.tdf"
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE Frames (Id INTEGER, Time REAL, AccumulationTime REAL)')
        connection.execute('INSERT INTO Frames VALUES (7, 1.5, 100.0)')
        connection.execute('CREATE TABLE GlobalMetaData (Key TEXT, Value TEXT)')
        connection.execute("INSERT INTO GlobalMetaData VALUES ('InstrumentSerialNumber', 'A123')")
    before = database.read_bytes()
    exp = oms.MSExperiment()
    attach_bruker_metadata(path, exp)
    assert database.read_bytes() == before
    assert json.loads(exp.getMetaValue("rawqc_bruker_frames_json")) == [
        {"Id": 7, "Time": 1.5, "AccumulationTime": 100.0}]
    assert json.loads(exp.getMetaValue("rawqc_bruker_global_metadata_json")) == {
        "InstrumentSerialNumber": "A123"}


def test_mzml_loader_preserves_explicit_injection_time_units(tmp_path):
    path = tmp_path / "seconds.mzML"
    exp = _exp("scan=1")
    spectrum = exp[0]
    info = oms.AcquisitionInfo()
    acquisition = oms.Acquisition()
    acquisition.setMetaValue("ion injection time", 0.025)
    info.push_back(acquisition)
    spectrum.setAcquisitionInfo(info)
    exp[0] = spectrum
    oms.MzMLFile().store(str(path), exp)
    ET.register_namespace("", "http://psi.hupo.org/ms/mzml")
    tree = ET.parse(path)
    for parameter in tree.iter():
        if parameter.get("accession") == "MS:1000927":
            parameter.set("unitAccession", "UO:0000010")
            parameter.set("unitName", "second")
            parameter.set("unitCvRef", "UO")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    loaded = load_experiment(path)
    units = json.loads(loaded[0].getMetaValue("rawqc_metadata_units_json"))
    assert units["acquisition[0]:MS:1000927"] == "UO:0000010"
    from rawQC.acquisition import compute_acquisition_metrics
    assert compute_acquisition_metrics(loaded)["Acquisition_ScanMetadata"]["ion_injection_time_ms"] == [25.0]
