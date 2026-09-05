"""Recorded run, instrument, sample and processing provenance.

Only metadata exposed by the reader is reported. In particular, OpenMS sample
volume is sample volume in mL, not injection volume. Numeric default values that
cannot be distinguished from absent metadata are null. No spectra or peaks are
copied into the provenance payload.
"""

import json
import math


_DESCRIPTIONS = {
    "Provenance_Run": (
        "Recorded experiment identifier, fraction identifier, comment and acquisition "
        "datetime (OpenMS datetime string; no timezone inferred), plus explicitly "
        "named method-file metadata with its source. Missing fields are null."
    ),
    "Provenance_Instrument": (
        "Reader-provided instrument name, model, vendor, serial number, customizations, "
        "acquisition software and instrument component types. Names are not inferred "
        "from the file extension. Conflicting serial numbers remain in recorded "
        "metadata and yield a null normalized serial number."
    ),
    "Provenance_Sample": (
        "Recorded sample name, number, organism, comment, state, vial, mass in g, "
        "sample volume in mL and concentration in g/L. OpenMS default zero numeric "
        "fields are null because absence is indistinguishable from zero. Injection "
        "volume is separate and normalized to uL only for explicit unit-bearing "
        "metadata keys; values without a known unit remain in recorded metadata."
    ),
    "Provenance_SourceFiles": (
        "Source files recorded by the reader, including original name/path, format, "
        "native-ID format and checksum. These values describe recorded provenance; "
        "rawQC does not recompute checksums or establish that the source files exist."
    ),
    "Provenance_Processing": (
        "Distinct recorded data-processing entries from loaded spectra and "
        "chromatograms, with software, processing actions, completion datetime, "
        "metadata and occurrence counts. First indices are zero-based. Counts refer "
        "to objects carrying each entry, not executions of a processing operation."
    ),
    "Provenance_RecordedMetadata": (
        "Column-oriented table of reader-provided experiment, instrument/component, "
        "sample, source-file and software metadata and Bruker GlobalMetadata. "
        "Columns source, key, value, accession and unit preserve each originating "
        "object. Generic MetaInfo values do not expose units through the Python "
        "reader API; their units are null. CV-term units are retained when available. "
        "Bruker values remain strings. Internal rawqc_* metadata and native frame "
        "payloads are excluded; nonfinite values become null."
    ),
}

PROVENANCE_METRIC_METADATA = {
    name: {"accession": None, "description": description}
    for name, description in _DESCRIPTIONS.items()
}


def _text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) and value.strip() else None


def _json_safe(value):
    """Keep JSON-compatible values without inventing text for opaque objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return None


def _metadata(obj):
    try:
        keys = obj.getKeys()
    except TypeError:  # Some pyOpenMS classes use an output argument.
        keys = []
        obj.getKeys(keys)
    return {
        _text(key): _json_safe(obj.getMetaValue(key))
        for key in keys
        if _text(key) and not _text(key).startswith("rawqc_")
    }


def _datetime(value):
    return value.get() if value.isValid() and not value.isNull() else None


def _enum_name(value, names):
    index = int(getattr(value, "value", value))
    return names[index] if 0 < index < len(names) else None


def _positive(value):
    return float(value) if math.isfinite(value) and value > 0 else None


def _software(obj):
    return {"name": _text(obj.getName()), "version": _text(obj.getVersion())}


def _record_metadata(rows, source, obj):
    for key, value in _metadata(obj).items():
        rows.append({"source": source, "key": key, "value": value,
                     "accession": None, "unit": None})
    if hasattr(obj, "getCVTerms"):
        for terms in obj.getCVTerms().values():
            for term in terms:
                unit = term.getUnit()
                unit_value = {"accession": _text(unit.accession),
                              "name": _text(unit.name), "cv_ref": _text(unit.cv_ref)}
                rows.append({"source": source, "key": _text(term.getName()),
                             "value": _json_safe(term.getValue()),
                             "accession": _text(term.getAccession()),
                             "unit": unit_value if any(unit_value.values()) else None})


def _unique_recorded(rows, keys, sources):
    values = []
    for row in rows:
        if row["source"] in sources and (row["key"] in keys or row["accession"] in keys):
            value = row["value"]
            if value is not None and value != "" and value not in values:
                values.append(value)
    return values[0] if len(values) == 1 else None


def _bruker_globals(exp):
    key = "rawqc_bruker_global_metadata_json"
    if not exp.metaValueExists(key):
        return {}
    try:
        result = json.loads(exp.getMetaValue(key))
    except (TypeError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


def _processing(exp):
    entries = {}
    for kind, objects in (("spectrum", exp), ("chromatogram", exp.getChromatograms())):
        for index, obj in enumerate(objects):
            seen = set()
            for processing in obj.getDataProcessing():
                software = processing.getSoftware()
                metadata = []
                _record_metadata(metadata, "processing", processing)
                _record_metadata(metadata, "processing.software", software)
                actions = processing.getAllNamesOfProcessingAction()
                record = {
                    "software": _software(software),
                    "actions": [actions[int(getattr(action, "value", action))]
                                for action in sorted(processing.getProcessingActions(),
                                                     key=lambda a: int(getattr(a, "value", a)))],
                    "completion_datetime": _datetime(processing.getCompletionTime()),
                    "metadata": metadata,
                }
                key = json.dumps(record, sort_keys=True, allow_nan=False)
                if key in seen:
                    continue
                seen.add(key)
                if key not in entries:
                    entries[key] = dict(record, spectrum_count=0, chromatogram_count=0,
                                        first_spectrum_index=None, first_chromatogram_index=None)
                entry = entries[key]
                entry[kind + "_count"] += 1
                if entry["first_" + kind + "_index"] is None:
                    entry["first_" + kind + "_index"] = index
    return list(entries.values())


def compute_provenance_metrics(exp):
    """Extract provenance without modifying the experiment or interpreting peaks."""
    rows = []
    instrument, sample = exp.getInstrument(), exp.getSample()
    for source, obj in (("experiment", exp), ("instrument", instrument),
                        ("sample", sample), ("instrument.software", instrument.getSoftware())):
        _record_metadata(rows, source, obj)
    for key, value in _bruker_globals(exp).items():
        rows.append({"source": "bruker.GlobalMetadata", "key": key,
                     "value": _json_safe(value), "accession": None, "unit": None})

    components = {"ion_sources": [], "mass_analyzers": [], "ion_detectors": []}
    for kind, objects, getter, names in (
        ("ion_sources", instrument.getIonSources(), "getIonizationMethod", "getAllNamesOfIonizationMethod"),
        ("mass_analyzers", instrument.getMassAnalyzers(), "getType", "getAllNamesOfAnalyzerType"),
        ("ion_detectors", instrument.getIonDetectors(), "getType", "getAllNamesOfType"),
    ):
        for index, obj in enumerate(objects):
            source = f"instrument.{kind}[{index}]"
            components[kind].append({"source": source,
                                     "type": _enum_name(getattr(obj, getter)(), getattr(obj, names)())})
            _record_metadata(rows, source, obj)

    source_files = []
    for index, obj in enumerate(exp.getSourceFiles()):
        source = f"source_files[{index}]"
        source_files.append({
            "source": source, "name": _text(obj.getNameOfFile()),
            "path": _text(obj.getPathToFile()), "file_type": _text(obj.getFileType()),
            "native_id_type": _text(obj.getNativeIDType()),
            "native_id_type_accession": _text(obj.getNativeIDTypeAccession()),
            "checksum": _text(obj.getChecksum()),
            "checksum_type": _enum_name(obj.getChecksumType(), obj.getAllNamesOfChecksumType()),
        })
        _record_metadata(rows, source, obj)

    native = "bruker.GlobalMetadata"
    sample_sources = {"sample", "experiment", native}
    instrument_result = {
        "name": _text(instrument.getName()), "model": _text(instrument.getModel()),
        "vendor": _text(instrument.getVendor()),
        "serial_number": _unique_recorded(rows, {"instrument serial number", "MS:1000529",
                                                   "InstrumentSerialNumber", "SerialNumber"},
                                           {"instrument", "experiment", native}),
        "customizations": _text(instrument.getCustomizations()),
        "software": _software(instrument.getSoftware()), **components,
    }
    injection_values = []
    # Unit-free vendor fields are deliberately retained only in recorded metadata.
    for keys, factor in (({"injection volume (uL)", "injection volume (µL)",
                          "InjectionVolume (uL)", "InjectionVolume_uL"}, 1.0),
                         ({"injection volume (mL)", "InjectionVolume_mL"}, 1000.0)):
        for row in rows:
            if row["source"] not in sample_sources or row["key"] not in keys:
                continue
            try:
                value = float(row["value"]) * factor
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value) and value >= 0 and value not in injection_values:
                injection_values.append(value)
    injection_volume = injection_values[0] if len(injection_values) == 1 else None

    methods = [dict(row) for row in rows if row["key"] in {
        "InstrumentMethodFile", "ProcessingMethodFile", "AcquisitionMethodFile",
        "instrument method file", "processing method file", "acquisition method file",
    }]
    sample_result = {
        "name": _text(sample.getName()) or _unique_recorded(rows, {"SampleName"}, {native}),
        "number": _text(sample.getNumber()) or _unique_recorded(rows, {"SampleId", "SampleID"}, {native}),
        "organism": _text(sample.getOrganism()), "comment": _text(sample.getComment()),
        "state": _enum_name(sample.getState(), sample.getAllNamesOfSampleState()),
        "mass_g": _positive(sample.getMass()), "volume_mL": _positive(sample.getVolume()),
        "concentration_g_per_L": _positive(sample.getConcentration()),
        "vial": _unique_recorded(rows, {"Vial", "VialPosition", "vial position"}, sample_sources),
        "injection_volume_uL": injection_volume,
    }
    table = {key: [row[key] for row in rows]
             for key in ("source", "key", "value", "accession", "unit")}
    return {
        "Provenance_Run": {"identifier": _text(exp.getIdentifier()),
                           "fraction_identifier": _text(exp.getFractionIdentifier()),
                           "comment": _text(exp.getComment()),
                           "acquisition_datetime": _datetime(exp.getDateTime()),
                           "method_files": methods},
        "Provenance_Instrument": instrument_result,
        "Provenance_Sample": sample_result,
        "Provenance_SourceFiles": source_files,
        "Provenance_Processing": _processing(exp),
        "Provenance_RecordedMetadata": table,
    }
