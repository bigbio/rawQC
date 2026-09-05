"""Recorded acquisition metadata and identification-free ion-filling summaries.

No peak-derived resolving power or ion count is inferred from scan settings.
The Python MetaInfo interface discards DataValue units. Input readers can retain
those units in ``rawqc_metadata_units_json`` (``source:key`` -> unit accession),
and an explicit null there prevents a canonical-unit fallback. For direct
MSExperiment callers only, the known OpenMS ion-injection-time keys use their
canonical millisecond convention. Ambiguous combined scans remain unnormalized.
"""

import json
import math
import re
from collections import Counter

import numpy as np
import pyopenms as oms

from .dia import _finite_float, _isolation_window


RT_BIN_SECONDS = 60.0
MAX_IT_RELATIVE_TOLERANCE = 1e-6
MAX_IT_ABSOLUTE_TOLERANCE_MS = 1e-6

_DESCRIPTIONS = {
    "Acquisition_ScanMetadata": (
        "Column-oriented metadata table in loaded spectrum order (zero-based spectrum_index). "
        "Retention time is seconds; native frame IDs are parsed only from frame= tokens. "
        "For Bruker precursor= spectra the frame ID is only the first contributing "
        "frame; per-spectrum frame timing is therefore null. "
        "Isolation bounds are m/z and FAIMS CV is volts. Ion injection and maximum "
        "injection times are milliseconds only when units are established. AGC target "
        "units remain null if unrecorded; normalized target percent is not an ion count. "
        "Activation energy in eV, collision energy in eV and normalized collision "
        "energy in percent are separate. Structured activation energy defaults of "
        "zero are missing unless a source parameter explicitly establishes zero. "
        "Reported mass resolving power is dimensionless and distinct from mass resolution. "
        "Bruker accumulation/ramp times are joined from frame metadata without treating "
        "them as Thermo ion injection times. Missing, invalid, unit-unknown or ambiguous "
        "values are null with status columns; OpenMS numeric defaults are not measurements."
    ),
    "Acquisition_MetadataSources": (
        "Recorded source parameters for normalized scan fields: source path, exact key, "
        "raw value, raw unit, normalization status and candidate normalized value. "
        "An explicit missing XML unit overrides canonical conventions; an exact "
        "unit-bearing parameter label, such as Max. Ion Time (ms), still supplies "
        "a recorded unit. For in-memory "
        "experiments without retained source units, MS:1000927/ion injection time use "
        "the OpenMS millisecond convention. Multiple conflicting candidates and multiple "
        "acquisition entries for actual injection time are retained but not summarized."
    ),
    "Acquisition_BrukerFrameMetadata": (
        "Original Bruker Frames table columns, one row per acquired frame, plus "
        "rawqc_loaded_spectrum_count and normalized accumulation_time_ms/ramp_time_ms. "
        "Frame NumPeaks and SummedIntensities are vendor metadata and are not repeated "
        "as spectrum counts or TICs. Zero loaded spectra means reader omission only "
        "when every loaded spectrum maps unambiguously to a known unique frame. "
        "Null without native frame metadata."
    ),
    "Acquisition_BrukerFrameSummary": (
        "Single-row table of acquired frame count, loaded frame count and reader-omitted "
        "frame count, with null omission counts when native IDs cannot be mapped. "
        "Acquisition-order frame RT transition counts, median positive interval in "
        "seconds and population CV (at least two positive intervals). Nonconsecutive "
        "numeric frame IDs are reported as transitions and do not prove missing acquisition."
    ),
    "IonInjectionTime_Summary": (
        "Ion injection time distributions in milliseconds, grouped by MS level, FAIMS "
        "voltage and, for DIA only, full recorded isolation-window identity including "
        "mobility. DDA is not grouped by selected precursor. Quantiles use linear "
        "interpolation. Counts identify scans with valid actual time, positive known "
        "maximum and both. Fraction at maximum uses paired scans only and actual >= "
        "maximum within relative 1e-6 / absolute 1e-6 ms tolerance; above-maximum "
        "values are also counted separately. Missing metadata is never zero fill time."
    ),
    "IonInjectionTime_RTSummary": (
        "The same ion-filling summaries additionally grouped into fixed 60-second "
        "half-open bins [start,end), aligned to RT zero. Spectra with missing, "
        "nonfinite or negative RT are excluded from this table only."
    ),
}
ACQUISITION_METRIC_METADATA = {
    name: {"accession": None, "description": description}
    for name, description in _DESCRIPTIONS.items()
}


def _json_value(value):
    """Keep raw metadata JSON-safe without hiding nonfinite values as measurements."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def _json_meta(obj, key, default):
    if not obj.metaValueExists(key):
        return default
    try:
        return json.loads(obj.getMetaValue(key))
    except (ValueError, TypeError, UnicodeDecodeError):
        return default


def _table(rows, columns):
    return {key: [row.get(key) for row in rows] for key in columns}


def _key(value):
    return str(value).strip().rstrip(":").strip().casefold()


def _unit(value):
    if value is None:
        return None
    normalized = _key(value)
    return {
        "uo:0000028": "ms", "millisecond": "ms", "milliseconds": "ms", "ms": "ms",
        "uo:0000010": "s", "second": "s", "seconds": "s", "s": "s",
        "uo:0000031": "min", "minute": "min", "minutes": "min", "min": "min",
        "uo:0000266": "eV", "electronvolt": "eV", "electronvolts": "eV", "ev": "eV",
        "uo:0000187": "%", "percent": "%", "percentage": "%", "%": "%",
        "ms:1000131": "number of detector counts", "number of detector counts": "number of detector counts",
        "count": "count", "counts": "count", "ion": "ions", "ions": "ions",
        "dimensionless": "dimensionless", "uo:0000186": "dimensionless",
    }.get(normalized, str(value))


# Only explicit labels or known CV conventions carry default units. An unknown
# AGC-target unit is deliberately retained as unknown, never interpreted as ions.
_FIELD_SPECS = {
    "ion_injection_time_ms": (
        {"MS:1000927": "ms", "ion injection time": "ms", "Ion Injection Time (ms)": "ms",
         "Ion Injection Time [ms]": "ms"}, "ms"),
    "max_ion_injection_time_ms": (
        {"Maximum Ion Injection Time (ms)": "ms", "Max Ion Injection Time (ms)": "ms",
         "Max. Ion Time (ms)": "ms", "Max Ion Time (ms)": "ms",
         "Maximum Injection Time (ms)": "ms", "Max Injection Time (ms)": "ms",
         "maximum ion injection time": None, "maximum injection time": None,
         "Max. Ion Time": None, "Max Ion Time": None}, "ms"),
    "agc_target": (
        {"AGC Target": None, "AGC Target (ions)": "ions", "AGC Target (%)": "%",
         "Normalized AGC Target (%)": "%", "Normalized AGC Target": "%"}, None),
    "activation_energy_ev": ({"MS:1000509": "eV", "activation energy": "eV"}, "eV"),
    "collision_energy_ev": ({"MS:1000045": "eV", "collision energy": "eV",
                             "Collision Energy (eV)": "eV"}, "eV"),
    "normalized_collision_energy_percent": (
        {"MS:1000138": "%", "percent collision energy": "%",
         "Normalized Collision Energy (%)": "%", "HCD Energy (%)": "%"}, "%"),
    "reported_mass_resolving_power": (
        {"MS:1000800": "dimensionless", "mass resolving power": "dimensionless",
         "Orbitrap Resolution": "dimensionless", "FT Resolution": "dimensionless",
         "FT Resolution Setting": "dimensionless"}, "dimensionless"),
    "reported_mass_resolution": ({"MS:1000011": None, "mass resolution": None}, None),
}
_FIELD_SPECS = {field: ({_key(k): u for k, u in aliases.items()}, target)
                for field, (aliases, target) in _FIELD_SPECS.items()}


def _sources(spectrum):
    yield "spectrum", spectrum
    info = spectrum.getAcquisitionInfo()
    yield "acquisition_info", info
    for index in range(info.size()):
        yield f"acquisition[{index}]", info[index]
    yield "instrument_settings", spectrum.getInstrumentSettings()
    for index, precursor in enumerate(spectrum.getPrecursors()):
        yield f"precursor[{index}]", precursor


def _extract_fields(spectrum, index, records):
    units = _json_meta(spectrum, "rawqc_metadata_units_json", {})
    if not isinstance(units, dict):
        units = {}
    candidates = {field: [] for field in _FIELD_SPECS}
    for source, obj in _sources(spectrum):
        keys = []
        obj.getKeys(keys)
        for raw_key in keys:
            raw_key = _json_value(raw_key)
            for field, (aliases, target) in _FIELD_SPECS.items():
                if _key(raw_key) not in aliases:
                    continue
                raw = _json_value(obj.getMetaValue(raw_key))
                marker = f"{source}:{raw_key}"
                known_unit = marker in units or raw_key in units
                raw_unit = units.get(marker, units.get(raw_key)) if known_unit else aliases[_key(raw_key)]
                unit_source = "retained_source_unit" if known_unit else "openms_key_convention"
                label_has_unit = bool(re.search(r"[\[(](?:ms|eV|ions|%)[\])]\s*:?$", raw_key, re.IGNORECASE))
                if label_has_unit and (raw_unit is None or not known_unit):
                    raw_unit = aliases[_key(raw_key)]
                    unit_source = "parameter_label"
                # Direct API users may specify companion unit metadata as well.
                if not known_unit:
                    for companion in (f"{raw_key} unit", f"{raw_key}_unit"):
                        if obj.metaValueExists(companion):
                            raw_unit = _json_value(obj.getMetaValue(companion))
                            unit_source = "companion_metadata"
                            break
                normalized_unit = _unit(raw_unit)
                number = _finite_float(raw)
                status = "recorded"
                if number is None or number < 0 or isinstance(raw, bool):
                    number, status = None, "invalid"
                elif target == "ms":
                    scale = {"ms": 1, "s": 1000, "min": 60000}.get(normalized_unit)
                    if scale is None:
                        number, status = None, "unit_unknown" if normalized_unit is None else "unit_unsupported"
                    else:
                        number = _finite_float(number * scale)
                        if number is None:
                            status = "invalid"
                elif target is not None and normalized_unit != target:
                    number, status = None, "unit_unknown" if normalized_unit is None else "unit_unsupported"
                if field == "max_ion_injection_time_ms" and number == 0:
                    number, status = None, "invalid"
                candidate = {
                    "spectrum_index": index, "field": field, "source": source,
                    "key": raw_key, "raw_value": raw, "raw_unit": raw_unit,
                    "unit_source": unit_source if raw_unit is not None else None,
                    "normalization_status": status, "normalized_value": number,
                }
                records.append(candidate)
                candidates[field].append((number, target or normalized_unit, status))
    # MS:1000509 is parsed into a dedicated getter, so its original CV parameter
    # no longer appears in getKeys(). The API documents eV, but its default zero
    # has no presence indicator. Retained XML unit keys establish explicit zero.
    for precursor_index, precursor in enumerate(spectrum.getPrecursors()):
        source = f"precursor[{precursor_index}]"
        energy = precursor.getActivationEnergy()
        marker = next((f"{source}:{key}" for key in ("MS:1000509", "activation energy")
                       if f"{source}:{key}" in units), None)
        if energy == 0 and marker is None:
            continue
        raw_unit = units[marker] if marker is not None else "eV"
        number = _finite_float(energy)
        status = "recorded"
        if number is None or number < 0:
            number, status = None, "invalid"
        elif _unit(raw_unit) != "eV":
            number, status = None, "unit_unknown" if raw_unit is None else "unit_unsupported"
        records.append({
            "spectrum_index": index, "field": "activation_energy_ev", "source": source,
            "key": "getActivationEnergy()", "raw_value": _json_value(energy), "raw_unit": raw_unit,
            "unit_source": ("retained_source_unit" if marker is not None else "openms_api_unit") if raw_unit is not None else None,
            "normalization_status": status, "normalized_value": number,
        })
        candidates["activation_energy_ev"].append((number, "eV", status))
    result = {}
    for field, options in candidates.items():
        usable = [(value, unit) for value, unit, status in options if status == "recorded"]
        # An actual fill time is not uniquely a spectrum property after combining
        # multiple acquisition entries. Preserve all raw candidates without adding.
        ambiguous = (field == "ion_injection_time_ms" and spectrum.getAcquisitionInfo().size() > 1
                     and bool(options))
        if ambiguous or len(set(usable)) > 1:
            value, unit, status = None, None, "ambiguous"
        elif usable and len(usable) == len(options):
            value, unit = usable[0]
            status = "recorded"
        else:
            value, unit = None, None
            status = options[0][2] if len(options) == 1 else "ambiguous" if options else "missing"
        result[field] = value
        result[f"{field}_status"] = status
        if field in {"agc_target", "reported_mass_resolution"}:
            result[f"{field}_unit"] = unit
    return result


_WINDOW_COLUMNS = (
    "isolation_lower_mz", "isolation_upper_mz", "drift_time", "drift_time_unit",
    "ion_mobility_lower", "ion_mobility_upper", "precursor_drift_time",
    "precursor_drift_lower_offset", "precursor_drift_upper_offset",
)
_FRAME_RE = re.compile(r"(?:^|\s)frame=(\d+)(?=\s|$)")


def _frame_id(native_id):
    matches = _FRAME_RE.findall(native_id)
    return int(matches[0]) if len(matches) == 1 else None


def _frame_data(exp):
    frames = _json_meta(exp, "rawqc_bruker_frames_json", None)
    if not isinstance(frames, list) or any(not isinstance(row, dict) for row in frames):
        return None, {}, None
    frame_units = _json_meta(exp, "rawqc_bruker_frame_units_json", {})
    if not isinstance(frame_units, dict):
        frame_units = {}
    ids = [row.get("Id") for row in frames]
    valid_ids = all(isinstance(value, int) and not isinstance(value, bool) for value in ids)
    unique_ids = valid_ids and len(set(ids)) == len(ids)
    frame_map = {row["Id"]: row for row in frames} if unique_ids else {}
    native_ids = [spectrum.getNativeID() for spectrum in exp]
    loaded_ids = [_frame_id(native_id) for native_id in native_ids]
    aggregated = any(re.search(r"(?:^|\s)precursor=", native_id) for native_id in native_ids)
    complete_mapping = unique_ids and not aggregated and all(value in frame_map for value in loaded_ids)
    loaded_counts = Counter(loaded_ids)
    enriched = []
    for raw in frames:
        row = dict(raw)
        row["rawqc_loaded_spectrum_count"] = loaded_counts[raw.get("Id")] if complete_mapping else None
        for original, normalized in (("AccumulationTime", "accumulation_time_ms"),
                                     ("RampTime", "ramp_time_ms")):
            value = _finite_float(raw.get(original))
            scale = {"ms": 1, "s": 1000, "min": 60000}.get(_unit(frame_units.get(original)))
            row[normalized] = (_finite_float(value * scale)
                               if value is not None and value >= 0 and scale is not None else None)
        enriched.append(row)
    times = [_finite_float(row.get("Time")) for row in frames]
    intervals = [right - left for left, right in zip(times, times[1:])
                 if left is not None and right is not None]
    positive = [value for value in intervals if value > 0 and math.isfinite(value)]
    summary = {
        "acquisition_frame_count": len(frames),
        "loaded_frame_count": len(set(loaded_ids)) if complete_mapping else None,
        "reader_omitted_frame_count": sum(loaded_counts[value] == 0 for value in ids) if complete_mapping else None,
        "unmapped_loaded_spectrum_count": sum(value not in frame_map for value in loaded_ids),
        "nonfinite_rt_frame_count": sum(time is None for time in times),
        "nonincreasing_rt_transition_count": sum(value <= 0 for value in intervals),
        "positive_interval_count": len(positive),
        "positive_interval_median_seconds": _finite_float(np.median(positive)) if positive else None,
        "positive_interval_cv": _finite_float(np.std(positive) / np.mean(positive)) if len(positive) > 1 else None,
        "nonconsecutive_id_transition_count": sum(right != left + 1 for left, right in zip(ids, ids[1:])) if valid_ids else None,
    }
    enriched_map = {row["Id"]: row for row in enriched} if unique_ids else {}
    original_columns = sorted({key for row in frames for key in row})
    return _table(enriched, original_columns + ["rawqc_loaded_spectrum_count", "accumulation_time_ms", "ramp_time_ms"]), enriched_map, _table([summary], summary)


_GROUP_COLUMNS = ("ms_level", "faims_cv_volts", "isolation_metadata_status") + _WINDOW_COLUMNS
_STATS_COLUMNS = (
    "spectrum_count", "actual_time_count", "positive_max_time_count", "paired_time_count",
    "at_max_time_count", "above_max_time_count", "fraction_at_max_time",
    "injection_time_min_ms", "injection_time_q25_ms", "injection_time_median_ms",
    "injection_time_q75_ms", "injection_time_p95_ms", "injection_time_max_ms",
)


def _group_key(row, mode):
    window = tuple(row[column] for column in _WINDOW_COLUMNS) if mode == "dia" and row["ms_level"] > 1 else (None,) * len(_WINDOW_COLUMNS)
    status = row["isolation_metadata_status"] if mode == "dia" and row["ms_level"] > 1 else None
    return (row["ms_level"], row["faims_cv_volts"], status) + window


def _summarize(rows, mode, by_rt=False):
    groups = {}
    for row in rows:
        group = _group_key(row, mode)
        if by_rt:
            rt = row["retention_time_seconds"]
            if rt is None or rt < 0:
                continue
            start = math.floor(rt / RT_BIN_SECONDS) * RT_BIN_SECONDS
            group += (start, start + RT_BIN_SECONDS)
        groups.setdefault(group, []).append(row)
    columns = _GROUP_COLUMNS + (("rt_bin_start_seconds", "rt_bin_end_seconds") if by_rt else ())
    summaries = []
    for key, scans in groups.items():
        actual = [row["ion_injection_time_ms"] for row in scans if row["ion_injection_time_ms"] is not None]
        paired = [(row["ion_injection_time_ms"], row["max_ion_injection_time_ms"])
                  for row in scans if row["ion_injection_time_ms"] is not None
                  and row["max_ion_injection_time_ms"] is not None]
        at_max = sum(value >= maximum or math.isclose(value, maximum,
                         rel_tol=MAX_IT_RELATIVE_TOLERANCE, abs_tol=MAX_IT_ABSOLUTE_TOLERANCE_MS)
                     for value, maximum in paired)
        above_max = sum(value > maximum and not math.isclose(value, maximum,
                            rel_tol=MAX_IT_RELATIVE_TOLERANCE, abs_tol=MAX_IT_ABSOLUTE_TOLERANCE_MS)
                        for value, maximum in paired)
        quantiles = [_finite_float(value) for value in np.quantile(actual, [0, .25, .5, .75, .95, 1])] if actual else [None] * 6
        result = dict(zip(columns, key))
        result.update(zip(_STATS_COLUMNS, [len(scans), len(actual),
            sum(row["max_ion_injection_time_ms"] is not None for row in scans),
            len(paired), at_max, above_max, at_max / len(paired) if paired else None] + quantiles))
        summaries.append(result)
    return _table(summaries, columns + _STATS_COLUMNS)


def compute_acquisition_metrics(exp, acquisition_mode="auto"):
    """Return custom tables for recorded scan/frame metadata and ion filling."""
    from .input import resolve_acquisition_mode

    mode = resolve_acquisition_mode(exp, acquisition_mode)
    frame_table, frames, frame_summary = _frame_data(exp)
    records, rows = [], []
    for index, spectrum in enumerate(exp):
        native_id = spectrum.getNativeID()
        frame_id = _frame_id(native_id)
        rt = _finite_float(spectrum.getRT())
        # The default OpenMS RT is -1. Negative RTs are not usable run times.
        rt = rt if rt is not None and rt >= 0 else None
        window, reason = _isolation_window(spectrum)
        drift_unit = spectrum.getDriftTimeUnit()
        drift_unit = int(getattr(drift_unit, "value", drift_unit))
        faims_enum = oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE
        faims_unit = int(getattr(faims_enum, "value", faims_enum))
        faims = _finite_float(spectrum.getDriftTime()) if drift_unit == faims_unit else None
        aggregated = bool(re.search(r"(?:^|\s)precursor=", native_id))
        row = {
            "spectrum_index": index, "native_id": native_id or None,
            "frame_id": frame_id,
            "frame_association_status": ("first_contributing_frame" if aggregated else "recorded") if frame_id is not None else "missing",
            "retention_time_seconds": rt,
            "ms_level": spectrum.getMSLevel(), "acquisition_entry_count": spectrum.getAcquisitionInfo().size(),
            "precursor_count": len(spectrum.getPrecursors()),
            "isolation_metadata_status": reason.lower() if reason else "recorded",
            "faims_cv_volts": faims,
        }
        row.update(zip(_WINDOW_COLUMNS, window or (None,) * len(_WINDOW_COLUMNS)))
        row.update(_extract_fields(spectrum, index, records))
        frame = frames.get(frame_id, {}) if not aggregated else {}
        row["accumulation_time_ms"] = frame.get("accumulation_time_ms")
        row["ramp_time_ms"] = frame.get("ramp_time_ms")
        rows.append(row)
    base_columns = ("spectrum_index", "native_id", "frame_id", "frame_association_status", "retention_time_seconds",
                    "ms_level", "acquisition_entry_count", "precursor_count", "isolation_metadata_status",
                    "faims_cv_volts") + _WINDOW_COLUMNS
    field_columns = []
    for field in _FIELD_SPECS:
        field_columns.extend((field, f"{field}_status"))
        if field in {"agc_target", "reported_mass_resolution"}:
            field_columns.append(f"{field}_unit")
    return {
        "Acquisition_ScanMetadata": _table(rows, base_columns + tuple(field_columns) + ("accumulation_time_ms", "ramp_time_ms")),
        "Acquisition_MetadataSources": _table(records, ("spectrum_index", "field", "source", "key", "raw_value", "raw_unit", "unit_source", "normalization_status", "normalized_value")),
        "Acquisition_BrukerFrameMetadata": frame_table,
        "Acquisition_BrukerFrameSummary": frame_summary,
        "IonInjectionTime_Summary": _summarize(rows, mode),
        "IonInjectionTime_RTSummary": _summarize(rows, mode, by_rt=True),
    }
