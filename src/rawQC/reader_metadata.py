"""Preserve source metadata that the Python reader interface does not expose.

TDF access is read-only. The mzML pass streams XML and discards binary arrays;
it retains parameter units lost when OpenMS DataValues become Python scalars.
"""

from contextlib import closing
import json
import sqlite3
import xml.etree.ElementTree as ET


def attach_bruker_metadata(path, exp):
    uri = (path / "analysis.tdf").resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row[0].lower(): row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "frames" in tables:
            # Fixed SQL identifiers: never interpolate vendor metadata into SQL.
            frames = [dict(row) for row in connection.execute("SELECT * FROM Frames ORDER BY Id")]
            exp.setMetaValue("rawqc_bruker_frames_json", json.dumps(frames))
            # TDF accumulation/ramp durations use ms, while Frames.Time uses s.
            # See from_tdf_connection / bottleneck_time_ms in:
            # https://github.com/TalusBio/tdf_simulator/blob/c55e6a1d03607ac214f5dcab532e8324337c79ec/tdf_simulator/config.py
            # Its frames.py divides max(accumulation, ramp) by 1000 for Time.
            exp.setMetaValue("rawqc_bruker_frame_units_json", json.dumps({
                "Time": "UO:0000010", "AccumulationTime": "UO:0000028",
                "RampTime": "UO:0000028"}))
        if "globalmetadata" in tables:
            metadata = {row[0]: row[1] for row in connection.execute(
                "SELECT Key, Value FROM GlobalMetadata")}
            exp.setMetaValue("rawqc_bruker_global_metadata_json", json.dumps(metadata))


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _parameters(element, groups):
    for child in element:
        tag = _tag(child)
        if tag in ("cvParam", "userParam"):
            yield child.attrib
        elif tag == "referenceableParamGroupRef":
            yield from groups.get(child.get("ref"), ())


def _spectrum_units(element, groups):
    units = {}

    def collect(node, source):
        for parameter in _parameters(node, groups):
            unit = parameter.get("unitAccession") or parameter.get("unitName")
            for key in (parameter.get("accession"), parameter.get("name")):
                if key:
                    path = source + ":" + key
                    # Conflicting repeat annotations must not choose a unit.
                    units[path] = unit if path not in units or units[path] == unit else None

    collect(element, "spectrum")
    for child in element:
        if _tag(child) == "scanList":
            collect(child, "acquisition_info")
            scans = [node for node in child if _tag(node) == "scan"]
            for index, scan in enumerate(scans):
                collect(scan, f"acquisition[{index}]")
        elif _tag(child) == "precursorList":
            precursors = [node for node in child if _tag(node) == "precursor"]
            for index, precursor in enumerate(precursors):
                for node in precursor.iter():
                    collect(node, f"precursor[{index}]")
    return units


def attach_mzml_units(path, exp):
    """Match units by spectrum order AND native ID, including reference groups.

    Null explicitly represents a missing/ambiguous XML unit, rather than the
    canonical-unit convention used for untracked in-memory OpenMS metadata.
    The parser keeps at most one spectrum's XML tree; peak arrays are discarded.
    """
    groups = {}
    stack = []
    index = 0
    with open(path, "rb") as stream:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            tag = _tag(element)
            if event == "start":
                stack.append(element)
                continue
            if tag == "referenceableParamGroup":
                groups[element.get("id")] = list(_parameters(element, groups))
            elif tag == "spectrum":
                if index >= exp.size() or exp[index].getNativeID() != element.get("id", ""):
                    raise ValueError("mzML spectrum order/native IDs do not match unit metadata")
                spectrum = exp[index]
                spectrum.setMetaValue("rawqc_metadata_units_json",
                                      json.dumps(_spectrum_units(element, groups)))
                exp[index] = spectrum
                index += 1
            # Retain spectrum metadata until its end; discard large binary text
            # immediately and remove finished top-level records from the tree.
            within_record = any(_tag(node) in ("spectrum", "referenceableParamGroup")
                                for node in stack[:-1])
            if tag in ("binary", "spectrum", "referenceableParamGroup") or not within_record:
                if len(stack) > 1:
                    stack[-2].remove(element)
                element.clear()
            stack.pop()
    if index != exp.size():
        raise ValueError("mzML spectrum count does not match unit metadata")
