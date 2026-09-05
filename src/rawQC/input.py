"""Input readers and acquisition-mode selection, independent of QC metrics."""

from pathlib import Path
from contextlib import closing
import sqlite3
from .reader_metadata import attach_bruker_metadata, attach_mzml_units

import pyopenms as oms


INPUT_FORMATS = {
    ".mzml": ("MS:1000584", "mzML format"),
    ".d": ("MS:1002817", "Bruker TDF format"),
    ".raw": ("MS:1000563", "Thermo RAW format"),
}


def _bruker_is_dia(path):
    """Use the same DIA acquisition tables as BrukerTimsFile, even if all peaks are empty."""
    uri = (path / "analysis.tdf").resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        for table in ("DiaFrameMsMsWindows", "DiaFrameMsMsInfo"):
            if table in tables and connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                return True
    return False


def input_format(filename):
    """Return the PSI-MS input format; a .d suffix denotes supported TDF only."""
    suffix = Path(filename).suffix.lower()
    if suffix not in INPUT_FORMATS:
        raise ValueError(f"Unsupported input {filename!s}; expected .mzML, Bruker TDF .d, or Thermo .raw")
    return INPUT_FORMATS[suffix]


def load_experiment(filename):
    """Load mzML, Bruker TDF, or Thermo RAW if the installed build supports it.

    Bruker AUTO export retains individual DIA isolation windows and per-peak
    ion mobility. Default reader settings do not request cross-frame averaging,
    denoising or optional centroiding. TSF and other vendor .d directories are
    not TDF inputs and are rejected before invoking the reader.
    """
    path = Path(filename)
    input_format(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".d":
        if not path.is_dir():
            raise ValueError(f"Bruker .d input must be a directory: {path}")
        if not all((path / name).is_file() for name in ("analysis.tdf", "analysis.tdf_bin")):
            raise ValueError(
                f"{path} is not a supported Bruker TDF dataset: analysis.tdf and "
                "analysis.tdf_bin are required. TSF and other .d formats are not supported."
            )
        reader_type = getattr(oms, "BrukerTimsFile", None)
        if reader_type is None:
            raise RuntimeError(
                "This pyOpenMS build has no BrukerTimsFile reader. Install the pinned "
                "nightly from https://pypi.openms.de/simple/ or convert to mzML."
            )
        # The Python binding returns the experiment (unlike the C++ API).
        is_dia = _bruker_is_dia(path)
        reader = reader_type()
        exp = reader.load(str(path))
        attach_bruker_metadata(path, exp)
        if is_dia:
            # The loader omits empty frames/windows. Retain the acquisition
            # metadata counts so QC can report these reader omissions separately.
            metadata, _ = reader.readDIAMetadata(str(path))
            exp.setMetaValue("acquisition_mode", "dia")
            exp.setMetaValue("rawqc_bruker_expected_ms1_spectra", int(metadata.nr_ms1_spectra))
            exp.setMetaValue("rawqc_bruker_expected_ms2_spectra", int(sum(metadata.nr_ms2_spectra)))
    else:
        if not path.is_file():
            raise ValueError(f"Expected a {suffix} file, not a directory: {path}")
        exp = oms.MSExperiment()
        if suffix == ".mzml":
            oms.MzMLFile().load(str(path), exp)
            attach_mzml_units(path, exp)
        else:
            # RAW is recognized even in builds without Thermo support, so neither
            # FileHandler.isSupported nor the FileTypes enum is a feature check.
            try:
                oms.FileHandler().loadExperiment(str(path), exp)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Could not read Thermo RAW input {path}: {exc}. Native Thermo "
                    "loading requires a pyOpenMS build with WITH_THERMO_RAW and its "
                    ".NET runtime. The pinned Linux/Windows wheels disable this "
                    "reader; convert to mzML on those platforms."
                ) from exc
    exp.updateRanges()
    return exp


def resolve_acquisition_mode(exp, acquisition_mode="auto"):
    """Select DIA from explicit metadata; otherwise retain DDA behavior.

    Window width or repeated precursor m/z alone cannot distinguish DIA from
    targeted acquisition. Unannotated DIA (including converted files) therefore
    needs an explicit ``dia`` override. BrukerTimsFile's DIA native IDs carry
    ``windowGroup=``; all MS2 spectra must carry that marker for this fallback.
    """
    mode = acquisition_mode.lower()
    if mode not in {"auto", "dda", "dia"}:
        raise ValueError("acquisition_mode must be 'auto', 'dda', or 'dia'")
    if mode != "auto":
        return mode

    def annotated_dia(obj):
        for key in ("MS:1003215", "data-independent acquisition"):
            if obj.metaValueExists(key):
                value = str(obj.getMetaValue(key)).lower()
                if value not in {"false", "0", "no"}:
                    return True
        for key in ("acquisition_mode", "acquisition mode"):
            if obj.metaValueExists(key):
                if str(obj.getMetaValue(key)).lower() in {"dia", "diapasef", "dia-pasef", "swath"}:
                    return True
        return False

    if annotated_dia(exp):
        return "dia"
    ms2_count = 0
    dia_count = 0
    for spectrum in exp:
        if spectrum.getMSLevel() == 2:
            ms2_count += 1
            if annotated_dia(spectrum) or "windowGroup=" in spectrum.getNativeID():
                dia_count += 1
    if ms2_count and dia_count == ms2_count:
        return "dia"
    if dia_count:
        raise ValueError(
            "Mixed DIA and unannotated MS2 spectra: select --acquisition-mode explicitly "
            "or split the acquisition before computing method-specific metrics."
        )
    return "dda"
