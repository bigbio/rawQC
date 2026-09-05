"""Identification-free metrics for spectra known to be acquired with DIA.

Acquisition-mode selection belongs to the caller. Isolation width alone cannot
reliably distinguish DIA from DDA. These are rawQC custom metrics, not PSI-MS
terms; fragment signal is never interpreted as selected-precursor intensity.
"""

import math
from statistics import mean, median, pstdev

import numpy as np
import pyopenms as oms


_DESCRIPTIONS = {
    "DIA_MS2_Count": "Number of loaded MS2 spectra in a run selected as DIA.",
    "DIA_MS2_TIC_Area": (
        "Trapezoidal integral in intensity-times-seconds of loaded MS2 fragment "
        "intensity sums, first summing all spectra with exactly the same finite "
        "retention time and then integrating across sorted distinct times. "
        "Spectra with nonfinite RT, any nonfinite fragment intensity or a "
        "nonfinite intensity sum are excluded. Isolation metadata is not "
        "required. Null with fewer than two distinct valid times. This metric "
        "describes the loaded spectra and is not a vendor detector TIC."
    ),
    "DIA_IsolationMetadata_MissingCount": (
        "Number of MS2 spectra without a precursor or with both isolation offsets "
        "equal to zero. These spectra are excluded from the window table."
    ),
    "DIA_IsolationMetadata_InvalidCount": (
        "Number of single-precursor MS2 spectra with nonfinite or invalid isolation "
        "bounds (nonpositive target or lower bound, negative offsets). These "
        "spectra are excluded from the window table."
    ),
    "DIA_IsolationMetadata_AmbiguousCount": (
        "Number of MS2 spectra with multiple precursor entries. Their fragment "
        "signal cannot be assigned to one window and is excluded from the window table."
    ),
    "DIA_IsolationWindow_Count": (
        "Number of distinct valid isolation windows, distinguished by m/z bounds "
        "rounded to six decimal places and recorded ion-mobility/FAIMS metadata."
    ),
    "DIA_IsolationWindow_MzRange": (
        "Minimum lower bound and maximum upper bound of valid DIA isolation "
        "windows in m/z. This envelope does not imply gap-free coverage."
    ),
    "DIA_IsolationWindow_WidthRange": (
        "Minimum and maximum width (upper minus lower m/z bound) of distinct "
        "valid DIA isolation windows. Overlapping windows remain separate."
    ),
    "DIA_IsolationWindow_Summary": (
        "Column-oriented table of distinct DIA isolation windows over loaded "
        "spectra; windows omitted by a native reader are not counted as empty. "
        "Columns: lower_mz, "
        "upper_mz, width_mz, drift_time, drift_time_unit (OpenMS enum value), "
        "ion_mobility_lower, ion_mobility_upper (reader-provided mobility bounds; "
        "Bruker uses inverse reduced mobility), precursor_drift_time, "
        "precursor_drift_lower_offset, precursor_drift_upper_offset (recorded "
        "precursor mobility selection, in its reported mobility unit), "
        "scan_count, empty_scan_count "
        "(finite fragment-intensity sum equals zero), invalid_tic_scan_count "
        "(any nonfinite fragment intensity), tic_sum and tic_median. TIC values "
        "sum fragment intensities, omit invalid scans, and are not normalized "
        "by isolation width or interpreted as precursor intensity. Missing "
        "mobility or TIC values are null."
    ),
    "DIA_IsolationWindow_RevisitSummary": (
        "Column-oriented table indexed by zero-based window_index in the isolation "
        "window summary. Adjacent observations of each window are compared in "
        "acquisition order without bridging nonfinite RTs. Positive finite intervals "
        "have count, median, minimum, maximum and population CV in seconds; equal-RT "
        "and decreasing-RT transitions are counted separately, so simultaneous "
        "diaPASEF windows do not create zero-duration cycles. Nonfinite RT count "
        "counts observations; invalid_interval_count counts transitions with an "
        "invalid endpoint or overflowing difference. long_interval_count counts "
        "positive intervals strictly above 1.5 times their median, only when at "
        "least three positive intervals exist. It flags potential sampling "
        "interruptions, not proven missing acquisitions or missing frames."
    ),
    "DIA_IsolationWindow_MzCoverage": (
        "Column-oriented table of run-wide observed m/z interval projections, "
        "grouped by mobility mode and reported OpenMS drift-time unit; each FAIMS "
        "voltage is separate. Identical projected intervals are deduplicated. "
        "covered_width_mz is their union, gap_width_mz is uncovered width inside "
        "the observed envelope, gap_count counts internal gaps, and overlap_width_mz "
        "is width covered by at least two distinct projected intervals (not excess "
        "multiplicity). coverage_fraction is union divided by envelope width. "
        "Projections across IM selections are not two-dimensional coverage and "
        "run-wide coverage does not imply a complete acquisition cycle."
    ),
    "DIA_IsolationWindow_MzIMCoverage": (
        "Column-oriented table of run-wide observed m/z–ion-mobility rectangle "
        "coverage for windows with a known common unit (milliseconds, inverse "
        "reduced mobility, or CCS). FAIMS and unknown units have no 2D rows. "
        "Reader-provided mobility lower/upper bounds take precedence; otherwise "
        "valid precursor drift-time offsets define bounds. Windows without "
        "valid nonnegative bounds are counted as excluded; exact rectangles "
        "are deduplicated. Areas have units m/z times the reported mobility unit. "
        "covered_area is the rectangle union, uncovered_area is the portion of "
        "the observed bounding rectangle outside that union, and overlap_area "
        "is area covered by at least two distinct rectangles. These geometric "
        "gaps can be intentional and are not claims of missing acquisitions; "
        "no intended method envelope or per-cycle completeness is inferred. "
        "Area and envelope values are null when no valid rectangles exist."
    ),
    "DIA_MS1BoundedCycle_Count": (
        "Number of acquisition-order intervals between consecutive MS1 spectra "
        "with increasing finite RT and at least one MS2 spectrum; every MS2 RT "
        "must lie within its bounding MS1 RTs. Unbounded leading/trailing MS2 "
        "spectra and invalid intervals are excluded. These intervals are MS1 "
        "survey cycles, which need not be complete DIA or FAIMS supercycles."
    ),
    "DIA_MS1BoundedCycle_DurationMedian": (
        "Median duration in seconds of valid MS1-bounded DIA survey cycles. "
        "Equal-RT MS2 spectra do not create separate cycle boundaries."
    ),
    "DIA_MS1BoundedCycle_DurationCV": (
        "Population standard deviation divided by mean duration of valid "
        "MS1-bounded DIA survey cycles; only emitted with at least two cycles."
    ),
    "DIA_MS1BoundedCycle_MS2CountRange": (
        "Minimum and maximum MS2 spectrum count per valid MS1-bounded DIA survey cycle."
    ),
    "DIA_MS1BoundedCycle_MS2CountMedian": (
        "Median MS2 spectrum count per valid MS1-bounded DIA survey cycle."
    ),
    "DIA_MS1BoundedCycle_WindowCountRange": (
        "Minimum and maximum distinct valid isolation-window count per valid "
        "MS1-bounded DIA survey cycle. Missing/ambiguous windows are excluded."
    ),
    "DIA_MS1BoundedCycle_WindowCountMedian": (
        "Median distinct valid isolation-window count per valid MS1-bounded DIA "
        "survey cycle. Missing/ambiguous windows are excluded."
    ),
    "DIA_Bruker_ExpectedMS1SpectrumCount": (
        "Number of MS1 frame spectra reported by Bruker DIA acquisition metadata, "
        "including spectra the native reader may omit when empty; null without "
        "reader-provided acquisition counts."
    ),
    "DIA_Bruker_ExpectedMS2SpectrumCount": (
        "Number of MS2 window spectra reported by Bruker DIA acquisition metadata, "
        "including spectra the native reader may omit when empty; null without "
        "reader-provided acquisition counts."
    ),
    "DIA_Bruker_ReaderOmittedMS1SpectrumCount": (
        "Bruker DIA metadata MS1 frame-spectrum count minus loaded MS1 spectrum "
        "count. Counts reader omissions, not missing acquisitions; null without "
        "metadata or if the metadata count is smaller than the loaded count."
    ),
    "DIA_Bruker_ReaderOmittedMS2SpectrumCount": (
        "Bruker DIA metadata MS2 window-spectrum count minus loaded MS2 spectrum "
        "count. Counts reader omissions, not missing acquisitions; null without "
        "metadata or if the metadata count is smaller than the loaded count."
    ),
}

DIA_METRIC_METADATA = {
    key: {"accession": None, "description": description}
    for key, description in _DESCRIPTIONS.items()
}


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _enum_value(value):
    return int(getattr(value, "value", value))


def _mobility_key(spectrum, precursor):
    """Keep native diaPASEF mobility ranges and FAIMS voltages distinct."""
    drift_time = _finite_float(spectrum.getDriftTime())
    unit = spectrum.getDriftTimeUnit()
    # Cython exposes an integer; direct bindings may expose an enum.
    unit = _enum_value(unit)
    faims_unit = _enum_value(oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE)
    if unit != faims_unit and drift_time == -1.0:  # OpenMS missing sentinel
        drift_time = None
    lower = upper = None
    if spectrum.metaValueExists("ion mobility lower limit"):
        lower = _finite_float(spectrum.getMetaValue("ion mobility lower limit"))
    if spectrum.metaValueExists("ion mobility upper limit"):
        upper = _finite_float(spectrum.getMetaValue("ion mobility upper limit"))
    precursor_drift = _finite_float(precursor.getDriftTime())
    precursor_lower = precursor_upper = None
    if precursor_drift == -1.0:
        precursor_drift = None
    if precursor_drift is not None:
        precursor_lower = _finite_float(precursor.getDriftTimeWindowLowerOffset())
        precursor_upper = _finite_float(precursor.getDriftTimeWindowUpperOffset())
    return (drift_time, unit if unit != 0 else None, lower, upper,
            precursor_drift, precursor_lower, precursor_upper)


def _isolation_window(spectrum):
    """Return a window key or a reason why fragment signal is unassignable."""
    precursors = spectrum.getPrecursors()
    if not precursors:
        return None, "Missing"
    if len(precursors) != 1:
        return None, "Ambiguous"
    precursor = precursors[0]
    center = _finite_float(precursor.getMZ())
    lower_offset = _finite_float(precursor.getIsolationWindowLowerOffset())
    upper_offset = _finite_float(precursor.getIsolationWindowUpperOffset())
    if lower_offset == 0 and upper_offset == 0:
        return None, "Missing"
    if (center is None or lower_offset is None or upper_offset is None
            or center <= 0 or lower_offset < 0 or upper_offset < 0
            or center - lower_offset <= 0):
        return None, "Invalid"
    lower = round(center - lower_offset, 6)
    upper = round(center + upper_offset, 6)
    if not math.isfinite(upper) or upper <= lower:
        return None, "Invalid"
    return (lower, upper) + _mobility_key(spectrum, precursor), None


def _window_sort_key(window):
    return tuple((value is not None, value if value is not None else 0)
                 for value in window)


def _revisit_summary(sorted_windows):
    columns = ("window_index", "observation_count", "nonfinite_rt_count",
               "positive_interval_count", "equal_rt_interval_count",
               "decreasing_rt_interval_count", "invalid_interval_count",
               "interval_min_seconds", "interval_median_seconds",
               "interval_max_seconds", "interval_cv", "long_interval_count")
    table = {column: [] for column in columns}
    for index, (_, record) in enumerate(sorted_windows):
        rts = record["rts"]
        intervals = []
        equal = decreasing = invalid = 0
        for previous, current in zip(rts, rts[1:]):
            interval = (_finite_float(current - previous)
                        if current is not None and previous is not None else None)
            if interval is None:
                invalid += 1
            elif interval == 0:
                equal += 1
            elif interval < 0:
                decreasing += 1
            else:
                intervals.append(interval)
        midpoint = median(intervals) if intervals else None
        values = (index, len(rts), sum(rt is None for rt in rts), len(intervals),
                  equal, decreasing, invalid, min(intervals) if intervals else None,
                  midpoint, max(intervals) if intervals else None,
                  pstdev(intervals) / mean(intervals) if len(intervals) > 1 else None,
                  sum(interval / midpoint > 1.5 for interval in intervals)
                  if len(intervals) >= 3 else None)
        for column, value in zip(columns, values):
            table[column].append(value)
    return table


def _coverage_group(window):
    """Keep unannotated, unknown-unit, IM and individual FAIMS selections apart."""
    _, _, drift, unit, im_lower, im_upper, prec_drift, _, _ = window
    if unit == _enum_value(oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE):
        return "faims", unit, drift
    known = {_enum_value(oms.DriftTimeUnit.MILLISECOND),
             _enum_value(oms.DriftTimeUnit.VSSC), _enum_value(oms.DriftTimeUnit.CCS)}
    if unit in known:
        return "ion_mobility", unit, None
    if unit is not None or any(value is not None for value in
                               (drift, im_lower, im_upper, prec_drift)):
        return "unknown_mobility", unit, None
    return "none", None, None


def _mobility_bounds(window, reader_bounds_present=False):
    """Use explicit reader bounds or, when absent, precursor selection offsets."""
    _, _, _, _, lower, upper, drift, lower_offset, upper_offset = window
    if lower is None and upper is None:
        if (reader_bounds_present or drift is None or lower_offset is None or upper_offset is None
                or lower_offset < 0 or upper_offset < 0):
            return None
        lower = _finite_float(drift - lower_offset)
        upper = _finite_float(drift + upper_offset)
    if lower is None or upper is None or lower < 0 or upper <= lower:
        return None
    return lower, upper


def _interval_measure(intervals):
    """Return union width, width with multiplicity >=2, and union components."""
    events = {}
    for lower, upper in intervals:
        events[lower] = events.get(lower, 0) + 1
        events[upper] = events.get(upper, 0) - 1
    covered = overlap = 0.0
    components = active = 0
    previous = None
    for position, delta in sorted(events.items()):
        if previous is not None:
            width = position - previous
            if active:
                covered += width
            if active >= 2:
                overlap += width
        if active == 0 and active + delta > 0:
            components += 1
        active += delta
        previous = position
    return covered, overlap, components


def _rectangle_measure(rectangles):
    """Integrate an IM interval union through m/z slabs without double counting."""
    events = {}
    for mz_lower, mz_upper, im_lower, im_upper in rectangles:
        events.setdefault(mz_lower, []).append((im_lower, im_upper, 1))
        events.setdefault(mz_upper, []).append((im_lower, im_upper, -1))
    active = {}
    previous = None
    covered = overlap = 0.0
    for position, changes in sorted(events.items()):
        if previous is not None and active:
            im_covered, im_overlap, _ = _interval_measure(
                interval for interval, count in active.items() for _ in range(count))
            covered += (position - previous) * im_covered
            overlap += (position - previous) * im_overlap
        for lower, upper, delta in changes:
            interval = lower, upper
            active[interval] = active.get(interval, 0) + delta
            if not active[interval]:
                del active[interval]
        previous = position
    return covered, overlap


def _coverage_summary(sorted_windows):
    mz_columns = ("mobility_mode", "drift_time_unit", "faims_cv", "window_count",
                  "distinct_mz_interval_count", "mz_lower", "mz_upper",
                  "envelope_width_mz", "covered_width_mz", "gap_width_mz",
                  "overlap_width_mz", "coverage_fraction", "gap_count")
    im_columns = ("mobility_mode", "drift_time_unit", "faims_cv", "window_count",
                  "rectangle_count", "excluded_window_count", "mz_lower", "mz_upper",
                  "ion_mobility_lower", "ion_mobility_upper", "envelope_area",
                  "covered_area", "uncovered_area", "overlap_area", "coverage_fraction")
    mz_table = {column: [] for column in mz_columns}
    im_table = {column: [] for column in im_columns}
    groups = {}
    for window, record in sorted_windows:
        groups.setdefault(_coverage_group(window), []).append((window, record))
    for group, records in sorted(groups.items(), key=lambda item: (
            item[0][0], _window_sort_key(item[0][1:]))):
        windows = [window for window, _ in records]
        intervals = {(window[0], window[1]) for window in windows}
        mz_lower = min(interval[0] for interval in intervals)
        mz_upper = max(interval[1] for interval in intervals)
        envelope = mz_upper - mz_lower
        covered, overlap, components = _interval_measure(intervals)
        values = group + (len(windows), len(intervals), mz_lower, mz_upper, envelope,
                          _finite_float(covered), _finite_float(max(0, envelope - covered)),
                          _finite_float(overlap), _finite_float(covered / envelope),
                          max(0, components - 1))
        for column, value in zip(mz_columns, values):
            mz_table[column].append(value)
        if group[0] != "ion_mobility":
            continue
        rectangles = set()
        excluded = 0
        for window, record in records:
            bounds = _mobility_bounds(window, record["reader_mobility_bounds_present"])
            if bounds is None:
                excluded += 1
            else:
                rectangles.add(window[:2] + bounds)
        if rectangles:
            mz_lower = min(rectangle[0] for rectangle in rectangles)
            mz_upper = max(rectangle[1] for rectangle in rectangles)
            im_lower = min(rectangle[2] for rectangle in rectangles)
            im_upper = max(rectangle[3] for rectangle in rectangles)
            envelope = _finite_float((mz_upper - mz_lower) * (im_upper - im_lower))
            covered, overlap = _rectangle_measure(rectangles)
            geometry = (mz_lower, mz_upper, im_lower, im_upper, envelope,
                        _finite_float(covered),
                        _finite_float(max(0, envelope - covered)) if envelope is not None else None,
                        _finite_float(overlap),
                        _finite_float(covered / envelope) if envelope is not None and envelope > 0 else None)
        else:
            geometry = (None,) * 9
        values = group + (len(windows), len(rectangles), excluded) + geometry
        for column, value in zip(im_columns, values):
            im_table[column].append(value)
    return mz_table, im_table


def compute_dia_metrics(exp):
    """Summarize DIA isolation geometry, fragment signal and MS1 survey cycles.

    Use acquisition order, without sorting or modifying the experiment. A
    multi-precursor spectrum is reported as ambiguous instead of duplicating
    its TIC across windows. The window table includes every assignable MS2
    spectrum, even if no defensible cycle boundaries are available.
    """
    windows = {}
    tic_by_rt = {}
    missing = {"Missing": 0, "Invalid": 0, "Ambiguous": 0}
    ms1_count = ms2_count = 0
    cycle_start = None
    cycle_rts = []
    cycle_windows = set()
    durations, scan_counts, window_counts = [], [], []

    for spectrum in exp:
        level = spectrum.getMSLevel()
        if level == 1:
            ms1_count += 1
            rt = _finite_float(spectrum.getRT())
            if (cycle_start is not None and rt is not None and rt > cycle_start
                    and cycle_rts and all(t is not None and cycle_start <= t <= rt
                                          for t in cycle_rts)):
                durations.append(rt - cycle_start)
                scan_counts.append(len(cycle_rts))
                window_counts.append(len(cycle_windows))
            cycle_start = rt
            cycle_rts = []
            cycle_windows = set()
        elif level == 2:
            ms2_count += 1
            rt = _finite_float(spectrum.getRT())
            cycle_rts.append(rt)
            intensities = spectrum.get_peaks()[1]
            tic = None
            if np.all(np.isfinite(intensities)):
                tic = _finite_float(np.sum(intensities, dtype=np.float64))
            if rt is not None and tic is not None:
                tic_by_rt[rt] = tic_by_rt.get(rt, 0.0) + tic
            window, reason = _isolation_window(spectrum)
            if reason is not None:
                missing[reason] += 1
                continue
            cycle_windows.add(window)
            record = windows.setdefault(window, {"count": 0, "empty": 0,
                                                  "invalid": 0, "tics": [], "rts": [],
                                                  "reader_mobility_bounds_present": False})
            record["count"] += 1
            record["rts"].append(rt)
            record["reader_mobility_bounds_present"] |= (
                spectrum.metaValueExists("ion mobility lower limit")
                or spectrum.metaValueExists("ion mobility upper limit"))
            if tic is None:
                record["invalid"] += 1
                continue
            record["tics"].append(tic)
            record["empty"] += int(tic == 0)

    metrics = {key: None for key in DIA_METRIC_METADATA}
    metrics["DIA_MS2_Count"] = ms2_count
    if len(tic_by_rt) >= 2:
        times = sorted(tic_by_rt)
        integrate = getattr(np, "trapezoid", None) or np.trapz
        metrics["DIA_MS2_TIC_Area"] = _finite_float(integrate(
            [tic_by_rt[rt] for rt in times], x=times))
    metrics.update({"DIA_IsolationMetadata_" + k + "Count": v for k, v in missing.items()})
    metrics["DIA_IsolationWindow_Count"] = len(windows)
    columns = ("lower_mz", "upper_mz", "width_mz", "drift_time", "drift_time_unit",
               "ion_mobility_lower", "ion_mobility_upper", "precursor_drift_time",
               "precursor_drift_lower_offset", "precursor_drift_upper_offset", "scan_count",
               "empty_scan_count", "invalid_tic_scan_count", "tic_sum", "tic_median")
    table = {key: [] for key in columns}
    # Numeric sorting with None first is stable even for mixed mobility metadata.
    sorted_windows = sorted(windows.items(), key=lambda item: _window_sort_key(item[0]))
    for window, record in sorted_windows:
        lower, upper, drift, unit, im_lower, im_upper, prec_drift, prec_lower, prec_upper = window
        tics = record["tics"]
        values = (lower, upper, round(upper - lower, 6), drift, unit, im_lower, im_upper,
                  prec_drift, prec_lower, prec_upper,
                  record["count"], record["empty"], record["invalid"],
                  sum(tics) if tics else None, median(tics) if tics else None)
        for key, value in zip(columns, values):
            table[key].append(value)
    metrics["DIA_IsolationWindow_Summary"] = table
    metrics["DIA_IsolationWindow_RevisitSummary"] = _revisit_summary(sorted_windows)
    (metrics["DIA_IsolationWindow_MzCoverage"],
     metrics["DIA_IsolationWindow_MzIMCoverage"]) = _coverage_summary(sorted_windows)
    if windows:
        metrics["DIA_IsolationWindow_MzRange"] = [min(table["lower_mz"]), max(table["upper_mz"])]
        metrics["DIA_IsolationWindow_WidthRange"] = [min(table["width_mz"]), max(table["width_mz"])]
    metrics["DIA_MS1BoundedCycle_Count"] = len(durations)
    if durations:
        metrics["DIA_MS1BoundedCycle_DurationMedian"] = median(durations)
        metrics["DIA_MS1BoundedCycle_MS2CountRange"] = [min(scan_counts), max(scan_counts)]
        metrics["DIA_MS1BoundedCycle_MS2CountMedian"] = median(scan_counts)
        metrics["DIA_MS1BoundedCycle_WindowCountRange"] = [min(window_counts), max(window_counts)]
        metrics["DIA_MS1BoundedCycle_WindowCountMedian"] = median(window_counts)
        if len(durations) > 1:
            metrics["DIA_MS1BoundedCycle_DurationCV"] = pstdev(durations) / mean(durations)
    for level, loaded_count in [(1, ms1_count), (2, ms2_count)]:
        key = "rawqc_bruker_expected_ms{}_spectra".format(level)
        if exp.metaValueExists(key):
            expected = _finite_float(exp.getMetaValue(key))
            if expected is not None and expected >= 0 and expected.is_integer():
                metrics["DIA_Bruker_ExpectedMS{}SpectrumCount".format(level)] = int(expected)
                if expected >= loaded_count:
                    metrics["DIA_Bruker_ReaderOmittedMS{}SpectrumCount".format(level)] = int(expected) - loaded_count
    return metrics
