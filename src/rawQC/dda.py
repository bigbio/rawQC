"""Precursor and charge metrics for data-dependent acquisition (DDA).

These metrics summarize individual selected precursor ions. Their historical
imports from ``rawQC.calculate_metrics`` remain available for existing callers.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyopenms as oms

from .spectrum_utils import _filter_by_mslevel, _nanmedian


def _precursor_values(specs: List[oms.MSSpectrum]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract precursor m/z, intensity, and charge from MS2/MSn spectra.

    Args:
        specs: list of MSSpectrum objects

    Returns:
        tuple: (mzs, intensities, charges) as numpy arrays
               NaN values indicate missing precursor information
    """
    mzs, intens, charges = [], [], []
    for sp in specs:
        precs = sp.getPrecursors()
        if not precs:
            mzs.append(np.nan); intens.append(np.nan); charges.append(np.nan)
            continue
        p = precs[0]
        mzs.append(float(p.getMZ()) if p.getMZ() else np.nan)
        # A present precursor keeps its recorded intensity, including a legitimate
        # 0.0 (truthiness would silently turn 0.0 into NaN and drop it). Only an
        # absent precursor (handled above) yields NaN.
        intens.append(float(p.getIntensity()))
        charges.append(int(p.getCharge()) if p.getCharge() else np.nan)
    return np.array(mzs, dtype=float), np.array(intens, dtype=float), np.array(charges, dtype=float)


def precursor_intensities(specs: List[oms.MSSpectrum],
                          fallback_to_ms2_tic: bool = True) -> Tuple[np.ndarray, int, int]:
    """
    Extract MSn precursor intensities with an explicit zero/missing policy.

    QuaMeter detects a zero or unrecorded precursor intensity and falls back to
    the corresponding MSn (MS2) total ion current, so a recorded 0.0 is not
    silently dropped. rawQC follows that convention by default:

      * spectrum with no precursor      -> NaN (counted as ``n_missing``)
      * precursor intensity <= 0        -> the spectrum's TIC (counted as
                                           ``n_fallback``) when
                                           ``fallback_to_ms2_tic`` is True,
                                           otherwise the value 0.0 is preserved
      * precursor intensity  > 0        -> that intensity

    Args:
        specs: list of MSSpectrum objects (typically MS2)
        fallback_to_ms2_tic: apply the QuaMeter MS2-TIC fallback (default True)

    Returns:
        tuple: (intensities, n_fallback, n_missing)
    """
    vals: List[float] = []
    n_fallback = 0
    n_missing = 0
    for sp in specs:
        precs = sp.getPrecursors()
        if not precs:
            vals.append(np.nan)
            n_missing += 1
            continue
        inten = float(precs[0].getIntensity())
        if inten <= 0.0:
            if fallback_to_ms2_tic:
                inten = float(sp.calculateTIC())
                n_fallback += 1
            else:
                inten = 0.0
        vals.append(inten)
    return np.array(vals, dtype=float), n_fallback, n_missing


def mz_acquisition_range(exp: oms.MSExperiment, ms_level: int = 2) -> Tuple[float, float]:
    """
    m/z acquisition range (MS:4000069).

    MS:4000069:
    "Upper and lower limit of m/z precursor values at which MSn spectra are
    recorded." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The precursor m/z values of the peaks within the spectra are obtained,
    (3) The minimum and maximum precursor m/z values are obtained and returned.

    Details:
        MS:4000069
        is_a: MS:4000004 ! n-tuple
        relationship: has_metric_category MS:4000009 ! ID free metric
        relationship: has_metric_category MS:4000019 ! MS metric
        relationship: has_units MS:1000040 ! m/z
        relationship: has_value_concept STATO:0000035 ! range

    Note:
        This reads precursor m/z values, so it is meaningful only for MSn
        (ms_level >= 2). MS1 spectra have no precursor and would yield (NaN, NaN);
        rawQC therefore does not emit an MS1 precursor m/z range.

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 2)

    Returns:
        tuple: (min_mz, max_mz) as floats

    Example:
        >>> mz_min, mz_max = mz_acquisition_range(exp, ms_level=2)
    """
    specs = _filter_by_mslevel(exp, ms_level)
    mzs, _, _ = _precursor_values(specs)
    mzs = mzs[~np.isnan(mzs)]
    if mzs.size == 0: return (np.nan, np.nan)
    return (float(np.min(mzs)), float(np.max(mzs)))


def precursor_intensity_stats(exp: oms.MSExperiment, ms_level: int = 2) -> Dict[str, float]:
    """
    MS2 precursor intensity distribution (MS:4000116).

    MS:4000116:
    "From the distribution of MS2 precursor intensities, the quantiles. E.g. a
    value triplet represents the quartiles Q1, Q2, Q3. The intensity
    distribution of the precursors informs about the dynamic range of the
    acquisition." [PSI:MS]

    Also calculates mean (MS:4000117) and standard deviation (MS:4000118).

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The intensity of the precursor ions within the spectra are obtained,
    (3) The 25%, 50%, and 75% quantile, mean, and standard deviation of the
        precursor intensity values are obtained (NA values are removed) and returned.

    Details:
        MS:4000116
        is_a: MS:4000004 ! n-tuple
        relationship: has_metric_category MS:4000009 ! ID free metric
        relationship: has_metric_category MS:4000022 ! MS2 metric
        relationship: has_value_concept STATO:0000291 ! quantile
        relationship: has_units MS:1000043 ! intensity unit

        MS:4000117 (mean)
        relationship: has_value_concept STATO:0000401 ! sample mean

        MS:4000118 (sigma/sd)
        relationship: has_value_concept STATO:0000237 ! standard deviation

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 2)

    Returns:
        dict: Precursor intensity statistics (Q1, Q2, Q3, Mean, Sd)

    Example:
        >>> stats = precursor_intensity_stats(exp, ms_level=2)
        >>> print(stats['PrecursorIntensity_Q2'])  # median
    """
    specs = _filter_by_mslevel(exp, ms_level)
    preI, _, _ = precursor_intensities(specs)
    preI = preI[~np.isnan(preI)]
    if preI.size == 0:
        return {
            "PrecursorIntensity_Q1": np.nan,
            "PrecursorIntensity_Q2": np.nan,
            "PrecursorIntensity_Q3": np.nan,
            "PrecursorIntensity_Mean": np.nan,
            "PrecursorIntensity_Sd": np.nan,
        }
    q1, q2, q3 = np.quantile(preI, [0.25, 0.50, 0.75])
    return {
        "PrecursorIntensity_Q1": float(q1),
        "PrecursorIntensity_Q2": float(q2),
        "PrecursorIntensity_Q3": float(q3),
        "PrecursorIntensity_Mean": float(np.mean(preI)),
        "PrecursorIntensity_Sd": float(np.std(preI, ddof=1)) if preI.size > 1 else np.nan,
    }


def median_precursor_mz(exp: oms.MSExperiment, ms_level: int = 2,
                        accepted_native_ids: Optional[set] = None) -> float:
    """
    ID-free proxy for the median precursor m/z of identified data points (MS:4000152).

    MS:4000152:
    "Median m/z value for MS2 precursors of all quantification data points after
    user-defined acceptance criteria are applied. These data points may be for
    example XIC profiles, isotopic pattern areas, or reporter ions." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The precursor m/z values are obtained,
    (3) The median value is returned (NAs are removed).

    Details:
        MS:4000152
        is_a: MS:4000003 ! single value
        is_a: MS:4000008 ! ID based
        relationship: has_metric_category MS:4000022 ! MS2 metric
        relationship: has_units MS:1000040 ! m/z

    Note:
        MS:4000152 is an ID-based term ("all quantification data points after
        user-defined acceptance criteria are applied"). Without identifications
        rawQC computes the median precursor m/z over ALL MS2 precursors, which is
        an ID-free proxy and is emitted WITHOUT the MS:4000152 accession. Pass
        ``accepted_native_ids`` (accepted spectrum native IDs) to restrict the
        computation to identified spectra and reproduce the ID-based definition.

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 2)
        accepted_native_ids: optional set of accepted spectrum native IDs

    Returns:
        float: Median precursor m/z

    Example:
        >>> median_mz = median_precursor_mz(exp, ms_level=2)
    """
    specs = _filter_by_mslevel(exp, ms_level)
    if accepted_native_ids is not None:
        wanted = set(accepted_native_ids)
        specs = [s for s in specs if s.getNativeID() in wanted]
    preMz, _, _ = _precursor_values(specs)
    return _nanmedian(preMz)


def extent_identified_precursor_intensity(exp: oms.MSExperiment, ms_level: int = 2,
                                          accepted_native_ids: Optional[set] = None) -> float:
    """
    ID-free proxy for the extent of identified MS2 precursor intensity (MS:4000157).

    MS:4000157:
    "Ratio of 95th over 5th percentile of MS2 precursor intensity for all
    quantification data points after user-defined acceptance criteria are
    applied. Can be used to approximate the dynamic range of signal." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The intensities of the precursor ions are obtained,
    (3) The 5% and 95% quantile of these intensities are obtained
        (NA values are removed),
    (4) The ratio between the 95% and the 5% intensity quantile is calculated
        and returned.

    Details:
        MS:4000157
        synonym: "MS1-3A" RELATED [PMID:19837981]
        is_a: MS:4000001 ! QC metric
        is_a: MS:4000003 ! single value
        is_a: MS:4000008 ! ID based
        relationship: has_metric_category MS:4000022 ! MS2 metric

    Note:
        MS:4000157 is an ID-based term (MS1-3A) whose reference implementation
        (SMAQC) is based on identified-peptide XIC peak-apex intensities. Without
        identifications rawQC computes the 95/5 ratio over ALL MS2 precursor
        intensities, which is an ID-free proxy and is emitted WITHOUT the
        MS:4000157 accession. Pass ``accepted_native_ids`` to restrict to
        identified spectra. Precursor intensity values that are NA are removed.

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 2)
        accepted_native_ids: optional set of accepted spectrum native IDs

    Returns:
        float: Ratio of 95th/5th percentile intensities

    Example:
        >>> extent = extent_identified_precursor_intensity(exp, ms_level=2)
    """
    specs = _filter_by_mslevel(exp, ms_level)
    # Optional accepted-ID filter (#40) applied before the QuaMeter MS2-TIC
    # fallback intensity extraction (#39).
    if accepted_native_ids is not None:
        wanted = set(accepted_native_ids)
        specs = [s for s in specs if s.getNativeID() in wanted]
    preI, _, _ = precursor_intensities(specs)
    preI = preI[~np.isnan(preI)]
    if preI.size == 0: return np.nan
    q5, q95 = np.quantile(preI, [0.05, 0.95])
    if q5 == 0: return np.nan
    return float(q95 / q5)


def charge_metrics(exp: oms.MSExperiment, ms_level: int = 2) -> Dict[str, float]:
    """
    Charge-related metrics for MS2 precursors.

    Calculates:
    - Min/Max charge states
    - Ratio of 3+ over 2+ (MS:4000169/MS:4000170)
    - Ratio of 4+ over 2+ (MS:4000171/MS:4000172)
    - Mean MS2 precursor charge (MS:4000173/MS:4000174)
    - Median MS2 precursor charge (MS:4000175/MS:4000176)
    - MS2 precursor charge state fractions (MS:4000063)

    MS:4000169/MS:4000170:
    "The ratio of 3+ over 2+ MS2 precursor charge count of all/identified spectra.
    Higher ratios of 3+/2+ MS2 precursor charge count may preferentially favor
    longer e.g. peptides." [PSI:MS]

    MS:4000171/MS:4000172:
    "The ratio of 4+ over 2+ MS2 precursor charge count of all/identified spectra."

    MS:4000173/MS:4000174:
    "Mean MS2 precursor charge in all/identified spectra" [PSI:MS]

    MS:4000175/MS:4000176:
    "Median MS2 precursor charge in all/identified spectra" [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The precursor charge is obtained,
    (3) Charge ratios, mean, and median are calculated.

    Details:
        MS:4000169/MS:4000171
        synonym: "IS-3B"/"IS-3C" RELATED [PMID:19837981]
        is_a: MS:4000003 ! single value
        is_a: MS:4000009 ! ID free metric
        relationship: has_metric_category MS:4000020 ! ion source metric
        relationship: has_metric_category MS:4000022 ! MS2 metric

    Note:
        Returns NaN if either charge state is missing (matching R implementation).
        For 3over2: NaN if either charge 2 or 3 is absent.
        For 4over2: NaN if either charge 2 or 4 is absent.

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 2)

    Returns:
        dict: Charge metrics (ChargeRatio_3over2, ChargeRatio_4over2, ChargeMean, ChargeMedian)

    Example:
        >>> metrics = charge_metrics(exp, ms_level=2)
        >>> print(metrics['ChargeMean'])
    """
    specs = _filter_by_mslevel(exp, ms_level)
    n_ms2 = len(specs)
    _, _, charges = _precursor_values(specs)
    # Known charge states (>=1); unknown = missing, zero, or non-physical
    # negative charge, kept as its own bin. QuaMeter stores unknown charge as 0
    # and divides every bin by ALL MS2 scans, so the fractions have the reference
    # denominator and sum to 1.0.
    c = charges[~np.isnan(charges)].astype(int)
    out: Dict[str, Any] = {}

    # Charge-state fraction table (MS:4000063), denominator = all MS2 scans.
    labels = ["1", "2", "3", "4", "5", ">=6", "unknown"]
    if n_ms2 > 0:
        counts_by_bin = [
            int(np.sum(c == 1)),
            int(np.sum(c == 2)),
            int(np.sum(c == 3)),
            int(np.sum(c == 4)),
            int(np.sum(c == 5)),
            int(np.sum(c >= 6)),
            # Unknown = every MS2 scan without a valid (>=1) charge: missing,
            # zero, AND any non-physical negative charge. Basing this on the
            # count of valid charges (not c.size) keeps the fractions summing to
            # 1.0 even if a negative charge sneaks through.
            int(n_ms2 - int(np.sum(c >= 1))),
        ]
        fractions = [float(n / n_ms2) for n in counts_by_bin]
    else:
        counts_by_bin = [0, 0, 0, 0, 0, 0, 0]
        fractions = [np.nan] * 7
    out["MS2_PrecursorCharge_Fractions"] = {
        "charge_state": list(labels),
        "count": counts_by_bin,
        "fraction": fractions,
    }

    if c.size == 0:
        out["ChargeMin"] = np.nan
        out["ChargeMax"] = np.nan
        out["ChargeRatio_3over2"] = np.nan
        out["ChargeRatio_4over2"] = np.nan
        out["ChargeMean"] = np.nan
        out["ChargeMedian"] = np.nan
        return out

    # Min and Max charge states (over known charges)
    out["ChargeMin"] = int(np.min(c))
    out["ChargeMax"] = int(np.max(c))

    vals, counts = np.unique(c, return_counts=True)
    table = dict(zip(vals.tolist(), counts.tolist()))
    # Match R implementation: return NaN if either charge state is missing
    # R: if (all(c(2, 3) %in% names(chargeTable)))
    if 2 in table and 3 in table:
        out["ChargeRatio_3over2"] = float(table[3] / table[2])
    else:
        out["ChargeRatio_3over2"] = np.nan

    if 2 in table and 4 in table:
        out["ChargeRatio_4over2"] = float(table[4] / table[2])
    else:
        out["ChargeRatio_4over2"] = np.nan

    out["ChargeMean"] = float(np.mean(c))
    out["ChargeMedian"] = float(np.median(c))
    return out


def compute_dda_metrics(exp: oms.MSExperiment) -> Dict[str, Any]:
    """Compute the existing individual-precursor metrics for a DDA run."""
    computed: Dict[str, Any] = {
        "MzRange_MS2": [float(value) for value in mz_acquisition_range(exp, 2)],
        "PrecursorMz_MS2_Median": median_precursor_mz(exp, 2),
        "ExtentPrecursorIntensity_95over5_MS2": extent_identified_precursor_intensity(exp, 2),
    }
    computed.update(precursor_intensity_stats(exp, 2))
    _, n_fallback, _ = precursor_intensities(_filter_by_mslevel(exp, 2))
    computed["PrecursorIntensity_FallbackCount"] = int(n_fallback)
    computed.update(charge_metrics(exp, 2))
    return computed
