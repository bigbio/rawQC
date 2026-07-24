
from urllib.request import urlretrieve
import numpy as np
import json
from datetime import datetime
import pyopenms as oms
from mzqc import MZQCFile as qc
from typing import List, Tuple, Dict, Any, Optional, Union
import os
import click
import textwrap

# -------------------------------------------------------------------------
# Demo data (replace with your own mzML files)
# -------------------------------------------------------------------------
# Dictionary mapping URLs to local filenames for example mzML files
# You can replace these with your own URLs or local file paths

# mix a proteomic dataset with some metabolomics data for fun
MZML_FILES = {
    "https://raw.githubusercontent.com/OpenMS/OpenMS/refs/heads/develop/share/OpenMS/examples/BSA/BSA1.mzML": "BSA1.mzML",
    "https://raw.githubusercontent.com/OpenMS/OpenMS/refs/heads/develop/share/OpenMS/examples/BSA/BSA2.mzML": "BSA2.mzML",
    "https://raw.githubusercontent.com/OpenMS/OpenMS/refs/heads/develop/share/OpenMS/examples/BSA/BSA3.mzML": "BSA3.mzML",

    "https://abibuilder.cs.uni-tuebingen.de/archive/openms/Tutorials/Example_Data/Metabolomics/datasets/2012_02_03_PStd_050_1.mzML": "2012_02_03_PStd_050_1.mzML",
    "https://abibuilder.cs.uni-tuebingen.de/archive/openms/Tutorials/Example_Data/Metabolomics/datasets/2012_02_03_PStd_050_2.mzML": "2012_02_03_PStd_050_2.mzML",
    "https://abibuilder.cs.uni-tuebingen.de/archive/openms/Tutorials/Example_Data/Metabolomics/datasets/2012_02_03_PStd_050_3.mzML": "2012_02_03_PStd_050_3.mzML",

    "https://abibuilder.cs.uni-tuebingen.de/archive/openms/Tutorials/Example_Data/Metabolomics/datasets/2012_02_03_PStd_10_1.mzML": "2012_02_03_PStd_10_1.mzML",
    "https://abibuilder.cs.uni-tuebingen.de/archive/openms/Tutorials/Example_Data/Metabolomics/datasets/2012_02_03_PStd_10_2.mzML": "2012_02_03_PStd_10_2.mzML",
    "https://abibuilder.cs.uni-tuebingen.de/archive/openms/Tutorials/Example_Data/Metabolomics/datasets/2012_02_03_PStd_10_3.mzML": "2012_02_03_PStd_10_3.mzML",
}

# -------------------------------------------------------------------------
# PSI:MS / mzQC metric metadata (accession + description)
# -------------------------------------------------------------------------
METRIC_METADATA = {
    # Chromatography duration
    "ChromatographyDuration": {
        "accession": "MS:4000053",
        "description": "The retention time duration of the chromatography in seconds."
    },

    # Number of spectra
    "NumberOfSpectra_MS1": {
        "accession": "MS:4000059",
        "description": "The number of MS1 events in the run."
    },
    "NumberOfSpectra_MS2": {
        "accession": "MS:4000060",
        "description": "The number of MS2 events in the run."
    },

    # Peak density quantiles
    "PeakDensity_MS1_Q1": {
        "accession": "MS:4000061",
        "description": "First quantile of MS1 peak density (peaks per scan)."
    },
    "PeakDensity_MS1_Q2": {
        "accession": "MS:4000061",
        "description": "Second quantile of MS1 peak density (peaks per scan)."
    },
    "PeakDensity_MS1_Q3": {
        "accession": "MS:4000061",
        "description": "Third quantile of MS1 peak density (peaks per scan)."
    },
    "PeakDensity_MS2_Q1": {
        "accession": "MS:4000062",
        "description": "First quantile of MS2 peak density (peaks per scan)."
    },
    "PeakDensity_MS2_Q2": {
        "accession": "MS:4000062",
        "description": "Second quantile of MS2 peak density (peaks per scan)."
    },
    "PeakDensity_MS2_Q3": {
        "accession": "MS:4000062",
        "description": "Third quantile of MS2 peak density (peaks per scan)."
    },

    # Total peaks
    "NumberOfSpectralPeaks": {
        "accession": None,
        "description": "Total number of peaks across all spectra in the run."
    },
    "NumberOfChromatogramDataPoints": {
        "accession": None,
        "description": "Total number of data points across all chromatograms (stored array lengths, not resolved chromatographic peaks)."
    },

    # FAIMS
    "FAIMS_CV_Count": {
        "accession": None,
        "description": "Number of distinct FAIMS compensation voltages used."
    },
    "FAIMS_CV_Values": {
        "accession": None,
        "description": "Sorted distinct FAIMS compensation voltages used, in volts (V), as an n-tuple."
    },
    "FAIMS_CV_Range": {
        "accession": None,
        "description": "Range [min, max] of FAIMS compensation voltages, in volts (V)."
    },

    # Empty scans
    "EmptyScans_MS1": {
        "accession": "MS:4000099",
        "description": "Number of MS1 scans where the peaks' intensity sums to 0 (i.e. no peaks or only 0-intensity peaks)."
    },
    "EmptyScans_MS2": {
        "accession": "MS:4000100",
        "description": "Number of MS2 scans where the peaks' intensity sums to 0 (i.e. no peaks or only 0-intensity peaks)."
    },

    # m/z and RT acquisition ranges (each a two-value [min, max] n-tuple)
    # MS:4000069 = precursor m/z acquisition range (MSn only);
    # MS:4000070 = retention-time acquisition range.
    "MzRange_MS2": {
        "accession": "MS:4000069",
        "description": "Lower and upper limit of precursor m/z values at which MS2 spectra are recorded, as [min, max]."
    },
    "RtRange_MS1": {
        "accession": "MS:4000070",
        "description": "Lower and upper limit of retention time (seconds) at which MS1 spectra are recorded, as [min, max]."
    },
    "RtRange_MS2": {
        "accession": "MS:4000070",
        "description": "Lower and upper limit of retention time (seconds) at which MS2 spectra are recorded, as [min, max]."
    },

    # Fastest acquisition frequency
    "FastestFrequency_MS1": {
        "accession": "MS:4000065",
        "description": "Fastest observed frequency of MS1 spectrum acquisition (Hz)."
    },
    "FastestFrequency_MS2": {
        "accession": "MS:4000066",
        "description": "Fastest observed frequency of MS2 spectrum acquisition (Hz)."
    },

    # RT over MS quantiles (four interval fractions as one n-tuple)
    "RT_MS1_Quantiles": {
        "accession": "MS:4000184",
        "description": "The four RT interval fractions of MS1 events (between the 25/50/75th scan-time percentiles), normalized by the MS1 acquisition duration; sums to 1.0."
    },
    "RT_MS2_Quantiles": {
        "accession": "MS:4000185",
        "description": "The four RT interval fractions of MS2 events (between the 25/50/75th scan-time percentiles), normalized by the MS2 acquisition duration; sums to 1.0."
    },

    # TIC quartile ratios
    "TIC_MS1_Change_Q2": {
        "accession": "MS:4000186",
        "description": "Log ratio of MS1 TIC-change Q2 to Q1. TIC changes are differences between successive MS1 TIC values."
    },
    "TIC_MS1_Change_Q3": {
        "accession": "MS:4000186",
        "description": "Log ratio of MS1 TIC-change Q3 to Q2. TIC changes are differences between successive MS1 TIC values."
    },
    "TIC_MS1_Change_Q4": {
        "accession": "MS:4000186",
        "description": "Log ratio of MS1 TIC-change Q4 to Q3. TIC changes are differences between successive MS1 TIC values."
    },
    "TIC_MS1_Ratio_Q2": {
        "accession": "MS:4000187",
        "description": "Log ratio of MS1 TIC Q2 to Q1."
    },
    "TIC_MS1_Ratio_Q3": {
        "accession": "MS:4000187",
        "description": "Log ratio of MS1 TIC Q3 to Q2."
    },
    "TIC_MS1_Ratio_Q4": {
        "accession": "MS:4000187",
        "description": "Log ratio of MS1 TIC Q4 to Q3."
    },

    # TIC accumulation RT quantiles (four interval fractions as one n-tuple)
    "RT_TIC_Quantiles": {
        "accession": "MS:4000183",
        "description": "The four RT interval fractions between the retention times at which the cumulative MS1 TIC reaches 25/50/75% of the total, normalized by the MS1 acquisition duration; sums to 1.0."
    },

    # Charge metrics
    "ChargeMean": {
        "accession": "MS:4000173",
        "description": "Mean MS2 precursor charge in all spectra."
    },
    "ChargeMedian": {
        "accession": "MS:4000175",
        "description": "Median MS2 precursor charge in all spectra."
    },
    "ChargeMin": {
        "accession": None,
        "description": "Minimum MS2 precursor charge state observed."
    },
    "ChargeMax": {
        "accession": None,
        "description": "Maximum MS2 precursor charge state observed."
    },
    "ChargeRatio_3over2": {
        "accession": None,
        "description": "The ratio of 3+ over 2+ MS2 precursor charge count. Higher ratios may preferentially favor longer peptides."
    },
    "ChargeRatio_4over2": {
        "accession": None,
        "description": "The ratio of 4+ over 2+ MS2 precursor charge count."
    },
    # MS2 precursor charge fractions (one table, denominator = all MS2 scans)
    "MS2_PrecursorCharge_Fractions": {
        "accession": "MS:4000063",
        "description": "Fraction of MS2 precursors per charge state (1, 2, 3, 4, 5, >=6, and unknown/missing) as a table with charge_state, count and fraction columns; fractions are over all MS2 scans and sum to 1.0.",
    },

    # Custom metrics (non-PSI:MS)
    "TIC_MS1_CV": {
        "accession": None,
        "description": "Coefficient of variation of MS1 total ion current. Indicates stability of MS1 signal."
    },
    "TIC_MS2_CV": {
        "accession": None,
        "description": "Coefficient of variation of MS2 total ion current. Indicates stability of MS2 signal."
    },
    "BasePeak_MS1_Mean": {
        "accession": None,
        "description": "Mean of base peak intensities across all MS1 spectra."
    },
    "BasePeak_MS2_Mean": {
        "accession": None,
        "description": "Mean of base peak intensities across all MS2 spectra."
    },
    "BasePeak_All_Max": {
        "accession": "MS:4000202",
        "description": "Maximum base peak intensity observed across all spectra.",
    },
    "ScanRate_MS1": {
        "accession": None,
        "description": "MS1 scan rate (scans per minute) over the MS1 acquisition span."
    },
    "ScanRate_MS2": {
        "accession": None,
        "description": "MS2 scan rate (scans per minute) over the MS2 acquisition span."
    },
    "MS1_to_MS2_Ratio": {
        "accession": None,
        "description": "Ratio of MS1 to MS2 spectra counts."
    },

    # Signal jumps/falls
    "TIC_MS1_SignalJump10x_Count": {
        "accession": "MS:4000097",
        "description": "Number of times MS1 TIC increased more than 10-fold between adjacent scans. High counts may indicate ESI stability issues."
    },
    "TIC_MS1_SignalFall10x_Count": {
        "accession": "MS:4000098",
        "description": "Number of times MS1 TIC decreased more than 10-fold between adjacent scans. High counts may indicate ESI stability issues."
    },

    # Precursor intensity stats
    "PrecursorIntensity_Q1": {
        "accession": None,
        "description": "25th percentile (Q1) of MS2 precursor intensities."
    },
    "PrecursorIntensity_Q2": {
        "accession": None,
        "description": "50th percentile (Q2/median) of MS2 precursor intensities."
    },
    "PrecursorIntensity_Q3": {
        "accession": None,
        "description": "75th percentile (Q3) of MS2 precursor intensities."
    },
    "PrecursorIntensity_Mean": {
        "accession": None,
        "description": "Mean of MS2 precursor intensities."
    },
    "PrecursorIntensity_Sd": {
        "accession": None,
        "description": "Standard deviation of MS2 precursor intensities."
    },
    "PrecursorIntensity_FallbackCount": {
        "accession": None,
        "description": "Number of MS2 precursors whose zero/unrecorded intensity was replaced by the spectrum MS2 TIC (QuaMeter fallback)."
    },

    # Median precursor m/z
    "PrecursorMz_MS2_Median": {
        "accession": None,
        "description": "Median m/z value for MS2 precursors."
    },

    # RT IQR metrics
    "RT_MS1_IQR": {
        "accession": None,
        "description": "Interquartile range of retention times for MS1 spectra (seconds). Longer times indicate better chromatographic separation."
    },
    "RT_MS1_IQRRate": {
        "accession": None,
        "description": "Rate of MS1 spectra per second in the RT interquartile range. Higher rates indicate efficient sampling."
    },

    # Area under TIC
    "TIC_MS1_Area_RTQuantiles": {
        "accession": "MS:4000156",
        "description": "Area under the MS1 TIC for the four retention-time quartiles (0-25%, 25-50%, 50-75%, 75-100%) as one n-tuple."
    },
    "TIC_MS1_Area": {
        "accession": "MS:4000029",
        "description": "Time integral of the MS1 TIC over retention time (area under the total ion chromatogram; intensity x second)."
    },

    # Extent of precursor intensity
    "ExtentPrecursorIntensity_95over5_MS2": {
        "accession": None,
        "description": "Ratio of 95th to 5th percentile of MS2 precursor intensity. Approximates dynamic range of signal."
    },

    # Median TIC in RT ranges
    "MedianTIC_in_RT_MS1_IQR": {
        "accession": None,
        "description": "Median MS1 TIC in the RT range between Q1 and Q3 of retention times."
    },
    "TIC_MS1_MedianInHalfRange": {
        "accession": None,
        "description": "Median MS1 TIC in the shortest RT range containing half of all spectra."
    },

    # MS levels
    "NumberOfMSLevels": {
        "accession": None,
        "description": "The number of distinct MS levels present in the run (e.g., MS1, MS2, MS3)."
    },

    # Polarity statistics (custom counts). MS:1000129/MS:1000130 are scan-polarity
    # CV terms, not QC metrics, so they are NOT used as accessions for counts.
    "Polarity_MS1_positive": {
        "accession": None,
        "description": "Number of MS1 spectra acquired in positive polarity mode (scan polarity MS:1000130)."
    },
    "Polarity_MS1_negative": {
        "accession": None,
        "description": "Number of MS1 spectra acquired in negative polarity mode (scan polarity MS:1000129)."
    },
    "Polarity_MS1_unknown": {
        "accession": None,
        "description": "Number of MS1 spectra with unknown polarity."
    },
    "Polarity_MS2_positive": {
        "accession": None,
        "description": "Number of MS2 spectra acquired in positive polarity mode (scan polarity MS:1000130)."
    },
    "Polarity_MS2_negative": {
        "accession": None,
        "description": "Number of MS2 spectra acquired in negative polarity mode (scan polarity MS:1000129)."
    },
    "Polarity_MS2_unknown": {
        "accession": None,
        "description": "Number of MS2 spectra with unknown polarity."
    },

    # MS1 cycle time
    "AvgCycleTime_MS1": {
        "accession": None,
        "description": "Average time between consecutive MS1 scans (seconds). Indicates acquisition duty cycle and sampling rate."
    },

    # Chromatogram statistics
    "NumberOfChromatograms": {
        "accession": "MS:4000071",
        "description": "Total number of chromatograms in the mzML file."
    },
    "Chromatograms_TIC": {
        "accession": None,
        "description": "Number of Total Ion Current (TIC) chromatograms."
    },
    "Chromatograms_BPC": {
        "accession": None,
        "description": "Number of Base Peak Chromatograms (BPC)."
    },
    "Chromatograms_SRM": {
        "accession": None,
        "description": "Number of Selected Reaction Monitoring (SRM) chromatograms."
    },
    "Chromatograms_SIM": {
        "accession": None,
        "description": "Number of Selected Ion Monitoring (SIM) chromatograms."
    },
    "Chromatograms_XIC": {
        "accession": None,
        "description": "Number of extracted-ion / mass chromatograms (XIC)."
    },
    "Chromatograms_SIC": {
        "accession": None,
        "description": "Number of Selected Ion Current (SIC) chromatograms."
    },
    "Chromatograms_Unknown": {
        "accession": None,
        "description": "Number of chromatograms with unknown type."
    },
    "Chromatograms_RT_Min": {
        "accession": None,
        "description": "Minimum retention time covered by chromatograms (seconds)."
    },
    "Chromatograms_RT_Max": {
        "accession": None,
        "description": "Maximum retention time covered by chromatograms (seconds)."
    },

    # Peak type statistics (aggregated over all spectra of the level)
    "MS1_PeakType_Annotated": {
        "accession": None,
        "description": "Metadata peak type for MS1 aggregated over the run (centroid, profile, mixed, or unknown)."
    },
    "MS1_PeakType_Annotated_ProfileFraction": {
        "accession": None,
        "description": "Fraction of type-resolved MS1 spectra annotated as profile."
    },
    "MS1_PeakType_Estimated": {
        "accession": None,
        "description": "Peak type estimated from peak spacing for MS1 aggregated over the run (centroid, profile, mixed, or unknown)."
    },
    "MS1_PeakType_Estimated_ProfileFraction": {
        "accession": None,
        "description": "Fraction of estimatable MS1 spectra estimated as profile."
    },
    "MS2_PeakType_Annotated": {
        "accession": None,
        "description": "Metadata peak type for MS2 aggregated over the run (centroid, profile, mixed, or unknown)."
    },
    "MS2_PeakType_Annotated_ProfileFraction": {
        "accession": None,
        "description": "Fraction of type-resolved MS2 spectra annotated as profile."
    },
    "MS2_PeakType_Estimated": {
        "accession": None,
        "description": "Peak type estimated from peak spacing for MS2 aggregated over the run (centroid, profile, mixed, or unknown)."
    },
    "MS2_PeakType_Estimated_ProfileFraction": {
        "accession": None,
        "description": "Fraction of estimatable MS2 spectra estimated as profile."
    },

    # Mass analyzer information (one table, arbitrary number of analyzers)
    "MassAnalyzers": {
        "accession": None,
        "description": "Mass analyzers of the instrument as a table with index, type (e.g. ORBITRAP, TOF, IT, QUADRUPOLE), and resolution columns."
    },

    # Activation methods (one table, arbitrary number of methods/levels)
    "ActivationMethods": {
        "accession": None,
        "description": "Precursor activation methods observed as a table with ms_level, method (e.g. HCD, CID, ETD), and count columns."
    },


    "TIC_MS2_Area": {
        "accession": "MS:4000030",
        "description": "Time integral of the MS2 TIC over retention time (area under the total ion chromatogram; intensity x second)."
    },
    # NOTE: MS_Run_Duration (MS:4000067) was registered here but never computed
    # (a dead entry with a null description); removed in issue #44.
}

# Derived metadata lookups for convenience and validation

METRIC_ACCESSIONS = {k: v["accession"] for k, v in METRIC_METADATA.items() if v["accession"] is not None}
METRIC_DESCRIPTIONS = {k: v["description"] for k, v in METRIC_METADATA.items() if v["description"] is not None}
MISSING_METRIC_ACCESSIONS = [k for k, v in METRIC_METADATA.items() if v["accession"] is None]
MISSING_METRIC_DESCRIPTIONS = [k for k, v in METRIC_METADATA.items() if v["description"] is None]

# Authoritative presentation order for the static metrics. Dynamic metric
# families (chromatogram types, activation methods, analyzers, peak types, FAIMS)
# are appended in computation order after these. Kept as a single source of truth
# so ordering and metadata cannot drift apart (see validate_metric_registry).
METRIC_ORDER = [
    "NumberOfMSLevels",
    "NumberOfSpectra_MS1",
    "NumberOfSpectra_MS2",
    "MS1_to_MS2_Ratio",
    "ChromatographyDuration",
    "NumberOfChromatograms",
    "NumberOfChromatogramDataPoints",
    "NumberOfSpectralPeaks",
    "Polarity_MS1_unknown",
    "Polarity_MS2_unknown",
    "ScanRate_MS1",
    "ScanRate_MS2",
    "FastestFrequency_MS1",
    "FastestFrequency_MS2",
    "AvgCycleTime_MS1",
    "EmptyScans_MS1",
    "EmptyScans_MS2",
    "MzRange_MS2",
    "RtRange_MS1",
    "RtRange_MS2",
    "RT_MS1_Quantiles",
    "RT_MS2_Quantiles",
    "RT_MS1_IQR",
    "RT_MS1_IQRRate",
    "TIC_MS1_Area",
    "TIC_MS2_Area",
    "TIC_MS1_Area_RTQuantiles",
    "MedianTIC_in_RT_MS1_IQR",
    "TIC_MS1_MedianInHalfRange",
    "RT_TIC_Quantiles",
    "TIC_MS1_CV",
    "TIC_MS2_CV",
    "TIC_MS1_SignalJump10x_Count",
    "TIC_MS1_SignalFall10x_Count",
    "TIC_MS1_Change_Q2",
    "TIC_MS1_Change_Q3",
    "TIC_MS1_Change_Q4",
    "TIC_MS1_Ratio_Q2",
    "TIC_MS1_Ratio_Q3",
    "TIC_MS1_Ratio_Q4",
    "PeakDensity_MS1_Q1",
    "PeakDensity_MS1_Q2",
    "PeakDensity_MS1_Q3",
    "PeakDensity_MS2_Q1",
    "PeakDensity_MS2_Q2",
    "PeakDensity_MS2_Q3",
    "MS1_PeakType_Annotated",
    "MS1_PeakType_Annotated_ProfileFraction",
    "MS1_PeakType_Estimated",
    "MS1_PeakType_Estimated_ProfileFraction",
    "MS2_PeakType_Annotated",
    "MS2_PeakType_Annotated_ProfileFraction",
    "MS2_PeakType_Estimated",
    "MS2_PeakType_Estimated_ProfileFraction",
    "BasePeak_MS1_Mean",
    "BasePeak_MS2_Mean",
    "BasePeak_All_Max",
    "PrecursorMz_MS2_Median",
    "ChargeMin",
    "ChargeMax",
    "ChargeMean",
    "ChargeMedian",
    "ChargeRatio_3over2",
    "ChargeRatio_4over2",
    "MS2_PrecursorCharge_Fractions",
    "PrecursorIntensity_Q1",
    "PrecursorIntensity_Q2",
    "PrecursorIntensity_Q3",
    "PrecursorIntensity_Mean",
    "PrecursorIntensity_Sd",
    "PrecursorIntensity_FallbackCount",
    "ExtentPrecursorIntensity_95over5_MS2",
    "MassAnalyzers",
    "ActivationMethods",
    "FAIMS_CV_Count",
    "FAIMS_CV_Values",
    "FAIMS_CV_Range",
    "Chromatograms_RT_Min",
    "Chromatograms_RT_Max",
]

import re as _re_spec

# Dynamic metric families: keys generated at runtime (one per chromatogram type,
# activation method, mass analyzer, MS level, FAIMS field) rather than enumerated
# statically. A computed key that matches one of these patterns is considered
# covered by the specification even though it is not a fixed METRIC_METADATA key.
_DYNAMIC_METRIC_PATTERNS = [
    _re_spec.compile(r"^Chromatograms_[A-Za-z0-9 ]+$"),
    _re_spec.compile(r"^MS\d+_ActivationMethod_.+$"),
    _re_spec.compile(r"^MassAnalyzer_\d+_(Type|Resolution)$"),
    _re_spec.compile(r"^MS\d+_PeakType_.+$"),
    _re_spec.compile(r"^FAIMS_CV_.+$"),
]


def _is_dynamic_metric(name: str) -> bool:
    return any(p.match(name) for p in _DYNAMIC_METRIC_PATTERNS)


def validate_metric_registry(computed: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate a computed metric dict against the authoritative specification.

    Returns a dict of problem categories to offending metric names:
      * "uncovered": computed keys with neither a METRIC_METADATA entry nor a
        matching dynamic-family pattern (a computed metric lacking metadata);
      * "ordered_missing_metadata": METRIC_ORDER entries without a metadata entry;
      * "ordered_not_computed": METRIC_ORDER entries the run did not produce.
        METRIC_ORDER holds only static metrics, so a dynamic-looking stale entry
        (e.g. a reintroduced "MS2_ActivationMethod_0") is flagged here too;
      * "registered_orphans": metadata entries that are neither computed nor part
        of a dynamic family -- dead registrations such as the former
        MS_Run_Duration (registered but never produced).

    A test can assert every category is empty to catch registry/order/computation
    drift.
    """
    uncovered = [k for k in computed
                 if k not in METRIC_METADATA and not _is_dynamic_metric(k)]
    ordered_missing_metadata = [k for k in METRIC_ORDER if k not in METRIC_METADATA]
    ordered_not_computed = [k for k in METRIC_ORDER if k not in computed]
    registered_orphans = [k for k in METRIC_METADATA
                          if k not in computed and not _is_dynamic_metric(k)]
    return {
        "uncovered": uncovered,
        "ordered_missing_metadata": ordered_missing_metadata,
        "ordered_not_computed": ordered_not_computed,
        "registered_orphans": registered_orphans,
    }


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def _filter_by_mslevel(exp: oms.MSExperiment, level: int) -> List[oms.MSSpectrum]:
    """
    Filter MSExperiment spectra by MS level.

    Args:
        exp: MSExperiment object
        level: int, MS level to filter (1, 2, 3, etc.)

    Returns:
        list: Spectra matching the specified MS level
    """
    return [s for s in exp if s.getMSLevel() == level]

def _select_spectra(exp: oms.MSExperiment, level: int,
                    accepted_native_ids: Optional[set] = None) -> List[oms.MSSpectrum]:
    """
    Filter spectra by MS level and, optionally, by accepted identifications.

    When ``accepted_native_ids`` is None the result is every spectrum of the MS
    level (the ID-free proxy). When a set of accepted spectrum native IDs is
    supplied, only those spectra are kept, which realizes the ID-based PSI-MS
    definition ("after user-defined acceptance criteria are applied").

    Args:
        exp: MSExperiment object
        level: MS level to keep
        accepted_native_ids: optional set of accepted spectrum native IDs

    Returns:
        list: selected spectra
    """
    specs = _filter_by_mslevel(exp, level)
    if accepted_native_ids is not None:
        wanted = set(accepted_native_ids)
        specs = [s for s in specs if s.getNativeID() in wanted]
    return specs

def _rts(specs: List[oms.MSSpectrum]) -> np.ndarray:
    """
    Extract retention times from spectra.

    Args:
        specs: list of MSSpectrum objects

    Returns:
        np.ndarray: Array of retention times in seconds
    """
    return np.array([s.getRT() for s in specs], dtype=float) if specs else np.array([], dtype=float)

def _ion_counts(specs: List[oms.MSSpectrum]) -> np.ndarray:
    """
    Calculate total ion count (TIC) for each spectrum.

    Equivalent to R's ionCount() function.

    Args:
        specs: list of MSSpectrum objects

    Returns:
        np.ndarray: Array of TIC values (sum of intensities per spectrum)
    """
    return np.array([s.calculateTIC() for s in specs], dtype=float)


def _peak_counts(specs: List[oms.MSSpectrum]) -> np.ndarray:
    """
    Count the number of peaks observed in each spectrum.

    Args:
        specs: list of MSSpectrum objects

    Returns:
        np.ndarray: Array of peak counts per spectrum
    """
    return np.array([float(s.size()) for s in specs], dtype=float)


def _is_empty_spectrum(sp: oms.MSSpectrum) -> bool:
    """
    Check if a spectrum is empty or has zero total intensity.

    A spectrum is considered empty if it has no peaks or all peak intensities sum to 0.

    Args:
        sp: MSSpectrum object

    Returns:
        bool: True if spectrum is empty
    """
    if sp.size() == 0:
        return True
    return sp.calculateTIC() == 0.0

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

def _iqr(arr: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate interquartile range (Q3 - Q1).

    NaN values are removed before calculation.

    Args:
        arr: array-like numeric data

    Returns:
        float: IQR value, or NaN if insufficient data
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0: return np.nan
    q75, q25 = np.percentile(arr, [75, 25])
    return float(q75 - q25)

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of y over x (numpy 1.x/2.x compatible)."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))

def _nanmedian(arr: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate median with NaN removal.

    Args:
        arr: array-like numeric data

    Returns:
        float: Median value, or NaN if no valid data
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.median(arr)) if arr.size else np.nan

# -------------------------------------------------------------------------
# Helper functions for polarity and chromatogram analysis
# -------------------------------------------------------------------------
def _enum_int(member: Any) -> int:
    """Integer value of a pyOpenMS enum member, across binding generations.

    pyOpenMS <= 3.5 exposes enum members as plain ints (``int(member)`` works);
    pyOpenMS 3.6 switched to Python ``enum.Enum`` objects whose integer is on
    ``.value`` and for which ``int(member)`` raises ``TypeError``. Handle both.
    """
    try:
        return int(member)
    except (TypeError, ValueError):
        return int(member.value)


def _enum_name_map(enum_cls: Any) -> Dict[int, str]:
    """Build an ``int -> member-name`` mapping for a pyOpenMS enum class.

    pyOpenMS 3.4/3.5 removed the ``*ToString`` binding helpers that older code
    relied on (``IonSource.polarityToString`` etc.), so conversions are
    reconstructed from the enum members exposed as class attributes. This works
    regardless of whether members are ints (<= 3.5) or Enum objects (3.6).
    """
    mapping: Dict[int, str] = {}
    for member in dir(enum_cls):
        # Skip dunders, the getMapping helper, and the SIZE_OF sentinel. Do NOT
        # filter on isupper(): some members are legitimately mixed-case (e.g. the
        # activation methods ETciD/EThcD) and must not be dropped to "unknown".
        if member.startswith("_") or member == "getMapping" or "SIZE_OF" in member:
            continue
        try:
            mapping[_enum_int(getattr(enum_cls, member))] = member
        except (TypeError, ValueError, AttributeError):
            continue
    return mapping


# Reverse enum maps computed once at import time.
_POLARITY_NAMES = {
    _enum_int(oms.IonSource.Polarity.POSITIVE): "positive",
    _enum_int(oms.IonSource.Polarity.NEGATIVE): "negative",
}
_SPECTRUM_TYPE_NAMES = {
    _enum_int(oms.SpectrumSettings.SpectrumType.CENTROID): "centroid",
    _enum_int(oms.SpectrumSettings.SpectrumType.PROFILE): "profile",
}
_ACTIVATION_METHOD_NAMES = _enum_name_map(oms.Precursor.ActivationMethod)
_ANALYZER_TYPE_NAMES = _enum_name_map(oms.MassAnalyzer.AnalyzerType)


def _polarity_to_str(pol: Any) -> str:
    """
    Convert InstrumentSettings.Polarity enum to string.

    Args:
        pol: Polarity enum value

    Returns:
        str: "positive", "negative", or "unknown"
    """
    return _POLARITY_NAMES.get(_enum_int(pol), "unknown")


def _spectrum_type_to_str(spectrum_type: Any) -> str:
    """Convert a ``SpectrumSettings.SpectrumType`` enum value to a string.

    Returns "centroid", "profile", or "unknown".
    """
    return _SPECTRUM_TYPE_NAMES.get(_enum_int(spectrum_type), "unknown")


def _activation_method_to_str(method: Any) -> str:
    """Convert a ``Precursor.ActivationMethod`` enum value to its short name."""
    return _ACTIVATION_METHOD_NAMES.get(_enum_int(method), "unknown")


def _analyzer_type_to_str(analyzer_type: Any) -> str:
    """Convert a ``MassAnalyzer.AnalyzerType`` enum value to its name."""
    return _ANALYZER_TYPE_NAMES.get(_enum_int(analyzer_type), "unknown")


def _faims_compensation_voltages(exp: oms.MSExperiment) -> List[float]:
    """Collect distinct FAIMS compensation voltages present in the run.

    Reimplements OpenMS' ``FAIMSHelper::getCompensationVoltages`` (not bound in
    pyOpenMS 3.4/3.5). OpenMS' mzML reader stores a FAIMS compensation voltage
    (``MS:1001581``) as the spectrum *drift time* with unit
    ``FAIMS_COMPENSATION_VOLTAGE`` (see OpenMS MzMLHandler and FAIMSHelper), NOT
    as a metavalue -- so the CV is ``getDriftTime()`` for every spectrum whose
    ``getDriftTimeUnit()`` is that FAIMS unit. This matches the native
    FAIMSHelper exactly on 3.6.
    """
    faims_unit = _enum_int(oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE)
    voltages = set()
    for spec in exp:
        try:
            if _enum_int(spec.getDriftTimeUnit()) != faims_unit:
                continue
            dt = float(spec.getDriftTime())
        except (TypeError, ValueError, AttributeError):
            continue
        if np.isfinite(dt):
            voltages.add(dt)
    return sorted(voltages)

def _extract_spectrum_polarity(spec: oms.MSSpectrum) -> str:
    """
    Extract polarity from a spectrum.

    Args:
        spec: MSSpectrum object

    Returns:
        str: "positive", "negative", or "unknown"
    """
    try:
        pol = spec.getInstrumentSettings().getPolarity()
        return _polarity_to_str(pol)
    except Exception:
        return "unknown"

# -------------------------------------------------------------------------
# MsQuality Spectra metrics (translated)
# -------------------------------------------------------------------------
def chromatography_duration(exp: oms.MSExperiment) -> float:
    """
    Chromatography duration (MS:4000053).

    "The retention time duration of the chromatography in seconds." [PSI:MS]

    The metric is calculated as follows:
    (1) The retention time associated to all spectra is obtained,
    (2) The maximum and minimum retention time is obtained,
    (3) The difference between maximum and minimum is calculated and returned.

    Retention time values that are NA are removed.

    Details:
        MS:4000053
        synonym: "RT-Duration" RELATED [PMID:24494671]
        is_a: MS:4000003 ! single value
        relationship: has_metric_category MS:4000009 ! ID free metric
        relationship: has_metric_category MS:4000012 ! single run based metric
        relationship: has_metric_category MS:4000016 ! retention time metric
        relationship: has_value_type xsd:float
        relationship: has_value_concept NCIT:C25330 ! Duration
        relationship: has_units UO:0000010 ! second

    Args:
        exp: MSExperiment object

    Returns:
        float: Chromatography duration in seconds

    Example:
        >>> duration = chromatography_duration(exp)
    """
    rts_all = _rts(list(exp))
    # Filter non-finite retention times: a single NaN/inf RT would otherwise make
    # the duration (and every metric derived from it) non-finite. This matches
    # the docstring and MsQuality's na.rm=TRUE behavior.
    rts_all = rts_all[np.isfinite(rts_all)]
    return float(np.max(rts_all) - np.min(rts_all)) if rts_all.size else np.nan

def rt_over_ms_quantiles(exp: oms.MSExperiment, ms_level: int = 1) -> List[float]:
    """
    MS1 quantile RT fraction (MS:4000184) or MS2 quantile RT fraction (MS:4000185).

    Normative contract (issue #28): the current PSI-MS terms MS:4000184/MS:4000185
    describe an n-tuple of four RT *interval* widths, matching the original
    QuaMeter "RT-MS-Q1..Q4" metrics. rawQC reproduces the QuaMeter definition:

        Q1 = 25th percentile of the level's scan retention times
        Q2 = 50th percentile
        Q3 = 75th percentile
        interval_1 = (Q1 - RTmin) / duration
        interval_2 = (Q2 - Q1)  / duration
        interval_3 = (Q3 - Q2)  / duration
        interval_4 = (RTmax - Q3) / duration

    where ``duration`` is the acquisition span of *this MS level* (RTmax - RTmin
    of the level), not the whole experiment. The four intervals sum to 1.0.

    This replaces the previous behavior, which returned four cumulative RT
    endpoints, normalized by the whole-experiment duration, using an index
    partition that did not match any reference. (The obsolete MS:4000055/056
    accessions were also cited in the old docstring.) If exact MsQuality
    cumulative-endpoint compatibility is needed, it should be exposed as a
    separately named custom metric.

    Details:
        MS:4000184  synonym: "RT-MS-Q1" RELATED [PMID:24494671]  is_a: n-tuple
        MS:4000185  synonym: "RT-MSMS-Q1" RELATED [PMID:24494671]

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        list: four RT interval fractions (summing to 1.0), or [NaN]*4

    Example:
        >>> quantiles_ms1 = rt_over_ms_quantiles(exp, ms_level=1)
    """
    rts = _rts(_filter_by_mslevel(exp, ms_level))
    rts = rts[np.isfinite(rts)]
    if rts.size < 2:
        return [np.nan] * 4
    rtmin = float(np.min(rts))
    rtmax = float(np.max(rts))
    duration = rtmax - rtmin
    if duration <= 0:
        return [np.nan] * 4
    q1, q2, q3 = np.percentile(rts, [25, 50, 75])
    return [
        float((q1 - rtmin) / duration),
        float((q2 - q1) / duration),
        float((q3 - q2) / duration),
        float((rtmax - q3) / duration),
    ]

def tic_quartile_to_quartile_log_ratio(exp: oms.MSExperiment, ms_level: int = 1, mode: str = "TIC", relative_to: str = "previous") -> List[float]:
    """
    MS1 TIC-change quartile ratios (MS:4000186) or MS1 TIC quartile ratios (MS:4000187).

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)
        mode: str, either "TIC_change" or "TIC"
        relative_to: str, either "previous" or "Q1"

    Returns:
        list: Three float values representing log ratios [Q2/Q1, Q3/Q2, Q4/Q3]
              or [Q2/Q1, Q3/Q1, Q4/Q1] depending on relative_to parameter

    Note:
        For ``mode="TIC_change"`` (MS:4000186) the scan-to-scan changes are taken
        as *absolute* differences. The PSI-MS term explicitly calls the triplet
        "the original QuaMeter metrics", and QuaMeter computes the change with
        ``fabs``. Using signed ``diff`` (as MsQuality does) can yield negative
        quartiles whose ratios/logs are undefined (NaN); the absolute-change
        definition keeps every quartile non-negative and QuaMeter-compatible.

    Example:
        >>> ratios_change = tic_quartile_to_quartile_log_ratio(exp, mode="TIC_change")
        >>> ratios_tic = tic_quartile_to_quartile_log_ratio(exp, mode="TIC")
    """
    specs = _filter_by_mslevel(exp, ms_level)
    if not specs: return [np.nan, np.nan, np.nan]
    specs = sorted(specs, key=lambda s: s.getRT())
    tic = _ion_counts(specs)
    if mode == "TIC_change":
        if tic.size < 2: return [np.nan, np.nan, np.nan]
        # Absolute scan-to-scan change, matching the original QuaMeter fabs()
        # definition referenced by MS:4000186 (see Note above).
        tic = np.abs(np.diff(tic))
    qs = np.quantile(tic, [0, 0.25, 0.50, 0.75, 1.0])
    q1, q2, q3, q4 = qs[1], qs[2], qs[3], qs[4]
    with np.errstate(divide='ignore', invalid='ignore'):
        if relative_to == "Q1":
            ratios = np.array([q2/q1, q3/q1, q4/q1], dtype=float)
        else:
            ratios = np.array([q2/q1, q3/q2, q4/q3], dtype=float)
        logs = np.log(ratios)
    return [float(x) if np.isfinite(x) else np.nan for x in logs]

def scan_rate(exp: oms.MSExperiment, ms_level: int = 1) -> float:
    """
    Level-specific scan rate in scans per minute.

    The denominator is the acquisition span of the requested MS level itself
    (max minus min finite retention time for that level), not one combined
    MS1/MS2 run duration. Using the combined duration misreports a level's rate
    when one level starts later or ends earlier than the other.

    Non-finite retention times are removed. Returns NaN when the level has fewer
    than two finite retention times or a zero-length span (rate undefined).

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        float: scans per minute over that MS level's acquisition span, or NaN
    """
    rts = _rts(_filter_by_mslevel(exp, ms_level))
    rts = rts[np.isfinite(rts)]
    if rts.size < 2:
        return np.nan
    span_min = (float(np.max(rts)) - float(np.min(rts))) / 60.0
    if span_min <= 0:
        return np.nan
    return float(rts.size / span_min)


def number_spectra(exp: oms.MSExperiment, ms_level: int = 1) -> int:
    """
    Number of MS1 spectra (MS:4000059) or number of MS2 spectra (MS:4000060).

    MS:4000059:
    "The number of MS1 events in the run." [PSI:MS]

    MS:4000060:
    "The number of MS2 events in the run." [PSI:MS]

    For MS:4000059, ms_level is set to 1. For MS:4000060, ms_level is set to 2.

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The number of spectra are obtained (length) and returned.

    Details:
        MS:4000059
        synonym: "MS1-Count" EXACT [PMID:24494671]
        is_a: MS:4000003 ! single value
        relationship: has_metric_category MS:4000009 ! ID free metric
        relationship: has_units UO:0000189 ! count unit

        MS:4000060
        synonym: "MS2-Count" EXACT [PMID:24494671]
        relationship: has_metric_category MS:4000022 ! MS2 metric

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to count (default: 1)

    Returns:
        int: Number of spectra at specified MS level

    Example:
        >>> n_ms1 = number_spectra(exp, ms_level=1)
        >>> n_ms2 = number_spectra(exp, ms_level=2)
    """
    return int(len(_filter_by_mslevel(exp, ms_level)))


def peak_density_quantiles(exp: oms.MSExperiment, ms_level: int = 1,
                           probs: Tuple[float, ...] = (0.25, 0.50, 0.75)) -> List[float]:
    """
    MS1 density quantiles (MS:4000061) or MS2 density quantiles (MS:4000062).

    MS:4000061:
    "The first to n-th quantile of MS1 peak density (scan peak counts). A value
    triplet represents the original QuaMeter metrics, the quartiles of MS1
    density. The number of values in the tuple implies the quantile mode."
    [PSI:MS]

    MS:4000062:
    "The first to n-th quantile of MS2 peak density (scan peak counts). A value
    triplet represents the original QuaMeter metrics, the quartiles of MS2
    density. The number of values in the tuple implies the quantile mode."

    The metric is calculated as follows:
    (1) Filter spectra to the requested MS level and order by retention time.
    (2) Count the number of peaks observed per spectrum.
    (3) Calculate the requested quantiles of the peak counts.

    Details:
        MS:4000061
        synonym: "MS1-Density-Q1" RELATED [PMID:24494671]
        synonym: "MS1-Density-Q2" RELATED [PMID:24494671]
        synonym: "MS1-Density-Q3" RELATED [PMID:24494671]
        is_a: MS:4000004 ! n-tuple
        relationship: has_metric_category MS:4000009 ! ID free metric
        relationship: has_metric_category MS:4000021 ! MS1 metric
        relationship: has_value_concept STATO:0000291 ! quantile

        MS:4000062
        synonym: "MS2-Density-Q1" RELATED [PMID:24494671]
        synonym: "MS2-Density-Q2" RELATED [PMID:24494671]
        synonym: "MS2-Density-Q3" RELATED [PMID:24494671]
        relationship: has_metric_category MS:4000022 ! MS2 metric
        relationship: has_value_concept STATO:0000291 ! quantile

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)
        probs: tuple of quantile probabilities to calculate (default: quartiles)

    Returns:
        list: Float values representing the requested quantiles

    Example:
        >>> q_ms1 = peak_density_quantiles(exp, ms_level=1)
        >>> q_ms2 = peak_density_quantiles(exp, ms_level=2)
    """
    specs = sorted(_filter_by_mslevel(exp, ms_level), key=lambda s: s.getRT())
    # QuaMeter (the origin of MS:4000061/MS:4000062) skips spectra with
    # defaultArrayLength == 0 before accumulating peak counts, so zero-length
    # scans must not enter the density distribution as artificial 0-peak scans.
    # Empty scans are reported separately by MS:4000099/MS:4000100.
    specs = [s for s in specs if s.size() > 0]
    if not specs:
        return [np.nan for _ in probs]

    peak_counts = _peak_counts(specs)
    if peak_counts.size == 0:
        return [np.nan for _ in probs]

    qs = np.quantile(peak_counts, probs)
    return [float(x) if np.isfinite(x) else np.nan for x in qs]


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

def rt_acquisition_range(exp: oms.MSExperiment, ms_level: int = 1) -> Tuple[float, float]:
    """
    Retention time acquisition range (MS:4000070).

    MS:4000070:
    "Upper and lower limit of retention time at which spectra are recorded."
    [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The retention time values of the features within the spectra are obtained,
    (3) The minimum and maximum retention time values are obtained and returned.

    Details:
        MS:4000070
        is_a: MS:4000004 ! n-tuple
        relationship: has_metric_category MS:4000009 ! ID free metric
        relationship: has_metric_category MS:4000016 ! retention time metric
        relationship: has_units UO:0000010 ! second
        relationship: has_value_concept STATO:0000035 ! range

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        tuple: (min_rt, max_rt) as floats in seconds

    Example:
        >>> rt_min, rt_max = rt_acquisition_range(exp, ms_level=1)
    """
    specs = _filter_by_mslevel(exp, ms_level)
    rts = _rts(specs)
    if rts.size == 0: return (np.nan, np.nan)
    return (float(np.min(rts)), float(np.max(rts)))

def ms_signal_10x_change(exp: oms.MSExperiment, change: str = "jump", ms_level: int = 1) -> int:
    """
    MS1 signal jump (10x) count (MS:4000097) or MS1 signal fall (10x) count (MS:4000098).

    MS:4000097:
    "The number of times where MS1 TIC increased more than 10-fold between
    adjacent MS1 scans. An unusual high count of signal jumps or falls can
    indicate ESI stability issues." [PSI:MS]

    MS:4000098:
    "The number of times where MS1 TIC decreased more than 10-fold between
    adjacent MS1 scans. An unusual high count of signal jumps or falls can
    indicate ESI stability issues." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The spectra are ordered by retention time,
    (3) The intensity values of the features are obtained via ion count,
    (4) The signal jumps/declines of the intensity values with the two
        subsequent intensity values is calculated,
    (5) For MS:4000097, signal jumps by a factor of ten or more are counted;
        For MS:4000098, signal declines by a factor of ten or more are counted.

    Details:
        MS:4000097
        synonym: "IS-1A" RELATED []
        is_a: MS:4000003 ! single value
        relationship: has_metric_category MS:4000009 ! ID free metric
        relationship: has_metric_category MS:4000021 ! MS1 metric
        relationship: has_units UO:0000189 ! count unit

        MS:4000098
        synonym: "IS-1B" RELATED []

    Note:
        This function uses ionCount as an equivalent to the TIC.

    Args:
        exp: MSExperiment object
        change: str, either "jump" or "fall"
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        int: Count of 10x signal changes

    Example:
        >>> jumps = ms_signal_10x_change(exp, change="jump", ms_level=1)
        >>> falls = ms_signal_10x_change(exp, change="fall", ms_level=1)
    """
    if change not in ("jump", "fall"):
        raise ValueError(f"change must be 'jump' or 'fall', got {change!r}")
    specs = _filter_by_mslevel(exp, ms_level)
    # The CV terms require an integer count. With fewer than two spectra there
    # are no adjacent pairs, so the count is 0 (not NaN).
    if len(specs) < 2:
        return 0
    specs = sorted(specs, key=lambda s: s.getRT())
    tic = _ion_counts(specs)
    prev, foll = tic[:-1], tic[1:]
    # Only adjacent pairs with a finite, strictly positive previous TIC define a
    # meaningful fold-change. Pairs with a non-finite or zero denominator (empty
    # or missing scans) are excluded rather than being turned into inf/NaN by the
    # division. A fall to exactly zero (prev>0, foll==0) still counts as a >=10x
    # fall; a jump *from* zero is undefined and is not counted.
    valid = np.isfinite(prev) & np.isfinite(foll) & (prev > 0.0)
    ratio = foll[valid] / prev[valid]
    if change == "jump":
        return int(np.sum(ratio >= 10.0))
    else:
        return int(np.sum(ratio <= 0.1))

def number_empty_scans(exp: oms.MSExperiment, ms_level: int = 1) -> int:
    """
    Number of empty MS1 scans (MS:4000099), MS2 scans (MS:4000100), or MS3 scans (MS:4000101).

    MS:4000099:
    "Number of MS1 scans where the scans' peaks intensity sums to 0
    (i.e. no peaks or only 0-intensity peaks)." [PSI:MS]

    MS:4000100:
    "Number of MS2 scans where the scans' peaks intensity sums to 0
    (i.e. no peaks or only 0-intensity peaks)." [PSI:MS]

    MS:4000101:
    "Number of MS3 scans where the scans' peaks intensity sums to 0
    (i.e. no peaks or only 0-intensity peaks)." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The intensities per entry are obtained,
    (3) The number of intensity entries that are NULL, NA, or have a sum of 0
        are obtained and returned.

    Details:
        MS:4000099
        is_a: MS:4000003 ! single value
        relationship: has_metric_category MS:4000009 ! ID free metric
        relationship: has_metric_category MS:4000021 ! MS1 metric
        relationship: has_units UO:0000189 ! count unit

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        int: Count of empty scans

    Example:
        >>> empty_ms1 = number_empty_scans(exp, ms_level=1)
        >>> empty_ms2 = number_empty_scans(exp, ms_level=2)
    """
    specs = _filter_by_mslevel(exp, ms_level)
    return int(np.sum([_is_empty_spectrum(s) for s in specs]))

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

def rt_iqr(exp: oms.MSExperiment, ms_level: int = 1,
           accepted_native_ids: Optional[set] = None) -> float:
    """
    ID-free proxy for the interquartile RT period (MS:4000153, C-2A).

    MS:4000153 is an ID-based term ("after user-defined acceptance criteria are
    applied"). Without identifications rawQC computes the interquartile RT period
    over ALL spectra of the MS level, which is an ID-free proxy and is emitted
    WITHOUT the MS:4000153 accession. Pass ``accepted_native_ids`` (accepted
    spectrum native IDs) to restrict the computation to identified spectra and
    reproduce the ID-based definition.

    MS:4000153:
    "The interquartile retention time period, in seconds, for all quantification
    data points after user-defined acceptance criteria are applied over the
    complete run. Longer times indicate better chromatographic separation." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The retention time values are obtained,
    (3) The interquartile range is obtained from the values and returned
        (NA values are removed).

    Details:
        MS:4000153
        synonym: "C-2A" RELATED [PMID:19837981]
        is_a: MS:4000003 ! single value
        is_a: MS:4000008 ! ID based
        relationship: has_units UO:0000010 ! second

    Note:
        Retention time values that are NA are removed.
        The stored retention time information may have a different unit than seconds.

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        float: Interquartile range of retention times

    Example:
        >>> iqr = rt_iqr(exp, ms_level=1)
    """
    return _iqr(_rts(_select_spectra(exp, ms_level, accepted_native_ids)))

def rt_iqr_rate(exp: oms.MSExperiment, ms_level: int = 1,
                accepted_native_ids: Optional[set] = None) -> float:
    """
    ID-free proxy for the interquartile-RT-period rate (MS:4000154, C-2B).

    Like :func:`rt_iqr`, MS:4000154 is ID-based. Without identifications this is
    an ID-free proxy over all spectra of the MS level, emitted without the
    accession; ``accepted_native_ids`` restricts it to identified spectra.

    MS:4000154:
    "The rate of identified quantification data points for the interquartile
    retention time period, in identified quantification data points per second.
    Higher rates indicate efficient sampling and identification." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The retention time values are obtained,
    (3) The 25% and 75% quantiles are obtained from the retention time values
        (NA values are removed),
    (4) The number of eluted features between the 25% and 75% quantile is calculated,
    (5) The number of features is divided by the interquartile range of the
        retention time and returned.

    Details:
        MS:4000154
        synonym: "C-2B" RELATED [PMID:19837981]
        is_a: MS:4000003 ! single value
        is_a: MS:4000008 ! ID based
        relationship: has_units UO:0000106 ! hertz

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        float: Rate of features per second in IQR range

    Example:
        >>> rate = rt_iqr_rate(exp, ms_level=1)
    """
    specs = _select_spectra(exp, ms_level, accepted_native_ids)
    rts = _rts(specs)
    if rts.size == 0: return np.nan
    qs = np.quantile(rts, [0.25, 0.75])
    n = int(np.sum((rts >= qs[0]) & (rts <= qs[1])))
    denom = _iqr(rts)
    if not np.isfinite(denom) or denom == 0: return np.nan
    return float(n / denom)

def area_under_tic(exp: oms.MSExperiment, ms_level: int = 1) -> float:
    """
    Area under TIC (MS:4000155).

    MS:4000155:
    "The area under the total ion chromatogram." [PSI:MS]

    Decision (issue #30): the CV terms MS:4000029/MS:4000030/MS:4000155 describe
    an *area under a curve*, so rawQC computes a true **time integral** of the
    TIC against retention time (trapezoidal rule), not a bare per-spectrum sum.
    On irregularly sampled data a sum is not an area and has different
    dimensions. Unit: intensity x second.

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level and ordered by RT,
    (2) The TIC (ion count) is integrated over retention time (trapezoidal).

    Details:
        MS:4000155
        is_a: MS:4000003 ! single value
        relationship: has_metric_category MS:4000009 ! ID free metric
        relationship: has_metric_category MS:4000017 ! chromatogram metric

    Note:
        Requires at least two finite-RT spectra to define a time interval;
        otherwise NaN is returned (an area needs a non-zero RT span).

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        float: Time integral of the TIC over retention time (area under TIC)

    Example:
        >>> area = area_under_tic(exp, ms_level=1)
    """
    specs = _filter_by_mslevel(exp, ms_level)
    if not specs:
        return np.nan
    rts = _rts(specs)
    tic = _ion_counts(specs)
    # Drop non-finite pairs BEFORE ordering: sorting spectra by a NaN retention
    # time is unreliable (NaN comparisons are false), which would leave the
    # arrays unsorted and yield a negative "area" from the trapezoidal rule.
    finite = np.isfinite(rts) & np.isfinite(tic)
    rts, tic = rts[finite], tic[finite]
    if rts.size < 2:
        return np.nan
    order = np.argsort(rts)
    rts, tic = rts[order], tic[order]
    return _trapz(tic, rts)

def area_under_tic_rt_quantiles(exp: oms.MSExperiment, ms_level: int = 1) -> List[float]:
    """
    Area under TIC RT quantiles (MS:4000156).

    MS:4000156:
    "The area under the total ion chromatogram of the retention time quantiles.
    Number of quantiles are given by the n-tuple." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The spectra are ordered according to retention time,
    (3) The 0%, 25%, 50%, 75%, and 100% quantiles of the retention time
        values are obtained,
    (4) The ion count of the intervals between the 0%/25%, 25%/50%,
        50%/75%, and 75%/100% are obtained,
    (5) The ion counts of the intervals are summed (TIC) and the values returned.

    Details:
        MS:4000156
        is_a: MS:4000004 ! n-tuple
        is_a: MS:4000009 ! ID free
        is_a: MS:4000017 ! chromatogram metric

    Note:
        This function interprets the quantiles from [PSI:MS] definition as
        quartiles, i.e. the 0, 25, 50, 75 and 100% quantiles are used.
        Consistent with issue #30, each quartile value is a trapezoidal time
        integral of the TIC over its RT sub-interval (intensity x second), not a
        per-spectrum sum; the four values sum to the whole-run integral.

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        list: Four float values representing areas for each RT quartile

    Example:
        >>> areas = area_under_tic_rt_quantiles(exp, ms_level=1)
    """
    specs = _filter_by_mslevel(exp, ms_level)
    if len(specs) == 0: return [np.nan]*4
    rts = _rts(specs)
    tic = _ion_counts(specs)
    # Drop non-finite pairs before ordering (see area_under_tic).
    finite = np.isfinite(rts) & np.isfinite(tic)
    rts, tic = rts[finite], tic[finite]
    if rts.size < 2:
        return [np.nan] * 4
    order = np.argsort(rts)
    rts, tic = rts[order], tic[order]
    qs = np.quantile(rts, [0.0, 0.25, 0.50, 0.75, 1.0])
    # Integrate the TIC over retention time (issue #30, area-under-curve) and
    # split the integral at the quartile RT boundaries (issue #31: all four
    # values, the minimum-RT scan included via the boundary at qs[0], and the
    # four areas conserving the whole-run integral). The CUMULATIVE trapezoidal
    # integral interpolated at each boundary handles partial trapezoids that
    # straddle a boundary and never spuriously collapses a sparse quartile to 0.
    seg = 0.5 * (tic[1:] + tic[:-1]) * (rts[1:] - rts[:-1])
    cumint = np.concatenate(([0.0], np.cumsum(seg)))  # integral from rts[0] to rts[i]
    bounds = np.interp(qs, rts, cumint)                # cumulative area at each quartile RT
    return [float(bounds[i + 1] - bounds[i]) for i in range(4)]

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

def median_tic_rt_iqr(exp: oms.MSExperiment, ms_level: int = 1,
                      accepted_native_ids: Optional[set] = None) -> float:
    """
    ID-free proxy for the median TIC over the middle-half RT range (MS:4000158).

    MS:4000158 is ID-based. Without identifications this is computed over all
    spectra of the MS level (ID-free proxy, emitted without the accession);
    ``accepted_native_ids`` restricts it to identified spectra.

    MS:4000158:
    "Median of TIC values in the RT range in which half of quantification data
    points are identified (RT values of Q1 to Q3 of identifications). These
    data points may be for example XIC profiles, isotopic pattern areas, or
    reporter ions." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The spectra are ordered according to retention time,
    (3) The features between the 1st and 3rd quartile are obtained
        (half of the features that are present in the spectra),
    (4) The ion count of the features within the 1st and 3rd quartile is obtained,
    (5) The median value of the ion count is calculated (NA values are removed)
        and the median value is returned.

    Details:
        MS:4000158
        is_a: MS:4000001 ! QC metric
        is_a: MS:4000003 ! single value
        is_a: MS:4000008 ! ID based

    Note:
        This function uses ionCount as an equivalent to the TIC.
        Uses index-based quartiling (matching R implementation).

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        float: Median TIC in the RT IQR range

    Example:
        >>> median_tic = median_tic_rt_iqr(exp, ms_level=1)
    """
    specs = _select_spectra(exp, ms_level, accepted_native_ids)
    if not specs: return np.nan
    specs = sorted(specs, key=lambda s: s.getRT())
    tic = _ion_counts(specs)
    # Reproduce the MsQuality R partition exactly:
    #   ind <- rep(seq_len(4), length.out = n); ind <- sort(ind)
    #   Q1ToQ3 <- spectra[ind %in% c(2, 3), ]
    # np.resize recycles [1,2,3,4] element-wise to length n (== R's length.out),
    # then sorting yields R's recycle-and-sort group assignment. The previous
    # code used np.repeat with ceil(n/4) contiguous blocks, which gives a
    # different partition whenever n is not divisible by 4 (e.g. n=5: R group
    # sizes 2/1/1/1 vs. the old 2/2/1/0).
    n = len(specs)
    ind = np.sort(np.resize(np.arange(1, 5), n))
    sel = (ind == 2) | (ind == 3)
    return _nanmedian(tic[sel])

def median_tic_of_rt_range(exp: oms.MSExperiment, ms_level: int = 1,
                           accepted_native_ids: Optional[set] = None) -> float:
    """
    ID-free proxy for the median TIC over the shortest half-RT range (MS:4000159, MS1-2B).

    MS:4000159 is ID-based. Without identifications this is computed over all
    spectra of the MS level (ID-free proxy, emitted without the accession);
    ``accepted_native_ids`` restricts it to identified spectra.

    MS:4000159:
    "Median of TIC values in the shortest RT range in which half of the
    quantification data points are identified. These data points may be for
    example XIC profiles, isotopic pattern areas, or reporter ions." [PSI:MS]

    The metric is calculated as follows:
    (1) The spectra are filtered according to the MS level,
    (2) The spectra are ordered according to retention time,
    (3) The number of features in the spectra is obtained and the number for
        half of the features is calculated,
    (4) Iterate through the features (always by taking the neighbouring half
        of features) and calculate the retention time range of the set of features,
    (5) Retrieve the set of features with the minimum retention time range,
    (6) Calculate from the set of (5) the median TIC (NA values are removed)
        and return it.

    Details:
        MS:4000159
        synonym: "MS1-2B" RELATED [PMID:19837981]
        is_a: MS:4000001 ! QC metric
        is_a: MS:4000003 ! single value
        is_a: MS:4000008 ! ID based

    Note:
        This function uses ionCount as an equivalent to the TIC.
        Uses ceiling division (matching R implementation).

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        float: Median TIC in the shortest half-RT window

    Example:
        >>> median_tic = median_tic_of_rt_range(exp, ms_level=1)
    """
    specs = _select_spectra(exp, ms_level, accepted_native_ids)
    n = len(specs)
    if n == 0: return np.nan
    specs = sorted(specs, key=lambda s: s.getRT())
    rts = _rts(specs)
    tic = _ion_counts(specs)
    # Use ceiling like R: n_half <- ceiling(n / 2)
    half = int(np.ceil(n / 2))
    best_span, best_slice = None, None
    for i in range(0, n - half + 1):
        span = rts[i + half - 1] - rts[i]
        if best_span is None or span < best_span:
            best_span = span
            best_slice = slice(i, i + half)
    return _nanmedian(tic[best_slice])

def tic_quantile_rt_fraction(exp: oms.MSExperiment, ms_level: int = 1) -> List[float]:
    """
    TIC accumulation RT quantiles (MS:4000183).

    MS:4000183:
    "The interval when the respective quantile of the TIC accumulates divided by
    retention time duration. The number of values in the tuple implies the
    quantile mode." [PSI:MS]

    Normative contract (issue #32): the current term describes an n-tuple of
    RT *intervals*, and the original QuaMeter implementation returns four
    consecutive interval widths normalized by the MS-level duration. rawQC
    reproduces the QuaMeter definition:

        t1,t2,t3 = retention times at which the cumulative TIC first reaches
                   25%, 50%, 75% of the total TIC
        interval_1 = (t1 - RTmin) / duration
        interval_2 = (t2 - t1)    / duration
        interval_3 = (t3 - t2)    / duration
        interval_4 = (RTmax - t3) / duration

    where ``duration`` is the acquisition span of *this MS level* (RTmax - RTmin
    of the level), not the whole experiment. The four intervals sum to 1.0.

    This replaces the previous behavior, which emitted five cumulative RT
    positions normalized by the whole-experiment duration across all MS levels.
    If the MsQuality cumulative-position representation is needed, it should be
    exposed as a separately named custom metric.

    Details:
        MS:4000183
        synonym: "RT-TIC-Q1..Q4" RELATED [PMID:24494671]
        is_a: MS:4000004 ! n-tuple
        relationship: has_units UO:0000191 ! fraction

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)

    Returns:
        list: four RT interval fractions (summing to 1.0), or [NaN]*4

    Example:
        >>> fractions = tic_quantile_rt_fraction(exp, ms_level=1)
    """
    specs = _filter_by_mslevel(exp, ms_level)
    if not specs:
        return [np.nan] * 4
    rts = _rts(specs)
    tic = _ion_counts(specs)
    # Drop non-finite pairs BEFORE ordering: sorting spectra by a NaN retention
    # time is unreliable, so filtering after the sort would leave the arrays
    # unsorted and make rts[0]/rts[-1] and the cumulative TIC meaningless.
    finite = np.isfinite(rts) & np.isfinite(tic)
    rts, tic = rts[finite], tic[finite]
    if rts.size < 2:
        return [np.nan] * 4
    order = np.argsort(rts)
    rts, tic = rts[order], tic[order]
    rtmin, rtmax = float(rts[0]), float(rts[-1])
    duration = rtmax - rtmin
    if duration <= 0:
        return [np.nan] * 4
    cum = np.cumsum(tic)
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0:
        return [np.nan] * 4
    frac = cum / total
    # RT at which cumulative TIC first reaches each threshold.
    t1 = float(rts[int(np.argmax(frac >= 0.25))])
    t2 = float(rts[int(np.argmax(frac >= 0.50))])
    t3 = float(rts[int(np.argmax(frac >= 0.75))])
    return [
        (t1 - rtmin) / duration,
        (t2 - t1) / duration,
        (t3 - t2) / duration,
        (rtmax - t3) / duration,
    ]

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

# -------------------------------------------------------------------------
# Additional metrics
# -------------------------------------------------------------------------
def polarity_statistics(exp: oms.MSExperiment) -> Dict[str, Any]:
    """
    Extract polarity statistics per MS level.

    Returns counts of positive, negative, and unknown polarity scans
    for each MS level present in the experiment.

    Args:
        exp: MSExperiment object

    Returns:
        dict: Nested dictionary with MS levels and polarity counts

    Example:
        >>> stats = polarity_statistics(exp)
        >>> print(stats['MS1']['positive'])
    """
    from collections import defaultdict, Counter
    pol_per_level = defaultdict(Counter)

    for spec in exp:
        lvl = int(spec.getMSLevel())
        pol = _extract_spectrum_polarity(spec)
        pol_per_level[lvl][pol] += 1

    # Convert to regular dict with MS level as string keys
    result = {}
    for lvl in sorted(pol_per_level.keys()):
        result[f"MS{lvl}"] = dict(pol_per_level[lvl])

    return result

def avg_ms1_cycle_time(exp: oms.MSExperiment) -> float:
    """
    Average MS1 cycle time (mean ΔRT between consecutive MS1 scans).

    This metric indicates the average time between MS1 scans, which is
    useful for understanding acquisition duty cycle and sampling rate.

    Args:
        exp: MSExperiment object

    Returns:
        float: Average MS1 cycle time in seconds, or NaN if insufficient MS1 scans

    Example:
        >>> cycle_time = avg_ms1_cycle_time(exp)
    """
    ms1_specs = _filter_by_mslevel(exp, 1)
    if len(ms1_specs) < 2:
        return np.nan

    rts = _rts(ms1_specs)
    rts_sorted = np.sort(rts)

    if len(rts_sorted) < 2:
        return np.nan

    diffs = np.diff(rts_sorted)
    # Guard against pathological zeros
    diffs = diffs[diffs > 0]

    if diffs.size == 0:
        return np.nan

    return float(np.mean(diffs))


def fastest_ms_frequency(exp: oms.MSExperiment, ms_level: int = 1, window: float = 60.0) -> float:
    """
    Fastest frequency for MS level 1 collection (MS:4000065) or MS level 2 collection (MS:4000066).

    MS:4000065:
    "Fastest frequency for MS level 1 collection" [PSI:MS]

    MS:4000066:
    "Fastest frequency for MS level 2 collection" [PSI:MS]

    The original QuaMeter definition (PMID:24494671) is the *maximum acquisition
    rate sustained over a one-minute window*, not the inverse of the smallest gap
    between two scans. Reporting ``1 / min-gap`` lets a single unusually close
    pair of scans dominate the metric with an arbitrarily high value.

    This implementation reproduces the QuaMeter/macproqc one-minute moving
    window: for each scan at time ``t`` it counts how many scans fall in the
    interval ``[t, t + window]`` (inclusive), takes the maximum such count over
    all scans, and divides by the window length (60 s) to obtain a frequency in
    hertz. Because any single fast pair adds at most one scan to a window, it
    cannot dominate the result.

    Behavior for short inputs:
        * No finite retention times -> NaN.
        * If the level spans less than ``window`` seconds, the moving window
          still divides by the full window length, so the reported frequency is
          ``n_scans / window`` -- a conservative lower bound consistent with the
          reference implementations (there is no full one-minute window to
          average over).

    Details:
        MS:4000065
        synonym: "MS1-Freq-Max" EXACT [PMID:24494671]
        relationship: has_units UO:0000106 ! hertz
        MS:4000066
        synonym: "MS2-Freq-Max" EXACT [PMID:24494671]

    Args:
        exp: MSExperiment object
        ms_level: int, MS level to analyze (default: 1)
        window: float, moving-window length in seconds (default: 60.0)

    Returns:
        float: Fastest sustained acquisition frequency in hertz, or NaN if unavailable
    """
    rts = _rts(_filter_by_mslevel(exp, ms_level))
    rts = rts[np.isfinite(rts)]
    if rts.size == 0:
        return np.nan
    srt = np.sort(rts)
    # For each anchor t = srt[i]: count of scans in [t, t + window].
    hi = np.searchsorted(srt, srt + window, side="right")
    lo = np.searchsorted(srt, srt, side="left")
    max_count = int(np.max(hi - lo))
    return float(max_count / window)


def _peak_type_summary(counts: Dict[str, int]) -> str:
    """Summarize per-level peak-type counts into a stable run-wide label."""
    centroid = counts.get("centroid", 0)
    profile = counts.get("profile", 0)
    if centroid and profile:
        return "mixed"
    if profile:
        return "profile"
    if centroid:
        return "centroid"
    return "unknown"


def _profile_fraction(counts: Dict[str, int]) -> float:
    """Fraction of type-resolved spectra that are profile (NaN if none)."""
    denom = counts.get("centroid", 0) + counts.get("profile", 0)
    return float(counts.get("profile", 0) / denom) if denom else np.nan


def peak_type_statistics(exp: oms.MSExperiment) -> Dict[str, Any]:
    """
    Determine peak type (profile vs centroided) per MS level.

    Aggregates over **all** spectra of each MS level rather than sampling the
    first spectrum, so a mixed profile/centroid run (or a misleading first
    spectrum) is reported as "mixed" instead of being presented as homogeneous.
    For each level and for both the metadata annotation and the peak-spacing
    estimation, a stable summary label is produced:

        * "centroid" / "profile" -- all type-resolved spectra agree
        * "mixed"                 -- both centroid and profile occur
        * "unknown"               -- no spectrum carried a resolvable type

    A companion ``*_ProfileFraction`` value gives the fraction of type-resolved
    spectra that are profile, exposing annotation-vs-estimation disagreement and
    the degree of mixing.

    Args:
        exp: MSExperiment object

    Returns:
        dict: Peak-type summaries and profile fractions per MS level
    """
    from collections import defaultdict

    annotated_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    estimated_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    estimator = oms.PeakTypeEstimator()
    for spec in exp:
        level = int(spec.getMSLevel())
        annotated_counts[level][_spectrum_type_to_str(spec.getType())] += 1
        # Estimation needs enough peaks to be meaningful.
        if spec.size() > 10:
            estimated_counts[level][_spectrum_type_to_str(estimator.estimateType(spec))] += 1

    result: Dict[str, Any] = {}
    for level in sorted(set(annotated_counts) | set(estimated_counts)):
        ann = annotated_counts.get(level, {})
        est = estimated_counts.get(level, {})
        result[f"MS{level}_PeakType_Annotated"] = _peak_type_summary(ann)
        result[f"MS{level}_PeakType_Annotated_ProfileFraction"] = _profile_fraction(ann)
        result[f"MS{level}_PeakType_Estimated"] = _peak_type_summary(est)
        result[f"MS{level}_PeakType_Estimated_ProfileFraction"] = _profile_fraction(est)

    return result

def activation_method_statistics(exp: oms.MSExperiment) -> Dict[str, Any]:
    """
    Count activation methods per MS level as a single table.

    Returns a table (dict of columns) rather than one dynamic key per method, so
    an arbitrary number of activation methods is preserved without fixed key
    slots and the result is a single valid custom metric.

    Args:
        exp: MSExperiment object

    Returns:
        dict: table with columns ms_level, method, count (possibly empty columns)
    """
    from collections import Counter

    counts: "Counter" = Counter()
    for spec in exp:
        level = int(spec.getMSLevel())
        for pc in spec.getPrecursors():
            for am in pc.getActivationMethods():
                counts[(level, _activation_method_to_str(am))] += 1

    rows = sorted(counts.items())
    return {
        "ms_level": [lvl for (lvl, _name), _c in rows],
        "method": [name for (_lvl, name), _c in rows],
        "count": [c for _key, c in rows],
    }

def mass_analyzer_info(exp: oms.MSExperiment) -> Dict[str, Any]:
    """
    Extract mass analyzer information as a single table.

    Returns a table (dict of columns) covering every mass analyzer, rather than
    fixed MassAnalyzer_0/MassAnalyzer_1 key slots. Conversion no longer silently
    drops analyzers on error.

    Args:
        exp: MSExperiment object

    Returns:
        dict: table with columns index, type, resolution
    """
    indices: List[int] = []
    types: List[str] = []
    resolutions: List[Optional[float]] = []
    try:
        analyzers = exp.getInstrument().getMassAnalyzers()
    except Exception:
        analyzers = []
    for idx, ma in enumerate(analyzers or []):
        indices.append(idx)
        types.append(_analyzer_type_to_str(ma.getType()))
        res = float(ma.getResolution())
        resolutions.append(res if res > 0 else None)
    return {"index": indices, "type": types, "resolution": resolutions}

def number_of_ms_levels(exp: oms.MSExperiment) -> int:
    """
    Count the number of distinct MS levels in the experiment.

    Args:
        exp: MSExperiment object

    Returns:
        int: Number of distinct MS levels
    """
    levels = set()
    for spec in exp:
        levels.add(int(spec.getMSLevel()))

    return len(levels)

def total_peak_count(exp: oms.MSExperiment) -> int:
    """
    Count total number of peaks across all spectra.

    Args:
        exp: MSExperiment object

    Returns:
        int: Total peak count
    """
    return sum(spec.size() for spec in exp)

def max_base_peak_intensity(exp: oms.MSExperiment) -> float:
    """
    Maximum base peak intensity across all spectra (MS:4000202).

    MS:4000202:
    "The maximum base peak intensity of all spectra in a single run." [PSI:MS]

    Every spectrum is considered regardless of MS level (MS1, MS2, MS3, ...),
    matching the term definition and independent implementations that iterate
    all spectra. Spectra with no peaks (empty scans) are skipped. If the run
    contains no non-empty spectra, NaN is returned.

    Args:
        exp: MSExperiment object

    Returns:
        float: Maximum base peak intensity across the whole run, or NaN if empty
    """
    max_bp = np.nan
    for spec in exp:
        if spec.size() == 0:
            continue
        _, intens = spec.get_peaks()
        if intens.size == 0:
            continue
        m = float(np.max(intens))
        if np.isnan(max_bp) or m > max_bp:
            max_bp = m
    return max_bp

def chromatogram_point_count(exp: oms.MSExperiment) -> int:
    """
    Count total number of chromatogram data points.

    This sums the array lengths of all chromatograms, i.e. the number of stored
    (RT, intensity) data points, NOT the number of resolved chromatographic
    peaks. Peak detection is not performed here, so the metric is named
    accordingly (see NumberOfChromatogramDataPoints).

    Args:
        exp: MSExperiment object

    Returns:
        int: Total chromatogram data-point count
    """
    return sum(chrom.size() for chrom in exp.getChromatograms())


# OpenMS ChromatogramSettings.ChromatogramType (int) -> stable short key used for
# the Chromatograms_* metric names. Any value not listed maps to "Unknown".
_CHROM_TYPE_KEYS = {
    0: "XIC",   # MASS_CHROMATOGRAM (extracted-ion / mass chromatogram)
    1: "TIC",   # TOTAL_ION_CURRENT_CHROMATOGRAM
    2: "SIC",   # SELECTED_ION_CURRENT_CHROMATOGRAM
    3: "BPC",   # BASEPEAK_CHROMATOGRAM
    4: "SIM",   # SELECTED_ION_MONITORING_CHROMATOGRAM
    5: "SRM",   # SELECTED_REACTION_MONITORING_CHROMATOGRAM
}

def faims_compensation_voltages(exp: oms.MSExperiment) -> Dict[str, Any]:
    """
    Extract FAIMS compensation voltages if present.

    Args:
        exp: MSExperiment object

    Returns:
        dict: FAIMS CV information
    """
    result: Dict[str, Any] = {}

    cvs = _faims_compensation_voltages(exp)
    if cvs:
        values = [float(cv) for cv in cvs]
        # Distinguish the distinct values, their [min, max] range, and the count.
        # Voltages are in volts (documented in the metric metadata).
        result["FAIMS_CV_Count"] = len(values)
        result["FAIMS_CV_Values"] = values
        result["FAIMS_CV_Range"] = [min(values), max(values)]

    return result

def chromatogram_statistics(exp: oms.MSExperiment) -> Dict[str, Any]:
    """
    Extract chromatogram statistics from the experiment.

    Analyzes all chromatograms in the mzML file to determine:
    - Total number of chromatograms
    - Counts by type keyed by stable short names (TIC, BPC, SRM, SIM, XIC, SIC,
      Unknown), mapped from the OpenMS ChromatogramType enum
    - RT range covered by chromatograms

    RT-range policy: only finite chromatogram RT bounds contribute. If there are
    no chromatograms (or none with finite RT bounds, e.g. all empty), the RT
    range is (NaN, NaN).

    Args:
        exp: MSExperiment object

    Returns:
        dict: Chromatogram statistics including counts, types, and RT range

    Example:
        >>> stats = chromatogram_statistics(exp)
        >>> print(stats['total_chromatograms'])
    """
    from collections import Counter

    chroms = exp.getChromatograms()
    chrom_total = len(chroms)

    chrom_type_counts: "Counter" = Counter()
    chrom_rt_min = np.nan
    chrom_rt_max = np.nan

    for ch in chroms:
        key = _CHROM_TYPE_KEYS.get(int(ch.getChromatogramType()), "Unknown")
        chrom_type_counts[key] += 1

        # RT coverage: empty chromatograms have no (uninitialized) range and must
        # not corrupt the overall bounds. Only non-empty chromatograms with
        # finite RT bounds contribute.
        if ch.size() == 0:
            continue
        ch.updateRanges()
        cmin = ch.getMinRT()
        cmax = ch.getMaxRT()
        if np.isfinite(cmin):
            chrom_rt_min = cmin if np.isnan(chrom_rt_min) else min(chrom_rt_min, cmin)
        if np.isfinite(cmax):
            chrom_rt_max = cmax if np.isnan(chrom_rt_max) else max(chrom_rt_max, cmax)

    return {
        "total_chromatograms": chrom_total,
        "counts_by_type": dict(chrom_type_counts),
        "rt_range_min": chrom_rt_min,
        "rt_range_max": chrom_rt_max,
    }

# -------------------------------------------------------------------------
# Compute metrics (your originals + MsQuality ports)
# -------------------------------------------------------------------------
def compute_qc_metrics(exp: oms.MSExperiment) -> Dict[str, Any]:
    """Compute QC metrics and return them in a stable, presentation-ready order."""
    ms1_specs = _filter_by_mslevel(exp, 1)
    ms2_specs = _filter_by_mslevel(exp, 2)

    tic_ms1 = _ion_counts(ms1_specs)
    tic_ms2 = _ion_counts(ms2_specs)

    # Level-specific scan rates use each level's own acquisition span (see
    # scan_rate); the previous combined MS1+MS2 duration misreported a level's
    # rate whenever the two levels had different RT spans.
    scan_rate_ms1 = scan_rate(exp, 1)
    scan_rate_ms2 = scan_rate(exp, 2)

    density_ms1 = peak_density_quantiles(exp, 1)
    density_ms2 = peak_density_quantiles(exp, 2)
    chrom_stats = chromatogram_statistics(exp)
    charge_info = charge_metrics(exp, 2)
    pol_stats = polarity_statistics(exp)
    rt_quantiles_ms1 = rt_over_ms_quantiles(exp, 1)
    rt_quantiles_ms2 = rt_over_ms_quantiles(exp, 2)
    qareas = area_under_tic_rt_quantiles(exp, 1)
    tfr = tic_quantile_rt_fraction(exp, 1)

    def _safe_get(values, index):
        try:
            return values[index]
        except (IndexError, TypeError):
            return np.nan

    computed: Dict[str, Any] = {}

    # General overview
    computed["NumberOfMSLevels"] = number_of_ms_levels(exp)
    computed["NumberOfSpectra_MS1"] = number_spectra(exp, 1)
    computed["NumberOfSpectra_MS2"] = number_spectra(exp, 2)
    computed["MS1_to_MS2_Ratio"] = float(len(ms1_specs) / len(ms2_specs)) if len(ms2_specs) > 0 else np.nan
    computed["ChromatographyDuration"] = chromatography_duration(exp)
    computed["NumberOfChromatograms"] = chrom_stats["total_chromatograms"]
    computed["NumberOfChromatogramDataPoints"] = chromatogram_point_count(exp)
    computed["NumberOfSpectralPeaks"] = total_peak_count(exp)

    for level in (1, 2):
        level_key = f"MS{level}"
        level_counts = pol_stats.get(level_key, {})
        for polarity in ("positive", "negative", "unknown"):
            key = f"Polarity_{level_key}_{polarity}"
            computed[key] = int(level_counts.get(polarity, 0))

    computed["ScanRate_MS1"] = scan_rate_ms1
    computed["ScanRate_MS2"] = scan_rate_ms2
    computed["FastestFrequency_MS1"] = fastest_ms_frequency(exp, 1)
    computed["FastestFrequency_MS2"] = fastest_ms_frequency(exp, 2)
    computed["AvgCycleTime_MS1"] = avg_ms1_cycle_time(exp)
    computed["EmptyScans_MS1"] = number_empty_scans(exp, 1)
    computed["EmptyScans_MS2"] = number_empty_scans(exp, 2)

    # Precursor m/z range (MS:4000069) is defined for MSn only; MS1 spectra have
    # no precursor, so no MS1 precursor range is emitted. Each range is one
    # two-value [min, max] n-tuple.
    computed["MzRange_MS2"] = [float(x) for x in mz_acquisition_range(exp, 2)]
    computed["RtRange_MS1"] = [float(x) for x in rt_acquisition_range(exp, 1)]
    computed["RtRange_MS2"] = [float(x) for x in rt_acquisition_range(exp, 2)]

    # MS:4000184/MS:4000185 emitted as one four-value interval n-tuple each.
    computed["RT_MS1_Quantiles"] = [float(x) for x in rt_quantiles_ms1]
    computed["RT_MS2_Quantiles"] = [float(x) for x in rt_quantiles_ms2]
    computed["RT_MS1_IQR"] = rt_iqr(exp, 1)
    computed["RT_MS1_IQRRate"] = rt_iqr_rate(exp, 1)

    computed["TIC_MS1_Area"] = area_under_tic(exp, 1)
    computed["TIC_MS2_Area"] = area_under_tic(exp, 2)
    # All four RT-quartile areas are emitted as one MS:4000156 n-tuple (the old
    # code exposed only Q1-Q3 as separate scalars and discarded Q4).
    computed["TIC_MS1_Area_RTQuantiles"] = [float(x) for x in qareas]
    computed["MedianTIC_in_RT_MS1_IQR"] = median_tic_rt_iqr(exp, 1)
    computed["TIC_MS1_MedianInHalfRange"] = median_tic_of_rt_range(exp, 1)
    # MS:4000183 emitted as one four-value interval n-tuple.
    computed["RT_TIC_Quantiles"] = [float(x) for x in tfr]
    # Coefficient of variation uses the *sample* standard deviation (ddof=1),
    # consistent with the precursor-intensity SD (MS:4000118) and R's sd().
    computed["TIC_MS1_CV"] = float(np.std(tic_ms1, ddof=1) / np.mean(tic_ms1)) if tic_ms1.size > 1 and np.mean(tic_ms1) else np.nan
    computed["TIC_MS2_CV"] = float(np.std(tic_ms2, ddof=1) / np.mean(tic_ms2)) if tic_ms2.size > 1 and np.mean(tic_ms2) else np.nan

    computed["TIC_MS1_SignalJump10x_Count"] = ms_signal_10x_change(exp, "jump", 1)
    computed["TIC_MS1_SignalFall10x_Count"] = ms_signal_10x_change(exp, "fall", 1)

    tc = tic_quartile_to_quartile_log_ratio(exp, 1, mode="TIC_change", relative_to="previous")
    tr = tic_quartile_to_quartile_log_ratio(exp, 1, mode="TIC", relative_to="previous")
    computed["TIC_MS1_Change_Q2"] = _safe_get(tc, 0)
    computed["TIC_MS1_Change_Q3"] = _safe_get(tc, 1)
    computed["TIC_MS1_Change_Q4"] = _safe_get(tc, 2)
    computed["TIC_MS1_Ratio_Q2"] = _safe_get(tr, 0)
    computed["TIC_MS1_Ratio_Q3"] = _safe_get(tr, 1)
    computed["TIC_MS1_Ratio_Q4"] = _safe_get(tr, 2)

    computed["PeakDensity_MS1_Q1"] = _safe_get(density_ms1, 0)
    computed["PeakDensity_MS1_Q2"] = _safe_get(density_ms1, 1)
    computed["PeakDensity_MS1_Q3"] = _safe_get(density_ms1, 2)
    computed["PeakDensity_MS2_Q1"] = _safe_get(density_ms2, 0)
    computed["PeakDensity_MS2_Q2"] = _safe_get(density_ms2, 1)
    computed["PeakDensity_MS2_Q3"] = _safe_get(density_ms2, 2)

    peak_types = peak_type_statistics(exp)
    computed.update(peak_types)

    base_peaks_ms1: List[float] = []
    base_peaks_ms2: List[float] = []
    for spec in ms1_specs:
        if spec.size() > 0:
            _, intens = spec.get_peaks()
            if intens.size > 0:
                base_peaks_ms1.append(float(np.max(intens)))
    for spec in ms2_specs:
        if spec.size() > 0:
            _, intens = spec.get_peaks()
            if intens.size > 0:
                base_peaks_ms2.append(float(np.max(intens)))
    computed["BasePeak_MS1_Mean"] = float(np.mean(base_peaks_ms1)) if base_peaks_ms1 else np.nan
    computed["BasePeak_MS2_Mean"] = float(np.mean(base_peaks_ms2)) if base_peaks_ms2 else np.nan
    # MS:4000202 is defined over *all* spectra regardless of MS level, so it must
    # include MS3+ base peaks, not just the MS1/MS2 subsets collected above.
    computed["BasePeak_All_Max"] = max_base_peak_intensity(exp)

    computed["PrecursorMz_MS2_Median"] = median_precursor_mz(exp, 2)
    computed["ExtentPrecursorIntensity_95over5_MS2"] = extent_identified_precursor_intensity(exp, 2)
    computed.update(precursor_intensity_stats(exp, 2))
    _, _n_prec_fallback, _ = precursor_intensities(ms2_specs)
    computed["PrecursorIntensity_FallbackCount"] = int(_n_prec_fallback)

    computed["ChargeMin"] = charge_info.get("ChargeMin", np.nan)
    computed["ChargeMax"] = charge_info.get("ChargeMax", np.nan)
    computed["ChargeRatio_3over2"] = charge_info.get("ChargeRatio_3over2", np.nan)
    computed["ChargeRatio_4over2"] = charge_info.get("ChargeRatio_4over2", np.nan)
    computed["ChargeMean"] = charge_info.get("ChargeMean", np.nan)
    computed["ChargeMedian"] = charge_info.get("ChargeMedian", np.nan)
    computed["MS2_PrecursorCharge_Fractions"] = charge_info.get("MS2_PrecursorCharge_Fractions")

    # Acquisition/instrument facts as single valid custom metrics (tables) with
    # no fixed key slots, rather than dynamic per-method/per-analyzer keys.
    computed["MassAnalyzers"] = mass_analyzer_info(exp)
    computed["ActivationMethods"] = activation_method_statistics(exp)
    computed.update(faims_compensation_voltages(exp))

    # counts_by_type is already keyed by stable short names (TIC/BPC/SRM/SIM/XIC/
    # SIC/Unknown), so the metric names map directly to METRIC_METADATA.
    for chrom_type, count in chrom_stats["counts_by_type"].items():
        computed[f"Chromatograms_{chrom_type}"] = count
    computed["Chromatograms_RT_Min"] = chrom_stats["rt_range_min"]
    computed["Chromatograms_RT_Max"] = chrom_stats["rt_range_max"]

    desired_order = METRIC_ORDER

    ordered_metrics: Dict[str, Any] = {}
    remaining_metrics = dict(computed)

    for metric_name in desired_order:
        if metric_name in remaining_metrics:
            ordered_metrics[metric_name] = remaining_metrics.pop(metric_name)

    for metric_name, value in remaining_metrics.items():
        ordered_metrics[metric_name] = value

    return ordered_metrics
# -------------------------------------------------------------------------
# Metadata extraction (as in your script, with tiny safety tweaks)
# -------------------------------------------------------------------------
def extract_instrument_metadata(exp: oms.MSExperiment) -> Dict[str, str]:
    instrument_metadata = {}
    settings = exp.getExperimentalSettings()
    inst = settings.getInstrument()
    instrument_metadata["Instrument model name"] = inst.getName()
    instrument_metadata["Manufacturer"] = inst.getVendor()
    sw = inst.getSoftware()
    if sw.getName():
        instrument_metadata["Software"] = sw.getName() + ((" " + sw.getVersion()) if sw.getVersion() else "")
    analyzers = inst.getMassAnalyzers()
    if analyzers:
        try:
            instrument_metadata["Analyzer resolution"] = analyzers[0].getResolution()
        except Exception:
            pass
    if settings.getSourceFiles():
        sf = settings.getSourceFiles()[0]
        instrument_metadata["Original file name"] = sf.getNameOfFile()
        instrument_metadata["Original path"] = sf.getPathToFile()
    return instrument_metadata

# -------------------------------------------------------------------------
# Build mzQC with accessions
# -------------------------------------------------------------------------
import re as _re

# PSI-MS CV version rawQC's MS:* accessions are taken from. Update alongside the
# accessions; recorded in the generated file rather than a stale hard-coded value.
PSI_MS_CV_VERSION = "4.1.257"

# Local CV for rawQC custom metrics that have no PSI-MS accession. Every emitted
# quality metric must carry an accession matching ^[A-Z]+:[A-Z0-9]+$ (mzQC
# schema); custom metrics use a deterministic LOCAL:* accession from this CV.
_LOCAL_CV = qc.ControlledVocabulary(
    name="rawQC local quality metrics",
    version="1",
    uri="https://github.com/bigbio/rawQC",
)

# Instrument-metadata keys with a well-defined PSI-MS accession; others use LOCAL.
_INSTRUMENT_CV_ACCESSIONS = {
    "Instrument model name": ("MS:1000031", "instrument model"),
    "Software": ("MS:1000531", "software"),
}


def _local_accession(name: str) -> str:
    """Deterministic schema-valid LOCAL accession for a custom metric name."""
    suffix = _re.sub(r"[^A-Z0-9]", "", name.upper())
    return f"LOCAL:{suffix or 'UNNAMED'}"


def _jsonify_value(v: Any) -> Any:
    """Coerce a metric value to a JSON-serialisable form (scalar/array/table)."""
    def _clean(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return None
        return x

    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: ([_clean(x) for x in col] if isinstance(col, (list, tuple)) else _clean(col))
                for k, col in v.items()}
    return str(v)


def build_mzqc(run_data: List[Dict[str, Any]]) -> str:
    """
    Build mzQC JSON from multiple runs.

    Args:
        run_data: List of dictionaries, each containing:
            - 'filename': str
            - 'metrics': Dict[str, Any]
            - 'instrument_metadata': Dict[str, str]


    Returns:
        str: JSON-formatted mzQC string
    """
    cv_qc = qc.ControlledVocabulary(
        name="Proteomics Standards Initiative Quality Control Ontology",
        version="0.1.0",
        uri="https://github.com/HUPO-PSI/qcML-development/blob/master/cv/v0_1_0/qc-cv.obo"
    )
    cv_ms = qc.ControlledVocabulary(
        name="Proteomics Standards Initiative Mass Spectrometry Ontology",
        version=PSI_MS_CV_VERSION,
        uri="https://github.com/HUPO-PSI/psi-ms-CV/blob/master/psi-ms.obo"
    )

    # AnalysisSoftware is a CV term and must carry a valid accession; use the
    # generic PSI-MS "software" term (MS:1000531).
    anso = qc.AnalysisSoftware(
        accession="MS:1000531",
        name="pyOpenMS",
        version=str(getattr(oms, "__version__", "3.x")),
        uri="https://www.openms.de",
        description="OpenMS Python bindings used for ID-free QC metric computation and metadata extraction"
    )

    run_qualities = []

    for idx, run_info in enumerate(run_data, 1):
        mzml_file = run_info['filename']
        input_name = os.path.splitext(os.path.basename(mzml_file))[0]
        metrics_dict = run_info['metrics']
        instrument_metadata = run_info['instrument_metadata']

        # Instrument/input metadata belongs in the metadata structure, not in
        # qualityMetrics: attach it as fileProperties (CvParameters) of the input.
        file_properties = []
        for k, v in instrument_metadata.items():
            if v is None or v == "":
                continue
            acc, cv_name = _INSTRUMENT_CV_ACCESSIONS.get(k, (_local_accession(k), k))
            file_properties.append(qc.CvParameter(
                accession=acc, name=cv_name, value=_jsonify_value(v)))

        infi = qc.InputFile(name=input_name,
                            location=mzml_file,
                            fileFormat=qc.CvParameter(accession="MS:1000584", name="mzML format"),
                            fileProperties=file_properties)

        meta = qc.MetaDataParameters(
            inputFiles=[infi],
            analysisSoftware=[anso],
            label=f"run{idx}"
        )

        qmetrics = []
        for k, v in metrics_dict.items():
            metric_meta = METRIC_METADATA.get(k, {})
            description = metric_meta.get("description") or "rawQC ID-free QC metric"
            # Every emitted metric must have a valid accession + name. Use the
            # PSI-MS accession where declared; otherwise a deterministic LOCAL:*
            # accession backed by the rawQC local CV (custom metric).
            accession = metric_meta.get("accession") or _local_accession(k)

            qmetrics.append(qc.QualityMetric(
                name=k,
                accession=accession,
                value=_jsonify_value(v),
                description=description
            ))

        rq = qc.RunQuality(metadata=meta, qualityMetrics=qmetrics)
        run_qualities.append(rq)

    mzqc_obj = qc.MzQcFile(
        version="1.0.0",
        creationDate=datetime.now().isoformat(),
        runQualities=run_qualities,
        setQualities=[],
        controlledVocabularies=[cv_qc, cv_ms, _LOCAL_CV]
    )

    json_str = json.dumps(json.loads(qc.JsonSerialisable.to_json(mzqc_obj)), indent=2)
    return json_str

# -------------------------------------------------------------------------
# Table formatting functions
# -------------------------------------------------------------------------
def format_value(value: Any) -> str:
    """Format metric values for readable display."""
    if value is None:
        return "N/A"
    elif isinstance(value, float):
        if abs(value) < 0.001 or abs(value) > 1000:
            return f"{value:.2e}"
        else:
            return f"{value:.4f}"
    elif isinstance(value, int):
        return str(value)
    return str(value)

def create_table(headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> str:
    """Create a formatted table using pandas DataFrame."""
    import pandas as pd
    
    if not rows:
        return f"\n=== {title} ===\nNo data available.\n"
    
    # Create DataFrame from headers and rows
    df = pd.DataFrame(rows, columns=headers)
    
    # Convert to string with nice formatting
    table_str = df.to_string(index=False)
    
    # Add title if provided
    if title:
        return f"\n=== {title} ===\n{table_str}\n"
    else:
        return f"{table_str}\n"

def parse_mzqc_metrics(json_str: str) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, List[Any]]]:
    """
    Parse mzQC JSON with multiple runs and organize metrics.

    Returns:
        Tuple of (run_labels, qc_metrics_dict, instrument_metrics_dict)
        where each dict maps metric names to lists of values (one per run),
        preserving the original run order. Missing metrics are aligned by
        inserting 'N/A' placeholders for runs where the metric is absent.
    """
    try:
        data = json.loads(json_str)
        run_qualities = data['mzQC']['runQualities']

        run_labels = []
        qc_metrics_dict: Dict[str, Dict[str, Any]] = {}
        instrument_metrics_dict: Dict[str, List[Any]] = {}

        for run_idx, run in enumerate(run_qualities):
            # Get run label/filename
            label = run['metadata'].get('label', 'unknown')
            input_files = run['metadata'].get('inputFiles', [])
            if input_files:
                filename = input_files[0].get('name', label)
            else:
                filename = label
            run_labels.append(filename)

            # Track metrics present in this run
            seen_qc_this_run: set = set()
            seen_instr_this_run: set = set()

            metrics = run['qualityMetrics']

            # Snapshot keys that existed before processing this run
            prev_qc_keys = set(qc_metrics_dict.keys())
            prev_instr_keys = set(instrument_metrics_dict.keys())

            # Instrument/input metadata now lives in the metadata structure as
            # inputFile fileProperties, not as quality metrics.
            file_properties = (input_files[0].get('fileProperties') or []) if input_files else []
            for prop in file_properties:
                clean_name = prop.get('name', '')
                if clean_name not in instrument_metrics_dict:
                    instrument_metrics_dict[clean_name] = [format_value(None)] * run_idx
                instrument_metrics_dict[clean_name].append(format_value(prop.get('value')))
                seen_instr_this_run.add(clean_name)

            for metric in metrics:
                name = metric['name']
                value = metric.get('value')
                accession = metric.get('accession', '-')
                description = metric.get('description', '')

                formatted_value = format_value(value)

                # QC metrics - store with accession and description
                if name not in qc_metrics_dict:
                    # First time we see this metric -> pad previous runs
                    qc_metrics_dict[name] = {
                        'values': [format_value(None)] * run_idx,
                        'accession': accession,
                        'description': description
                    }
                # If we already have metadata, keep the first occurrence's accession/description
                qc_metrics_dict[name]['values'].append(formatted_value)
                seen_qc_this_run.add(name)

            # Append placeholder for any previously known metrics not present in this run
            missing_qc = prev_qc_keys - seen_qc_this_run
            for m in missing_qc:
                qc_metrics_dict[m]['values'].append(format_value(None))

            missing_instr = prev_instr_keys - seen_instr_this_run
            for m in missing_instr:
                instrument_metrics_dict[m].append(format_value(None))

        return run_labels, qc_metrics_dict, instrument_metrics_dict

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing mzQC JSON: {e}")
        return [], {}, {}

def print_metrics_tables(json_str: str) -> None:
    """Print formatted tables with QC metrics for multiple runs, showing values side-by-side."""
    run_labels, qc_metrics_dict, instrument_metrics_dict = parse_mzqc_metrics(json_str)

    if not run_labels:
        print("No data to display.")
        return

    print("\
" + "="*120)
    print("QUALITY METRICS SUMMARY")
    print("="*120)
    print(f"\
Analyzing {len(run_labels)} run(s): {', '.join(run_labels)}\
")

    # Instrument Metadata table
    if instrument_metrics_dict:
        headers = ["Property"] + run_labels
        rows = []
        for prop, values in sorted(instrument_metrics_dict.items()):
            rows.append([prop] + values)
        print(create_table(headers, rows, "INSTRUMENT METADATA"))


    # QC Metrics table - preserve insertion order from compute_qc_metrics()
    if qc_metrics_dict:
        headers = ["Metric Name"] + run_labels + ["CV Accession", "Description"]
        rows = []
        # Don't sort - preserve the logical ordering from compute_qc_metrics()
        for metric_name, metric_data in qc_metrics_dict.items():
            row = [metric_name] + metric_data['values'] + [metric_data['accession'], metric_data['description']]
            rows.append(row)
        print(create_table(headers, rows, "QC METRICS"))

# -------------------------------------------------------------------------
# TSV writer
# -------------------------------------------------------------------------
def derive_tsv_output_path(output_json_path: str) -> str:
    """
    Derive a TSV output path from the mzQC JSON output path.
    If the path ends with '.mzQC' (case-insensitive), replace it by '.tsv'.
    If the path ends with '.mzQC.json' (case-insensitive), replace it by '.tsv'.
    Otherwise, replace a generic '.json' extension by '.tsv' or append '.tsv'.
    """
    base = output_json_path
    lower = base.lower()
    if lower.endswith('.mzqc'):
        return base[: -len('.mzqc')] + '.tsv'
    if lower.endswith('.mzqc.json'):
        return base[: -len('.mzQC.json')] + '.tsv'
    if lower.endswith('.json'):
        return os.path.splitext(base)[0] + '.tsv'
    return base + '.tsv'


def write_metrics_tsv(json_str: str, tsv_path: str) -> None:
    """
    Write metrics table to a TSV file.
    Leading comment lines (prefixed with '#') contain instrument information per run.
    The table contains QC metrics with values for each run.
    """
    import pandas as pd

    run_labels, qc_metrics_dict, instrument_metrics_dict = parse_mzqc_metrics(json_str)

    lines: List[str] = []
    if run_labels:
        # Header with run labels
        lines.append("# Instrument metadata (values aligned to runs below)")
        lines.append("# Runs\t" + "\t".join(str(x) for x in run_labels))
        if instrument_metrics_dict:
            for prop, values in instrument_metrics_dict.items():
                # Ensure alignment with number of runs
                vals = list(values)
                if len(vals) < len(run_labels):
                    vals += [""] * (len(run_labels) - len(vals))
                lines.append("# " + prop + "\t" + "\t".join(vals))
        else:
            lines.append("# (no instrument metadata available)")
    else:
        lines.append("# (no runs found)")

    # Blank line before data table
    lines.append("")

    # Build metrics table (metrics only)
    headers = ["Metric Name"] + run_labels
    rows: List[List[str]] = []
    for metric_name, metric_data in qc_metrics_dict.items():
        vals = list(metric_data['values'])
        if len(vals) < len(run_labels):
            vals += [""] * (len(run_labels) - len(vals))
        accession = metric_data.get('accession')
        if accession and str(accession).strip() not in ("-", "None"):
            name_with_acc = f"{metric_name} ({accession})"
        else:
            name_with_acc = metric_name
        rows.append([name_with_acc] + vals)

    df = pd.DataFrame(rows, columns=headers) if rows else pd.DataFrame(columns=headers)

    # Write file
    with open(tsv_path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
        df.to_csv(fh, sep="\t", index=False)


def _flatten_metric_for_heatmap(name: str, value: Any) -> List[Tuple[str, float]]:
    """
    Expand a metric value into ``(row_label, float)`` pairs for the heatmap.

    Scalars map to a single row. n-tuples (lists) are expanded to ``name[i]``
    rows, and tables (dicts of columns) to ``name.column[label]`` rows using the
    first all-string column as the row label. Non-numeric entries are skipped, so
    tuple/table metrics (charge fractions, RT quantiles, ranges, activation/
    analyzer tables) are visualized instead of silently dropped.
    """
    def _num(x: Any) -> Optional[float]:
        if isinstance(x, bool) or x is None:
            return None
        try:
            f = float(x)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) else np.nan

    rows: List[Tuple[str, float]] = []
    if value is None or isinstance(value, str):
        return rows
    if isinstance(value, bool):
        return rows
    if isinstance(value, (int, float)):
        f = _num(value)
        if f is not None:
            rows.append((name, f))
    elif isinstance(value, (list, tuple)):
        for i, x in enumerate(value):
            f = _num(x)
            if f is not None:
                rows.append((f"{name}[{i}]", f))
    elif isinstance(value, dict):
        labels = None
        for col, vals in value.items():
            if isinstance(vals, (list, tuple)) and vals and all(isinstance(v, str) for v in vals):
                labels = list(vals)
                break
        for col, vals in value.items():
            if not isinstance(vals, (list, tuple)):
                continue
            if vals and all(isinstance(v, str) for v in vals):
                continue  # label / other string column, not a numeric series
            if list(vals) == list(range(len(vals))):
                continue  # a pure positional index column (0,1,...,n-1)
            for i, x in enumerate(vals):
                f = _num(x)
                if f is None:
                    continue
                lbl = labels[i] if (labels is not None and i < len(labels)) else str(i)
                rows.append((f"{name}.{col}[{lbl}]", f))
    return rows


# -------------------------------------------------------------------------
# Core function for library usage
# -------------------------------------------------------------------------
def calculate_metrics(
    mzml_files: List[str],
    output_file: Optional[str] = "multi_run_qc.mzQC",
    generate_plot: bool = True,
    plot_output: str = "idfree_qc_plot.png",
    show_tables: bool = False,
    show_json: bool = False,
    cmap_name: str = "RdBu_r",
    continue_on_error: bool = False
) -> bool:
    """
    Calculate QC metrics for one or more mzML files and generate mzQC output.
    
    This is the core function that can be imported and used programmatically.
    
    Args:
        mzml_files: List of paths to mzML files to process
        output_file: Path to save the mzQC JSON output (default: "multi_run_qc.mzQC").
                     Set to None to skip saving to file.
        generate_plot: Whether to generate a heatmap visualization (default: True)
        plot_output: Path to save the plot (default: "idfree_qc_plot.png")
        show_tables: Whether to print formatted metric tables (default: True)
        show_json: Whether to print the full JSON output (default: False)
        cmap_name: Colormap name for heatmap (default: "RdBu_r")
        continue_on_error: Continue processing other files if one fails (default: False)
    
    Returns:
        bool: True if any errors occurred during processing, False otherwise
        
    Example:
        >>> from pyopenms_idfreeqc.calculate_metrics import calculate_metrics
        >>> error_occurred = calculate_metrics(
        ...     mzml_files=["sample1.mzML", "sample2.mzML"],
        ...     output_file="my_qc.json"
        ... )
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    print("Processing mzML files and computing QC metrics...")
    all_run_data = []
    error_occurred = False

    for filename in mzml_files:
        print(f"\nProcessing {filename}...")
        
        try:
            # Load mzML
            fh = oms.MzMLFile()
            exp = oms.MSExperiment()
            fh.load(filename, exp)
            exp.updateRanges()

            # Compute metrics
            metrics = compute_qc_metrics(exp)
            instrument_meta = extract_instrument_metadata(exp)

            # Store run data
            all_run_data.append({
                'filename': filename,
                'metrics': metrics,
                'instrument_metadata': instrument_meta
            })

            print(f"  ✓ Computed {len(metrics)} QC metrics for {filename}")
            
        except Exception as e:
            error_occurred = True
            print(f"  ✗ Error processing {filename}: {e}")
            
            if not continue_on_error:
                print("Tip: Use --continue-on-error to continue processing despite errors")
                raise
            else:
                print(f"  → Skipping {filename} and continuing with remaining files")
                continue

    # Check if we have any successful files to process
    if not all_run_data:
        if error_occurred:
            print("\n✗ No files were successfully processed due to errors.")
            if continue_on_error:
                return True  # Return error status
            else:
                raise RuntimeError("No files were successfully processed")
        else:
            raise ValueError("No mzML files provided for processing")
    
    # Build mzQC JSON with all runs
    print(f"\nBuilding mzQC JSON file from {len(all_run_data)} successfully processed files...")
    json_str = build_mzqc(all_run_data)

    if show_json:
        print("="*120)
        print("mzQC JSON OUTPUT")
        print("="*120)
        print(json_str)
        print("="*120)

    # Print metrics as nice formatted tables with multiple runs side-by-side
    if show_tables:
        print_metrics_tables(json_str)

    # Save to file
    if output_file:
        with open(output_file, "w") as fh:
            fh.write(json_str)
        print(f"\n✓ mzQC file saved to: {output_file}")
        # Also save TSV metrics table next to JSON
        try:
            tsv_output_path = derive_tsv_output_path(output_file)
            write_metrics_tsv(json_str, tsv_output_path)
            print(f"✓ TSV metrics table saved to: {tsv_output_path}")
        except Exception as e:
            print(f"Warning: Failed to save TSV metrics table: {e}")

    # Generate plot if requested
    if generate_plot:
        # Build heatmap rows directly from the structured mzQC values so that
        # n-tuples and tables (charge fractions, RT quantiles, ranges, analyzer/
        # activation tables) are expanded into per-element rows rather than
        # dropped because a list/dict cannot occupy a single heatmap cell.
        mzqc_data = json.loads(json_str)
        run_qualities = mzqc_data["mzQC"]["runQualities"]
        num_runs = len(run_qualities)

        run_labels = []
        for rq in run_qualities:
            input_files = rq["metadata"].get("inputFiles", [])
            run_labels.append(input_files[0].get("name") if input_files
                              else rq["metadata"].get("label", "run"))

        all_metrics_for_heatmap: Dict[str, List[float]] = {}
        for run_idx, rq in enumerate(run_qualities):
            for qm in rq["qualityMetrics"]:
                for row_label, fval in _flatten_metric_for_heatmap(qm["name"], qm.get("value")):
                    all_metrics_for_heatmap.setdefault(row_label, [np.nan] * num_runs)[run_idx] = fval

        # Create a DataFrame (rows = metrics, columns = runs)
        df_heatmap = pd.DataFrame(all_metrics_for_heatmap, index=run_labels).T

        # Drop rows where all values are NaN
        df_heatmap = df_heatmap.dropna(how='all')

        # Store the original DataFrame for annotations
        df_heatmap_original = df_heatmap.copy()

        # Scale data for color intensity (row-wise) using NumPy
        def _scale_row_with_numpy(row: pd.Series) -> pd.Series:
            """Scale numeric values in a Series using z-score normalization with NumPy."""
            values = row.to_numpy(dtype=float, copy=True)
            mask = np.isfinite(values)
            if mask.sum() <= 1:
                return row

            finite_values = values[mask]
            mean = finite_values.mean()
            std = finite_values.std(ddof=0)

            if std == 0:
                values[mask] = 0.0
            else:
                values[mask] = (finite_values - mean) / std
            values[~mask] = np.nan
            return pd.Series(values, index=row.index)

        # Apply scaling row-wise
        df_heatmap_scaled = df_heatmap.apply(_scale_row_with_numpy, axis=1, result_type='broadcast')
        
        def _row_to_strs(row: pd.Series) -> pd.Series:
            # decide if all finite values are integers
            finite = row.dropna().to_numpy(dtype=float)
            all_int = (finite.size > 0) and np.allclose(finite, np.round(finite), atol=1e-8)
            if all_int:
                return row.apply(lambda x: "" if pd.isna(x) else f"{int(round(float(x)))}")
            else:
                # Round to 2 decimal places for non-integer floats
                return row.apply(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")

        annot_df = df_heatmap_original.apply(_row_to_strs, axis=1)

        # Plot
        # Build colormap from parameter with NA color
        try:
            cmap_obj = sns.color_palette(cmap_name, as_cmap=True)
        except Exception:
            try:
                cmap_obj = mpl.colormaps.get_cmap(cmap_name)
            except Exception:
                try:
                    cmap_obj = mpl.cm.get_cmap(cmap_name)
                except Exception:
                    cmap_obj = sns.color_palette("RdBu_r", as_cmap=True)
        try:
            cmap_obj.set_bad("lightgray")
        except Exception:
            pass

        # Shorten run names to base name without extension (applies to both data and annotations)
        rename_cols = {col: os.path.splitext(os.path.basename(str(col)))[0] for col in df_heatmap_scaled.columns}
        df_heatmap_scaled_renamed = df_heatmap_scaled.copy().rename(columns=rename_cols)
        annot_df_renamed = annot_df.copy().rename(columns=rename_cols)

        # Plot heatmap with compact horizontal colorbar at the bottom
        fig, ax = plt.subplots(figsize=(15, max(8, len(df_heatmap_scaled_renamed) * 0.6)))
        hm = sns.heatmap(
            df_heatmap_scaled_renamed.astype(float),
            annot=annot_df_renamed,
            fmt="",
            cmap=cmap_obj,
            linewidths=.5,
            center=0.0,
            cbar=True,
            cbar_kws={"orientation": "horizontal", "pad": 0.08, "shrink": 0.7, "aspect": 30}
        )

        # Cross-out NaN cells for quick visual identification
        nan_mask = df_heatmap_scaled_renamed.isna()
        n_rows, n_cols = nan_mask.shape
        for i in range(n_rows):
            for j in range(n_cols):
                if nan_mask.iat[i, j]:
                    # draw 'X' across the cell bounds [j, j+1] x [i, i+1]
                    ax.plot([j, j+1], [i, i+1], color='dimgray', lw=1.0, alpha=0.9, zorder=3, solid_capstyle='round')
                    ax.plot([j, j+1], [i+1, i], color='dimgray', lw=1.0, alpha=0.9, zorder=3, solid_capstyle='round')

        # Rotate run (x-axis) labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

        ax.set_title('QC Metrics Heatmap Across Runs (Color Normalized per Row, Annotations Original)')
        ax.set_xlabel('Run')
        ax.set_ylabel('Metric')
        fig.tight_layout()

        plt.savefig(plot_output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Heatmap saved to: {plot_output}")

    return error_occurred


# -------------------------------------------------------------------------
# Click CLI wrapper
# -------------------------------------------------------------------------
@click.command(help=textwrap.dedent("""
Calculate ID-free QC metrics for mzML mass spectrometry files.

This tool computes comprehensive quality control metrics from mzML files
and outputs results in mzQC format with optional visualizations.

FILES: One or more mzML files to process. Supports wildcards (e.g., *.mzML)

\b
Examples:
  
  # Process specific files (supports wildcards)
  
  python calculate_metrics.py sample1.mzML sample2.mzML
  python calculate_metrics.py data/*.mzML

  # Use demo files (downloads if needed)

  python calculate_metrics.py --demo --download-demo

  # Custom output paths

  python calculate_metrics.py --demo -o my_qc.json -p my_plot.png

  # Library usage in Python code:

  from pyopenms_idfreeqc.calculate_metrics import calculate_metrics
  json_output = calculate_metrics(["sample1.mzML", "sample2.mzML"])
"""))
@click.argument(
    'files',
    nargs=-1,
    type=click.Path(exists=True),
    required=False
)
@click.option(
    '--demo',
    is_flag=True,
    help='Use demo files instead of user-provided files'
)
@click.option(
    '--output',
    '-o',
    default='multi_run_qc.mzQC',
    type=click.Path(),
    help='Output path for mzQC JSON; a TSV metrics table will also be saved next to it (default: multi_run_qc.mzQC)'
)
@click.option(
    '--plot',
    '-p',
    default='idfree_qc_plot.png',
    type=click.Path(),
    help='Output plot file path (default: idfree_qc_plot.png)'
)
@click.option(
    '--no-plot',
    is_flag=True,
    help='Skip generating the heatmap plot'
)
@click.option(
    '--show-tables',
    is_flag=True,
    help='Print formatted metric tables to console'
)
@click.option(
    '--show-json',
    is_flag=True,
    help='Print the full mzQC JSON output to console'
)
@click.option(
    '--download-demo',
    is_flag=True,
    help='Download demo files before processing'
)
@click.option(
    '--cmap',
    '-c',
    default='RdBu_r',
    help='Colormap name for heatmap (e.g., RdBu_r, viridis)'
)
@click.option(
    '--continue-on-error',
    is_flag=True,
    help='Continue processing files even if an error occurs (still exits with error code)'
)
def main(files, demo, output, plot, no_plot, show_tables, show_json, download_demo, cmap, continue_on_error):
    mzml_files = []
    
    if demo:
        # Use demo files
        if download_demo:
            print("Downloading demo mzML files...")
            for url, filename in MZML_FILES.items():
                if os.path.exists(filename):
                    print(f"  Skipping {filename} (already exists)")
                    continue
                print(f"  Downloading {filename}...")
                urlretrieve(url, filename)
            print("✅ Download complete.\n")
        
        # Add demo files to processing list
        mzml_files = list(MZML_FILES.values())
        
        # Check if files exist
        missing_files = [f for f in mzml_files if not os.path.exists(f)]
        if missing_files:
            click.echo(f"Error: Demo files not found: {', '.join(missing_files)}", err=True)
            click.echo("Run with --download-demo to download them first.", err=True)
            raise click.Abort()
    else:
        # Use user-provided files
        if not files:
            click.echo("Error: No files specified. Provide mzML file paths as arguments or use --demo for demo mode.", err=True)
            click.echo("\nExamples:", err=True)
            click.echo("  python calculate_metrics.py file1.mzML file2.mzML", err=True)
            click.echo("  python calculate_metrics.py data/*.mzML", err=True)
            click.echo("  python calculate_metrics.py --demo --download-demo", err=True)
            raise click.Abort()
        mzml_files = list(files)
    
    # Validate files exist
    for f in mzml_files:
        if not os.path.exists(f):
            click.echo(f"Error: File not found: {f}", err=True)
            raise click.Abort()
    
    # Call the core function
    try:
        error_occurred = calculate_metrics(
            mzml_files=mzml_files,
            output_file=output,
            generate_plot=not no_plot,
            plot_output=plot,
            show_tables=show_tables,
            show_json=show_json,
            cmap_name=cmap,
            continue_on_error=continue_on_error
        )
        
        # Exit with error code if any errors occurred during processing
        if error_occurred:
            import sys
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        raise


if __name__ == "__main__":
    main()
