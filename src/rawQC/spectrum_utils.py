"""Small spectrum and numeric helpers shared by acquisition metric modules."""

from typing import List, Optional, Union

import numpy as np
import pyopenms as oms


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
