"""Physics-informed peak feature extraction for XRF spectra."""

import numpy as np
from typing import List, Optional, Tuple

_PEAK_BINS: Optional[dict] = None
_MULTI_LINE_FEATURES: Optional[List[Tuple[str, str, int]]] = None

# Lines to consider, in priority order (strongest first)
_LINE_PRIORITY = ["ka1", "kb1", "la1", "lb1", "lg1", "ma1"]
_MIN_BIN_SEPARATION = 2   # lines closer than this are treated as duplicates


def _build_peak_bins(element_names: List[str],
                     bin_size: float = 0.05,
                     max_kev: float = 30.0) -> dict:
    from skbeam.core.constants import XrfElement
    preferred = ["ka1", "la1", "kb1", "la2", "ma1"]
    result = {}
    for elem in element_names:
        e = XrfElement(elem)
        in_range = [(n, en) for n, en in e.emission_line.all if 0 < en < max_kev]
        best = None
        for pref in preferred:
            match = next((en for n, en in in_range if n == pref), None)
            if match is not None:
                best = (pref, match, int(match / bin_size))
                break
        result[elem] = best
    return result


def _build_multi_line_features(element_names: List[str],
                                bin_size: float = 0.05,
                                max_kev: float = 30.0) -> List[Tuple[str, str, int]]:
    """
    Build a list of (element, line_name, bin_index) tuples for all non-redundant
    emission lines across all elements.

    Filters out:
    - Lines outside [0, max_kev]
    - Lines whose bin is within _MIN_BIN_SEPARATION of an already-selected bin
      for the same element (e.g. ka1/ka2 nearly coincide → keep only ka1)
    """
    from skbeam.core.constants import XrfElement
    features = []
    for elem in element_names:
        e = XrfElement(elem)
        in_range = {n: en for n, en in e.emission_line.all if 0 < en < max_kev}
        selected_bins = []
        for line in _LINE_PRIORITY:
            if line not in in_range:
                continue
            bin_idx = int(in_range[line] / bin_size)
            # Skip if too close to an already-selected line for this element
            if any(abs(bin_idx - b) < _MIN_BIN_SEPARATION for b in selected_bins):
                continue
            selected_bins.append(bin_idx)
            features.append((elem, line, bin_idx))
    return features


def get_peak_bins(element_names: List[str]) -> dict:
    global _PEAK_BINS
    if _PEAK_BINS is None or list(_PEAK_BINS.keys()) != element_names:
        _PEAK_BINS = _build_peak_bins(element_names)
    return _PEAK_BINS


def get_multi_line_features(element_names: List[str]) -> List[Tuple[str, str, int]]:
    global _MULTI_LINE_FEATURES
    if _MULTI_LINE_FEATURES is None:
        _MULTI_LINE_FEATURES = _build_multi_line_features(element_names)
    return _MULTI_LINE_FEATURES


def n_multi_line_features(element_names: List[str]) -> int:
    return len(get_multi_line_features(element_names))


def extract_peak_features(X: np.ndarray,
                          element_names: List[str],
                          window: int = 5) -> np.ndarray:
    """
    Extract primary-line peak integrals: one feature per element (41 total).

    For each element, sums counts in a ±window-bin region around its strongest
    in-range emission line.
    """
    peak_bins = get_peak_bins(element_names)
    n_bins = X.shape[1]
    features = np.zeros((X.shape[0], len(element_names)), dtype=np.float32)
    for i, elem in enumerate(element_names):
        info = peak_bins.get(elem)
        if info is None:
            continue
        _, _, bin_idx = info
        lo = max(0, bin_idx - window)
        hi = min(n_bins, bin_idx + window + 1)
        features[:, i] = X[:, lo:hi].sum(axis=1)
    return features


def extract_multi_line_features(X: np.ndarray,
                                 element_names: List[str],
                                 window: int = 5,
                                 include_ratios: bool = False) -> np.ndarray:
    """
    Extract multi-line peak integrals: one feature per non-redundant emission
    line across all elements.

    Args:
        X:              (N, 600) spectra
        element_names:  ordered list of element symbols
        window:         integration half-width in bins
        include_ratios: if True, also append ratio features
                        rᵢ = peakᵢ / (Σpeakⱼ + ε), giving a direct physical
                        proxy for concentration independent of cross-section magnitude.
                        Output shape becomes (N, 2 × n_lines).

    Returns:
        (N, n_lines) or (N, 2*n_lines) array
    """
    line_list = get_multi_line_features(element_names)
    n_bins = X.shape[1]
    absolute = np.zeros((X.shape[0], len(line_list)), dtype=np.float32)
    for i, (_, _, bin_idx) in enumerate(line_list):
        lo = max(0, bin_idx - window)
        hi = min(n_bins, bin_idx + window + 1)
        absolute[:, i] = X[:, lo:hi].sum(axis=1)

    if not include_ratios:
        return absolute

    totals = absolute.sum(axis=1, keepdims=True) + 1e-8
    ratios = absolute / totals
    return np.hstack([absolute, ratios])


def augment_with_peak_features(X: np.ndarray,
                                element_names: List[str],
                                log_spectrum: bool = True,
                                window: int = 5) -> np.ndarray:
    """Concatenate log-spectrum (600) + primary peak features (41) → (N, 641)."""
    spectrum = np.log1p(X).astype(np.float32) if log_spectrum else X.astype(np.float32)
    peaks = extract_peak_features(X, element_names, window=window)
    return np.concatenate([spectrum, peaks], axis=1)
