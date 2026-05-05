import numpy as np
import pandas as pd
from typing import Dict, Union


def _get_active_mask(target: np.ndarray) -> np.ndarray:
    """Returns a boolean mask where ground-truth concentration > 0."""
    return target > 0


def masked_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Mean Absolute Error computed only on elements with non-zero ground-truth.

    This is the primary regression metric. Predicting 0 for absent elements
    is trivial; only errors on active elements are meaningful.
    """
    mask = _get_active_mask(target)
    if mask.sum() == 0:
        return 0.0
    return float(np.abs(pred[mask] - target[mask]).mean())


def masked_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """MSE computed only on non-zero ground-truth elements."""
    mask = _get_active_mask(target)
    if mask.sum() == 0:
        return 0.0
    return float(((pred[mask] - target[mask]) ** 2).mean())


def masked_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """
    R² (coefficient of determination) on active elements only.

    R² = 1 - SS_res / SS_tot

    Returns NaN if the target has no variance (all concentrations equal),
    which can happen with single-element samples.
    """
    mask = _get_active_mask(target)
    if mask.sum() == 0:
        return float('nan')
    y_true = target[mask]
    y_pred = pred[mask]
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def per_element_mae(
    pred: np.ndarray,
    target: np.ndarray,
    element_names: list,
) -> pd.Series:
    """
    MAE for each element, computed only over samples where that element is active.

    Useful to identify which elements are harder to quantify.

    Args:
        pred:           (N, 41) predicted concentrations
        target:         (N, 41) ground-truth concentrations
        element_names:  list of 41 element symbols (same order as columns)

    Returns:
        pd.Series indexed by element symbol, value = MAE (NaN if element never active)
    """
    results = {}
    for i, el in enumerate(element_names):
        mask = target[:, i] > 0
        if mask.sum() == 0:
            results[el] = float('nan')
        else:
            results[el] = float(np.abs(pred[mask, i] - target[mask, i]).mean())
    return pd.Series(results, name='MAE_per_element')


def evaluate_all(
    pred: np.ndarray,
    target: np.ndarray,
    element_names: list = None,
) -> Dict[str, Union[float, pd.Series]]:
    """
    Computes all regression metrics at once.

    Returns a dict with keys:
        'masked_mae'     — primary metric
        'masked_mse'
        'masked_r2'
        'per_element_mae' — pd.Series (only if element_names provided)
    """
    metrics = {
        'masked_mae': masked_mae(pred, target),
        'masked_mse': masked_mse(pred, target),
        'masked_r2':  masked_r2(pred, target),
    }
    if element_names is not None:
        metrics['per_element_mae'] = per_element_mae(pred, target, element_names)
    return metrics
