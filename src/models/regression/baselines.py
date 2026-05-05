import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .metrics import evaluate_all


class PLSBaseline:
    """
    Partial Least Squares regression — the classical chemometrics baseline for XRF.

    PLS finds latent components that maximise covariance between spectra and
    concentrations, naturally handling multicollinearity between energy bins.

    Args:
        n_components: number of PLS latent components (typical range: 10–30)
    """
    def __init__(self, n_components: int = 20):
        self.model = PLSRegression(n_components=n_components)
        self.scaler = StandardScaler()
        self._element_names: Optional[List[str]] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, element_names: Optional[List[str]] = None):
        """
        Args:
            X_train:       (N, 600) spectra
            y_train:       (N, 41)  concentration targets
            element_names: optional list of element symbols for metric reporting
        """
        self._element_names = element_names
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predicted concentrations clipped to [0, 1] and renormalized
        so each row sums to 1 (matching the Softmax behaviour of the CNN).
        """
        X_scaled = self.scaler.transform(X)
        raw = self.model.predict(X_scaled)         # (N, 41), may contain negatives
        raw = np.clip(raw, 0, None)                # no negative concentrations
        totals = raw.sum(axis=1, keepdims=True)
        totals = np.where(totals < 1e-8, 1.0, totals)
        return raw / totals

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        pred = self.predict(X)
        return evaluate_all(pred, y, element_names=self._element_names)


class RidgeBaseline:
    """
    Ridge regression (L2-regularized linear model) — a simpler linear baseline.

    Uses MultiOutputRegressor to fit one Ridge regressor per element.
    Faster than PLS but ignores inter-element correlations.

    Args:
        alpha: L2 regularisation strength
    """
    def __init__(self, alpha: float = 1.0):
        self.model = MultiOutputRegressor(Ridge(alpha=alpha), n_jobs=-1)
        self.scaler = StandardScaler()
        self._element_names: Optional[List[str]] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, element_names: Optional[List[str]] = None):
        self._element_names = element_names
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        raw = self.model.predict(X_scaled)
        raw = np.clip(raw, 0, None)
        totals = raw.sum(axis=1, keepdims=True)
        totals = np.where(totals < 1e-8, 1.0, totals)
        return raw / totals

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        pred = self.predict(X)
        return evaluate_all(pred, y, element_names=self._element_names)
