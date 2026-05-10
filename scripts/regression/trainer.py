"""Model building and training helpers."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.regression import (
    CNNRegressor, ResNetRegressor, CNNRegressorV2, MLPRegressor,
    PLSBaseline, RidgeBaseline, RegressionTrainer,
)

DL_MODELS = {
    "cnn":     CNNRegressor,
    "resnet":  ResNetRegressor,
    "cnn_v2":  CNNRegressorV2,
    "mlp":     MLPRegressor,
}
BASELINE_MODELS = {
    "pls":   PLSBaseline,
    "ridge": RidgeBaseline,
}
ALL_MODELS = list(DL_MODELS) + list(BASELINE_MODELS)


def train_dl(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-3,
    epochs: int = 60,
    batch_size: int = 64,
    patience: int = 10,
    scaler: str = "log_minmax",
    mse_weight: float = 1.0,
    kl_weight: float = 0.5,
):
    model = DL_MODELS[model_name]()
    t = RegressionTrainer(
        model,
        learning_rate=lr,
        mse_weight=mse_weight,
        kl_weight=kl_weight,
        scaler=scaler,
    )
    print(f"[train] Training {model_name}...")
    history = t.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, patience=patience,
    )
    return t, history


def train_baseline(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    element_names: list,
):
    if model_name == "pls":
        b = PLSBaseline(n_components=20)
    elif model_name == "ridge":
        b = RidgeBaseline(alpha=1.0)
    else:
        raise ValueError(f"Unknown baseline: {model_name}")
    print(f"[train] Training {model_name}...")
    b.fit(X_train, y_train, element_names=element_names)
    return b
