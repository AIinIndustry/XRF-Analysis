"""Model building and training helpers."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.regression import (
    CNNRegressor, ResNetRegressor, CNNRegressorV2, CNNRegressorV3, CNNRegressorV4,
    MLPRegressor, PhysicsInformedRegressor,
    PLSBaseline, RidgeBaseline, RegressionTrainer,
    augment_with_peak_features, extract_peak_features,
)
from src.models.regression.peak_features import extract_multi_line_features, n_multi_line_features

DL_MODELS = {
    "cnn":           CNNRegressor,
    "resnet":        ResNetRegressor,
    "cnn_v2":        CNNRegressorV2,
    "cnn_v3":        CNNRegressorV3,
    "cnn_v4":        CNNRegressorV4,
    "mlp":           MLPRegressor,
    "physics":       PhysicsInformedRegressor,
    "peak_mlp":      lambda: MLPRegressor(input_dim=41),
    "peak_mlp_multi": None,    # input_dim set dynamically from element_names
    "peak_mlp_full":  None,    # 129 abs + 129 ratio + 41 clf probs = 299-dim
}

PHYSICS_MODELS    = {"physics"}
PEAK_ONLY_MODELS  = {"peak_mlp"}
PEAK_MULTI_MODELS = {"peak_mlp_multi", "peak_mlp_full"}
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
    element_names: list = None,
    seed: int = 42,
    workers: int = 12,
    n_train: int = None,
    lr: float = 1e-3,
    epochs: int = 60,
    batch_size: int = 64,
    patience: int = 10,
    scaler: str = "log_minmax",
    loss_fn: str = "mse_kl",
    lr_scheduler: bool = False,
    mse_weight: float = 1.0,
    kl_weight: float = 0.5,
):
    if model_name in PHYSICS_MODELS and element_names is not None:
        X_train = augment_with_peak_features(X_train, element_names)
        X_val   = augment_with_peak_features(X_val,   element_names)
    elif model_name in PEAK_ONLY_MODELS and element_names is not None:
        X_train = extract_peak_features(X_train, element_names)
        X_val   = extract_peak_features(X_val,   element_names)
    elif model_name in PEAK_MULTI_MODELS and element_names is not None:
        include_ratios = (model_name == "peak_mlp_full")
        X_peak_train = extract_multi_line_features(X_train, element_names,
                                                    include_ratios=include_ratios)
        X_peak_val   = extract_multi_line_features(X_val,   element_names,
                                                    include_ratios=include_ratios)
        if model_name == "peak_mlp_full":
            from scripts.regression.classifier_cache import load_or_train, predict_proba, CLF_DIR
            # Pick the right classifier preset based on what data is being trained on.
            # thin_window data → use thin_window_high_quality classifier if available,
            # otherwise fall back to standard high_quality classifier.
            is_thin = any(k in (element_names or []) for k in []) or False  # placeholder
            clf_preset_candidates = ["thin_window_high_quality", "high_quality"]
            clf_trainer = None
            for clf_preset in clf_preset_candidates:
                for clf_seed in [seed, 123, 42]:
                    clf_pt = CLF_DIR / f"{clf_preset}_n{n_train}_s{clf_seed}" / "clf_model.pt"
                    if clf_pt.exists():
                        _, clf_trainer = load_or_train(clf_preset, n_train, clf_seed, workers)
                        break
                if clf_trainer is not None:
                    break
            if clf_trainer is None:
                raise FileNotFoundError(
                    "No cached classifier found. Run: python -m scripts.regression.cli train-classifier"
                )
            X_train = np.hstack([X_peak_train, predict_proba(clf_trainer, X_train)])
            X_val   = np.hstack([X_peak_val,   predict_proba(clf_trainer, X_val)])
        else:
            X_train, X_val = X_peak_train, X_peak_val

    if model_name in PEAK_MULTI_MODELS:
        n_feats = X_train.shape[1]
        model = MLPRegressor(input_dim=n_feats)
    else:
        model = DL_MODELS[model_name]()
    t = RegressionTrainer(
        model,
        learning_rate=lr,
        mse_weight=mse_weight,
        kl_weight=kl_weight,
        loss_fn=loss_fn,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
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
