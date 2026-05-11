"""Train and cache CNNClassifier for use as feature extractor in regression."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLF_DIR = PROJECT_ROOT / "data" / "classifiers"
sys.path.insert(0, str(PROJECT_ROOT))


def _clf_path(preset: str, n_train: int, seed: int) -> Path:
    return CLF_DIR / f"{preset}_n{n_train}_s{seed}"


def find_classifier(n_train: int, prefer_preset: str = None) -> tuple:
    """
    Find any available cached classifier, optionally preferring a specific preset.
    Returns (preset, seed, path) or raises FileNotFoundError.
    """
    if CLF_DIR.exists():
        candidates = sorted(CLF_DIR.iterdir())
        # Prefer matching preset if requested
        if prefer_preset:
            for p in candidates:
                if p.name.startswith(prefer_preset) and (p / "clf_model.pt").exists():
                    parts = p.name.rsplit("_s", 1)
                    seed = int(parts[1])
                    preset = parts[0].replace(f"_n{n_train}", "").rstrip("_")
                    return preset, seed, p
        # Fall back to any available
        for p in candidates:
            if (p / "clf_model.pt").exists():
                return None, None, p
    raise FileNotFoundError("No cached classifier found. Run train-classifier first.")


def train_and_save(
    preset: str,
    n_train: int,
    n_val: int,
    seed: int,
    workers: int,
    save_path: Path = None,
) -> tuple:
    from src.models.classification.architectures import CNNClassifier
    from src.models.classification.trainer import ClassificationTrainer
    from scripts.regression.data import load_or_generate

    if save_path is None:
        save_path = _clf_path(preset, n_train, seed)
    save_path.mkdir(parents=True, exist_ok=True)

    d = load_or_generate(preset, n_train, n_val, n_val, seed, workers)
    y_clf_train = (d["y_train"] > 0).astype(np.float32)
    y_clf_val   = (d["y_val"]   > 0).astype(np.float32)

    clf = CNNClassifier(input_dim=600, num_classes=41)
    trainer = ClassificationTrainer(clf, learning_rate=1e-3)
    print("[classifier] Training CNNClassifier...")
    trainer.train(
        d["X_train"], y_clf_train,
        d["X_val"],   y_clf_val,
        epochs=60, batch_size=64, patience=10,
    )

    torch.save(trainer.model.state_dict(), save_path / "clf_model.pt")
    meta = {"preset": preset, "n_train": n_train, "n_val": n_val, "seed": seed}
    (save_path / "clf_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[classifier] Saved to {save_path.relative_to(PROJECT_ROOT)}")
    return clf, trainer


def load_or_train(
    preset: str,
    n_train: int,
    seed: int,
    workers: int = 12,
    n_val: int = 4000,
    n_test: int = 4000,
) -> tuple:
    from src.models.classification.architectures import CNNClassifier
    from src.models.classification.trainer import ClassificationTrainer

    save_path = _clf_path(preset, n_train, seed)
    model_file = save_path / "clf_model.pt"

    clf = CNNClassifier(input_dim=600, num_classes=41)

    if model_file.exists():
        print(f"[classifier] Loading cached model from {save_path.relative_to(PROJECT_ROOT)}")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        clf.load_state_dict(torch.load(model_file, map_location=device))
        trainer = ClassificationTrainer(clf, learning_rate=1e-3)
    else:
        clf, trainer = train_and_save(preset, n_train, n_val, seed, workers, save_path)

    trainer.model.eval()
    return clf, trainer


def predict_proba(trainer, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Return raw sigmoid probabilities (N, 41) — not thresholded."""
    from torch.utils.data import DataLoader, TensorDataset
    device = trainer.device
    loader = DataLoader(TensorDataset(torch.FloatTensor(X)),
                        batch_size=batch_size, shuffle=False)
    probs = []
    trainer.model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            probs.append(trainer.model(batch.to(device)).cpu().numpy())
    return np.concatenate(probs, axis=0)
