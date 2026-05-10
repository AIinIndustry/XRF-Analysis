"""Data generation and caching for regression experiments."""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "regression"

VALID_PRESETS = ("regression", "high_quality", "fast_scan", "balanced", "robust_training")


def _data_path(preset: str, n_train: int, n_val: int, n_test: int, seed: int) -> Path:
    return DATA_DIR / f"{preset}_n{n_train}-{n_val}-{n_test}_s{seed}"


def generate(
    preset: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    workers: int,
    force: bool = False,
) -> Path:
    """Generate and cache a regression dataset. Returns the data directory path."""
    if preset not in VALID_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Valid: {VALID_PRESETS}")

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.regression.generator import RegressionDataGenerator
    from src.data.common.base_generator import GeneratorConfig

    path = _data_path(preset, n_train, n_val, n_test, seed)

    if path.exists() and not force:
        print(f"[data] Using cached data at {path.relative_to(PROJECT_ROOT)}")
        return path

    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    gen = RegressionDataGenerator(seed=seed)
    config = getattr(GeneratorConfig.Presets, preset)()

    element_names = None
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        print(f"[data] Generating {split} split ({n} samples, preset={preset})...")
        X, y = gen.generate_dataset(
            n, min_elements=2, max_elements=5, config=config, num_workers=workers
        )
        np.save(path / f"X_{split}.npy", X)
        np.save(path / f"y_{split}.npy", y.values)
        if element_names is None:
            element_names = y.columns.tolist()

    meta = {
        "preset": preset,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
        "element_names": element_names,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[data] Saved to {path.relative_to(PROJECT_ROOT)}")
    return path


def load(preset: str, n_train: int, n_val: int, n_test: int, seed: int) -> dict:
    """Load a cached dataset. Raises if not found."""
    path = _data_path(preset, n_train, n_val, n_test, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached data at {path.relative_to(PROJECT_ROOT)}.\n"
            f"Run: python -m scripts.regression.cli generate --config {preset} "
            f"--n-train {n_train} --n-val {n_val} --n-test {n_test} --seed {seed}"
        )
    meta = json.loads((path / "meta.json").read_text())
    return {
        "X_train": np.load(path / "X_train.npy"),
        "y_train": np.load(path / "y_train.npy"),
        "X_val":   np.load(path / "X_val.npy"),
        "y_val":   np.load(path / "y_val.npy"),
        "X_test":  np.load(path / "X_test.npy"),
        "y_test":  np.load(path / "y_test.npy"),
        "element_names": meta["element_names"],
    }


def load_or_generate(
    preset: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    workers: int,
    force: bool = False,
) -> dict:
    """Load cached data if available, otherwise generate it first."""
    generate(preset, n_train, n_val, n_test, seed, workers, force)
    return load(preset, n_train, n_val, n_test, seed)
