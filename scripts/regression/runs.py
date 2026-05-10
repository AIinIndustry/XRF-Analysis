"""Run directory management: saving models, metrics, and configs."""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "regression"

sys.path.insert(0, str(PROJECT_ROOT))


def make_run_dir(label: str, run_name: Optional[str] = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{label}" + (f"_{run_name}" if run_name else "")
    path = RUNS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path


def save_config(run_dir: Path, config: dict):
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))


def save_metrics(run_dir: Path, metrics: dict):
    serialisable = {}
    for k, v in metrics.items():
        if hasattr(v, "to_dict"):          # pd.Series
            serialisable[k] = v.to_dict()
        elif isinstance(v, np.floating):
            serialisable[k] = float(v)
        else:
            serialisable[k] = v
    (run_dir / "metrics.json").write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"[runs] Metrics saved to {run_dir.relative_to(PROJECT_ROOT)}/metrics.json")


def save_dl_model(run_dir: Path, trainer):
    """Save a RegressionTrainer's model weights and fitted scaler."""
    import torch
    torch.save(trainer.model.state_dict(), run_dir / "model.pt")
    if trainer.scaler is not None:
        with open(run_dir / "scaler.pkl", "wb") as f:
            pickle.dump(trainer.scaler, f)


def save_sklearn_model(run_dir: Path, baseline):
    """Save a PLSBaseline or RidgeBaseline via joblib."""
    import joblib
    joblib.dump(baseline, run_dir / "model.joblib")


def print_metrics(metrics: dict, label: str = ""):
    header = f"  {label}" if label else ""
    print(f"\n{'='*50}{header}")
    for k, v in metrics.items():
        if k == "per_element_mae":
            continue
        print(f"  {k:20s}: {v:.4f}")
    if "per_element_mae" in metrics and metrics["per_element_mae"] is not None:
        top5 = metrics["per_element_mae"].dropna().nlargest(5)
        print(f"  {'top5_hard_elements':20s}: " +
              ", ".join(f"{el}={v:.3f}" for el, v in top5.items()))
    print("=" * 50)
