"""compare-models command: equivalent to notebook 4."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.regression import data, plots, runs
from scripts.regression.trainer import DL_MODELS, BASELINE_MODELS, train_dl, train_baseline


def run(args: argparse.Namespace):
    d = data.load_or_generate(
        args.config, args.n_train, args.n_val, args.n_test,
        args.seed, args.workers, args.force_data,
    )
    X_train, y_train = d["X_train"], d["y_train"]
    X_val,   y_val   = d["X_val"],   d["y_val"]
    X_test,  y_test  = d["X_test"],  d["y_test"]
    element_names    = d["element_names"]

    run_dir = runs.make_run_dir("compare_models", args.run_name)
    runs.save_config(run_dir, vars(args))

    results = {}
    histories = {}

    models_to_run = args.models if args.models else list(DL_MODELS) + list(BASELINE_MODELS)

    # --- DL models ---
    for name in models_to_run:
        if name not in DL_MODELS:
            continue
        trainer, history = train_dl(
            name, X_train, y_train, X_val, y_val,
            lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, scaler=args.scaler,
        )
        metrics = trainer.evaluate(X_test, y_test, element_names=element_names)
        results[name] = metrics
        histories[name] = history
        runs.print_metrics(metrics, label=name)

        plots.training_curve(
            history, f"{name} — Training Curve",
            run_dir / "plots" / f"training_curve_{name}.png",
        )
        preds = trainer.predict(X_test)
        plots.predictions(
            y_test, preds, X_test, element_names,
            f"{name} — Predictions",
            run_dir / "plots" / f"predictions_{name}.png",
        )
        runs.save_dl_model(run_dir / "plots" / ".." / f"model_{name}", trainer)
        (run_dir / f"model_{name}").mkdir(exist_ok=True)
        runs.save_dl_model(run_dir / f"model_{name}", trainer)
        runs.save_metrics(run_dir / f"model_{name}", metrics)

    # --- Baselines ---
    for name in models_to_run:
        if name not in BASELINE_MODELS:
            continue
        baseline = train_baseline(name, X_train, y_train, element_names)
        metrics = baseline.evaluate(X_test, y_test)
        results[name] = metrics
        runs.print_metrics(metrics, label=name)
        runs.save_sklearn_model(run_dir / f"model_{name}", baseline)
        (run_dir / f"model_{name}").mkdir(exist_ok=True)
        runs.save_sklearn_model(run_dir / f"model_{name}", baseline)
        runs.save_metrics(run_dir / f"model_{name}", metrics)

    # --- Summary ---
    summary = pd.DataFrame({
        name: {k: v for k, v in m.items() if k != "per_element_mae"}
        for name, m in results.items()
    }).T

    print("\n=== Model Comparison ===")
    print(summary.to_string(float_format="{:.4f}".format))

    runs.save_metrics(run_dir, {"summary": summary.to_dict()})
    plots.model_comparison(summary, run_dir / "plots" / "comparison.png")
    plots.per_element_mae(results, run_dir / "plots" / "per_element_mae.png")

    print(f"\n[done] Results saved to {run_dir.relative_to(PROJECT_ROOT)}")


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config",      default="regression",
                        choices=data.VALID_PRESETS)
    parser.add_argument("--n-train",     type=int, default=2000)
    parser.add_argument("--n-val",       type=int, default=400)
    parser.add_argument("--n-test",      type=int, default=400)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--workers",     type=int, default=12)
    parser.add_argument("--force-data",  action="store_true",
                        help="Regenerate data even if cached")
    parser.add_argument("--models",      nargs="+",
                        choices=list(DL_MODELS) + list(BASELINE_MODELS),
                        help="Subset of models to train (default: all)")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--scaler",      default="log_minmax",
                        choices=["standard", "log_minmax"])
    parser.add_argument("--run-name",    default=None,
                        help="Optional suffix for the run directory")
