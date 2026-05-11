"""train command: train a single model and save results."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.regression import data, plots, runs
from scripts.regression.trainer import (
    DL_MODELS, BASELINE_MODELS, PHYSICS_MODELS, PEAK_ONLY_MODELS, PEAK_MULTI_MODELS,
    train_dl, train_baseline,
)


def run(args: argparse.Namespace):
    d = data.load_or_generate(
        args.config, args.n_train, args.n_val, args.n_test,
        args.seed, args.workers, args.force_data,
    )
    X_train, y_train = d["X_train"], d["y_train"]
    X_val,   y_val   = d["X_val"],   d["y_val"]
    X_test,  y_test  = d["X_test"],  d["y_test"]
    element_names    = d["element_names"]

    run_dir = runs.make_run_dir(args.model, args.run_name)
    runs.save_config(run_dir, vars(args))

    if args.model in DL_MODELS:
        from src.models.regression import augment_with_peak_features, extract_peak_features
        from src.models.regression.peak_features import extract_multi_line_features

        trainer, history = train_dl(
            args.model, X_train, y_train, X_val, y_val,
            element_names=element_names,
            seed=args.seed,
            workers=args.workers,
            n_train=args.n_train,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            scaler=args.scaler,
            loss_fn=args.loss_fn,
            lr_scheduler=args.lr_scheduler,
        )
        if args.model in PHYSICS_MODELS:
            X_test_model = augment_with_peak_features(X_test, element_names)
        elif args.model in PEAK_ONLY_MODELS:
            X_test_model = extract_peak_features(X_test, element_names)
        elif args.model in PEAK_MULTI_MODELS:
            include_ratios = (args.model == "peak_mlp_full")
            X_test_peak = extract_multi_line_features(X_test, element_names,
                                                       include_ratios=include_ratios)
            if args.model == "peak_mlp_full":
                from scripts.regression.classifier_cache import load_or_train, predict_proba
                from pathlib import Path as _Path
                _clf_root = _Path(__file__).resolve().parent.parent.parent / "data" / "classifiers"
                clf_seed = args.seed
                if not (_clf_root / f"high_quality_n{args.n_train}_s{args.seed}" / "clf_model.pt").exists():
                    clf_seed = 123
                _, clf_trainer = load_or_train("high_quality", args.n_train,
                                               clf_seed, args.workers)
                X_test_model = np.hstack([X_test_peak, predict_proba(clf_trainer, X_test)])
            else:
                X_test_model = X_test_peak
        else:
            X_test_model = X_test
        metrics = trainer.evaluate(X_test_model, y_test, element_names=element_names)
        runs.print_metrics(metrics, label=args.model)
        runs.save_metrics(run_dir, metrics)
        runs.save_dl_model(run_dir, trainer)

        plots.training_curve(history, f"{args.model} — Training Curve",
                             run_dir / "plots" / "training_curve.png")
        preds = trainer.predict(X_test_model)
        plots.predictions(y_test, preds, X_test, element_names,
                          f"{args.model} — Predictions",
                          run_dir / "plots" / "predictions.png")
        plots.per_element_mae({args.model: metrics},
                              run_dir / "plots" / "per_element_mae.png")

    elif args.model in BASELINE_MODELS:
        baseline = train_baseline(args.model, X_train, y_train, element_names)
        metrics = baseline.evaluate(X_test, y_test)
        runs.print_metrics(metrics, label=args.model)
        runs.save_metrics(run_dir, metrics)
        runs.save_sklearn_model(run_dir, baseline)

    print(f"\n[done] Results saved to {run_dir.relative_to(PROJECT_ROOT)}")


def add_args(parser: argparse.ArgumentParser):
    all_models = sorted(set(list(DL_MODELS) + list(BASELINE_MODELS) + list(PEAK_MULTI_MODELS)))
    parser.add_argument("--model",       required=True, choices=all_models)
    parser.add_argument("--config",      default="regression",
                        choices=data.VALID_PRESETS)
    parser.add_argument("--n-train",     type=int, default=5000)
    parser.add_argument("--n-val",       type=int, default=400)
    parser.add_argument("--n-test",      type=int, default=400)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--workers",     type=int, default=12)
    parser.add_argument("--force-data",  action="store_true")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--scaler",      default="log_minmax",
                        choices=["standard", "log_minmax"])
    parser.add_argument("--loss-fn",     default="mse_kl",
                        choices=["mse_kl", "mae", "dirichlet"])
    parser.add_argument("--lr-scheduler", action="store_true",
                        help="Use ReduceLROnPlateau scheduler")
    parser.add_argument("--run-name",    default=None)
