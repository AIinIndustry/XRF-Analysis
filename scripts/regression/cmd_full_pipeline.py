"""full-pipeline command: equivalent to notebook 6 (3-stage + ablation)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.denoising.generator import DenoisingDataGenerator
from src.data.common.base_generator import GeneratorConfig
from src.models.denoising.architectures import CNNAutoencoder
from src.models.denoising.trainer import DenoisingTrainer
from src.models.classification.architectures import CNNClassifier
from src.models.classification.trainer import ClassificationTrainer
from src.models.regression.architectures import TwoStageRegressor
from src.models.regression import evaluate_all

from scripts.regression import data, plots, runs
from scripts.regression.trainer import train_dl


def _predict_two_stage(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    device = next(model.parameters()).device
    preds = []
    for i in range(0, len(X), batch_size):
        batch = torch.FloatTensor(X[i : i + batch_size]).to(device)
        with torch.no_grad():
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0)


def run(args: argparse.Namespace):
    run_dir = runs.make_run_dir("full_pipeline", args.run_name)
    runs.save_config(run_dir, vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Data ---
    clean_data = data.load_or_generate(
        "high_quality", args.n_train, args.n_val, args.n_test,
        args.seed, args.workers,
    )
    noisy_data = data.load_or_generate(
        "fast_scan", args.n_train, args.n_val, args.n_test,
        args.seed, args.workers,
    )

    element_names    = clean_data["element_names"]
    X_clean_train    = clean_data["X_train"]
    y_clean_train    = clean_data["y_train"]
    X_clean_val      = clean_data["X_val"]
    y_clean_val      = clean_data["y_val"]

    # Binary labels for classifier
    y_clf_train = (y_clean_train > 0).astype(np.float32)
    y_clf_val   = (y_clean_val   > 0).astype(np.float32)

    # --- Classifier ---
    print("[train] Training classifier...")
    clf = CNNClassifier(input_dim=600, num_classes=41)
    clf_trainer = ClassificationTrainer(clf, learning_rate=args.lr)
    clf_hist = clf_trainer.train(
        X_clean_train, y_clf_train, X_clean_val, y_clf_val,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
    )
    plots.training_curve(clf_hist, "Classifier", run_dir / "plots" / "training_clf.png")
    clf_metrics = clf_trainer.evaluate_metrics(X_clean_val, y_clf_val)
    print(f"[clf]  F1={clf_metrics.get('F1-Score', 'n/a'):.4f}  "
          f"Precision={clf_metrics.get('Precision', 'n/a'):.4f}  "
          f"Recall={clf_metrics.get('Recall', 'n/a'):.4f}")

    # --- Denoiser ---
    den_gen = DenoisingDataGenerator(seed=args.seed)
    den_cfg = GeneratorConfig.Presets.fast_scan()
    print("[train] Generating denoising pairs...")
    X_den_tr_n, X_den_tr_c = den_gen.generate_dataset(args.n_train, config=den_cfg, num_workers=args.workers)
    X_den_va_n, X_den_va_c = den_gen.generate_dataset(args.n_val,   config=den_cfg, num_workers=args.workers)
    denoiser    = CNNAutoencoder(input_dim=600)
    den_trainer = DenoisingTrainer(denoiser, learning_rate=args.lr)
    print("[train] Training denoiser...")
    den_hist = den_trainer.train(
        X_den_tr_n, X_den_tr_c, X_den_va_n, X_den_va_c,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
    )
    plots.training_curve(den_hist, "Denoiser", run_dir / "plots" / "training_denoiser.png")

    # --- Regressors ---
    trainer_clean, hist_clean = train_dl(
        "cnn_v2", X_clean_train, y_clean_train, X_clean_val, y_clean_val,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, scaler=args.scaler,
    )
    plots.training_curve(hist_clean, "Clean Regressor", run_dir / "plots" / "training_clean.png")

    trainer_noisy, hist_noisy = train_dl(
        "cnn_v2", noisy_data["X_train"], noisy_data["y_train"],
        noisy_data["X_val"], noisy_data["y_val"],
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, scaler=args.scaler,
    )
    plots.training_curve(hist_noisy, "Noisy Regressor", run_dir / "plots" / "training_noisy.png")

    # --- Assemble complete pipeline ---
    two_stage = TwoStageRegressor(
        classifier=clf_trainer.model,
        regressor=trainer_clean.model,
        soft_mask=True,
    )
    two_stage.eval()

    # --- Test set ---
    X_test_noisy  = noisy_data["X_test"]
    y_test        = noisy_data["y_test"]
    X_test_den    = den_trainer.predict(X_test_noisy)

    metrics_raw      = trainer_noisy.evaluate(X_test_noisy, y_test, element_names=element_names)
    metrics_denoised = trainer_clean.evaluate(X_test_den,   y_test, element_names=element_names)
    preds_complete   = _predict_two_stage(two_stage, X_test_den)
    metrics_complete = evaluate_all(y_test, preds_complete, element_names=element_names)

    pipeline_results = {
        "Raw":      metrics_raw,
        "Denoised": metrics_denoised,
        "Complete": metrics_complete,
    }
    for label, m in pipeline_results.items():
        runs.print_metrics(m, label=label)

    runs.save_metrics(run_dir, {
        k: {mk: mv for mk, mv in m.items() if mk != "per_element_mae"}
        for k, m in pipeline_results.items()
    })
    plots.pipeline_comparison_bars(pipeline_results, run_dir / "plots" / "pipeline_comparison.png")
    plots.per_element_mae(pipeline_results, run_dir / "plots" / "per_element_mae.png")
    plots.predictions(y_test, preds_complete, X_test_den, element_names,
                      "Complete Pipeline", run_dir / "plots" / "predictions_complete.png")

    # --- Noise ablation ---
    if args.noise_ablation:
        print("\n[ablation] Running noise level sweep...")
        from src.data.regression.generator import RegressionDataGenerator
        reg_gen = RegressionDataGenerator(seed=args.seed)
        noise_levels = [500, 1000, 3000, 5000, 10000, 20000, 30000]
        mae_raw, mae_den, mae_cpl = [], [], []

        for n in noise_levels:
            cfg = GeneratorConfig.Presets.fast_scan()
            cfg.n_counts_range = (n, n)
            X_n, y_n = reg_gen.generate_dataset(200, min_elements=2, max_elements=5,
                                                 config=cfg, num_workers=args.workers)
            X_n_den = den_trainer.predict(X_n)
            mae_raw.append(trainer_noisy.evaluate(X_n,     y_n.values)["masked_mae"])
            mae_den.append(trainer_clean.evaluate(X_n_den, y_n.values)["masked_mae"])
            p_cpl = _predict_two_stage(two_stage, X_n_den)
            mae_cpl.append(evaluate_all(y_n.values, p_cpl)["masked_mae"])
            print(f"  n={n:6d}  raw={mae_raw[-1]:.4f}  den={mae_den[-1]:.4f}  cpl={mae_cpl[-1]:.4f}")

        plots.noise_ablation(
            noise_levels,
            {"Raw": mae_raw, "Denoised": mae_den, "Complete": mae_cpl},
            run_dir / "plots" / "noise_ablation.png",
        )

    print(f"\n[done] Results saved to {run_dir.relative_to(PROJECT_ROOT)}")


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--n-train",        type=int, default=2000)
    parser.add_argument("--n-val",          type=int, default=400)
    parser.add_argument("--n-test",         type=int, default=400)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--workers",        type=int, default=12)
    parser.add_argument("--force-data",     action="store_true")
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--epochs",         type=int,   default=60)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--scaler",         default="log_minmax",
                        choices=["standard", "log_minmax"])
    parser.add_argument("--noise-ablation", action="store_true")
    parser.add_argument("--run-name",       default=None)
