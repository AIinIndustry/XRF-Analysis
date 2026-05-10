"""pipeline-compare command: equivalent to notebook 5 (raw vs denoised)."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.denoising.generator import DenoisingDataGenerator
from src.data.common.base_generator import GeneratorConfig
from src.models.denoising.architectures import CNNAutoencoder
from src.models.denoising.trainer import DenoisingTrainer

from scripts.regression import data, plots, runs
from scripts.regression.trainer import train_dl


def run(args: argparse.Namespace):
    run_dir = runs.make_run_dir("pipeline_compare", args.run_name)
    runs.save_config(run_dir, vars(args))

    np.random.seed(args.seed)

    # --- Clean regressor (for denoised pipeline) ---
    clean_data = data.load_or_generate(
        "high_quality", args.n_train, args.n_val, args.n_test,
        args.seed, args.workers,
    )
    trainer_clean, hist_clean = train_dl(
        "cnn_v2",
        clean_data["X_train"], clean_data["y_train"],
        clean_data["X_val"],   clean_data["y_val"],
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, scaler=args.scaler,
    )
    plots.training_curve(hist_clean, "Clean Regressor (CNN V2)",
                         run_dir / "plots" / "training_clean.png")

    # --- Noisy regressor (raw pipeline) ---
    noisy_data = data.load_or_generate(
        "fast_scan", args.n_train, args.n_val, args.n_test,
        args.seed, args.workers,
    )
    trainer_noisy, hist_noisy = train_dl(
        "cnn_v2",
        noisy_data["X_train"], noisy_data["y_train"],
        noisy_data["X_val"],   noisy_data["y_val"],
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, scaler=args.scaler,
    )
    plots.training_curve(hist_noisy, "Raw Regressor (CNN V2)",
                         run_dir / "plots" / "training_noisy.png")

    # --- Denoiser ---
    den_gen = DenoisingDataGenerator(seed=args.seed)
    den_cfg = GeneratorConfig.Presets.fast_scan()
    print("[train] Generating denoising pairs...")
    X_den_tr_n, X_den_tr_c = den_gen.generate_dataset(args.n_train, config=den_cfg, num_workers=args.workers)
    X_den_va_n, X_den_va_c = den_gen.generate_dataset(args.n_val,   config=den_cfg, num_workers=args.workers)

    denoiser = CNNAutoencoder(input_dim=600)
    den_trainer = DenoisingTrainer(denoiser, learning_rate=args.lr)
    print("[train] Training denoiser...")
    den_hist = den_trainer.train(
        X_den_tr_n, X_den_tr_c, X_den_va_n, X_den_va_c,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
    )
    plots.training_curve(den_hist, "Denoiser", run_dir / "plots" / "training_denoiser.png")

    # --- Evaluate ---
    X_test_noisy = noisy_data["X_test"]
    y_test        = noisy_data["y_test"]
    element_names = noisy_data["element_names"]

    X_test_denoised = den_trainer.predict(X_test_noisy)

    metrics_raw      = trainer_noisy.evaluate(X_test_noisy,    y_test, element_names=element_names)
    metrics_denoised = trainer_clean.evaluate(X_test_denoised, y_test, element_names=element_names)

    pipeline_results = {"Raw": metrics_raw, "Denoised": metrics_denoised}
    for label, m in pipeline_results.items():
        runs.print_metrics(m, label=label)

    runs.save_metrics(run_dir, {
        "raw":      {k: v for k, v in metrics_raw.items()      if k != "per_element_mae"},
        "denoised": {k: v for k, v in metrics_denoised.items() if k != "per_element_mae"},
    })
    plots.pipeline_comparison_bars(pipeline_results, run_dir / "plots" / "pipeline_comparison.png")
    plots.per_element_mae(pipeline_results, run_dir / "plots" / "per_element_mae.png")

    preds_raw      = trainer_noisy.predict(X_test_noisy)
    preds_denoised = trainer_clean.predict(X_test_denoised)
    plots.predictions(y_test, preds_raw,      X_test_noisy,    element_names,
                      "Raw Pipeline",      run_dir / "plots" / "predictions_raw.png")
    plots.predictions(y_test, preds_denoised, X_test_denoised, element_names,
                      "Denoised Pipeline", run_dir / "plots" / "predictions_denoised.png")

    # --- Noise ablation ---
    if args.noise_ablation:
        print("\n[ablation] Running noise level sweep...")
        from src.data.regression.generator import RegressionDataGenerator
        reg_gen = RegressionDataGenerator(seed=args.seed)
        noise_levels = [500, 1000, 3000, 5000, 10000, 20000, 30000]
        mae_raw, mae_den = [], []

        for n in noise_levels:
            cfg = GeneratorConfig.Presets.fast_scan()
            cfg.n_counts_range = (n, n)
            X_n, y_n = reg_gen.generate_dataset(200, min_elements=2, max_elements=5,
                                                 config=cfg, num_workers=args.workers)
            X_n_den = den_trainer.predict(X_n)
            mae_raw.append(trainer_noisy.evaluate(X_n,     y_n.values)["masked_mae"])
            mae_den.append(trainer_clean.evaluate(X_n_den, y_n.values)["masked_mae"])
            print(f"  n={n:6d}  raw={mae_raw[-1]:.4f}  denoised={mae_den[-1]:.4f}")

        plots.noise_ablation(
            noise_levels,
            {"Raw": mae_raw, "Denoised": mae_den},
            run_dir / "plots" / "noise_ablation.png",
        )
        runs.save_metrics(run_dir, {
            "noise_ablation": {"noise_levels": noise_levels, "raw": mae_raw, "denoised": mae_den}
        })

    print(f"\n[done] Results saved to {run_dir.relative_to(PROJECT_ROOT)}")


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--n-train",       type=int, default=2000)
    parser.add_argument("--n-val",         type=int, default=400)
    parser.add_argument("--n-test",        type=int, default=400)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--workers",       type=int, default=12)
    parser.add_argument("--force-data",    action="store_true")
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--scaler",        default="log_minmax",
                        choices=["standard", "log_minmax"])
    parser.add_argument("--noise-ablation", action="store_true",
                        help="Run noise level ablation sweep")
    parser.add_argument("--run-name",      default=None)
