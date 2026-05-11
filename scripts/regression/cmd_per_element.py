"""per-element and per-element-cnn commands."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.regression import data, plots, runs
from src.models.regression.per_element import (
    PerElementRidgeRegressor, PerElementCNNRegressor,
    SharedPerElementCNN, ElementTransformer, ElementTransformerV2,
    build_neighbor_map,
    train_per_element_cnn, predict_per_element_cnn,
    train_generic, predict_generic, apply_hard_mask,
)
from src.models.regression.peak_features import extract_multi_line_features
from src.models.regression.metrics import evaluate_all as _eval_all


def run(args: argparse.Namespace):
    d = data.load_or_generate(
        args.config, args.n_train, args.n_val, args.n_test,
        args.seed, args.workers, args.force_data,
    )
    X_train, y_train = d["X_train"], d["y_train"]
    X_test,  y_test  = d["X_test"],  d["y_test"]
    element_names    = d["element_names"]

    clf_probs_train = clf_probs_test = None
    if args.use_classifier:
        from scripts.regression.classifier_cache import load_or_train, predict_proba, CLF_DIR
        clf_trainer = None
        for clf_preset in ["thin_window_high_quality", "high_quality"]:
            for clf_seed in [args.seed, 123, 42]:
                p = CLF_DIR / f"{clf_preset}_n{args.n_train}_s{clf_seed}" / "clf_model.pt"
                if p.exists():
                    _, clf_trainer = load_or_train(clf_preset, args.n_train, clf_seed)
                    break
            if clf_trainer:
                break
        if clf_trainer is None:
            print("[warn] No classifier found, running without classifier features.")
        else:
            clf_probs_train = predict_proba(clf_trainer, X_train)
            clf_probs_test  = predict_proba(clf_trainer, X_test)
            print(f"[clf] Using classifier probabilities as features.")

    run_dir = runs.make_run_dir("per_element", args.run_name)
    runs.save_config(run_dir, vars(args))

    print(f"[train] Fitting PerElementRidgeRegressor (alpha={args.alpha}, "
          f"overlap_radius={args.overlap_radius})...")
    model = PerElementRidgeRegressor(alpha=args.alpha, overlap_radius=args.overlap_radius)
    model.fit(X_train, y_train, element_names, clf_probs=clf_probs_train)
    print("[train] Done.")

    preds_test = model.predict(X_test, clf_probs=clf_probs_test)
    if args.hard_mask > 0 and clf_probs_test is not None:
        preds_test = apply_hard_mask(preds_test, clf_probs_test, threshold=args.hard_mask)
    metrics = evaluate_all(preds_test, y_test, element_names=element_names)
    runs.print_metrics(metrics, label="per_element_ridge")
    runs.save_metrics(run_dir, metrics)

    plots.predictions(y_test, preds_test, X_test, element_names,
                      "Per-Element Ridge", run_dir / "plots" / "predictions.png")
    plots.per_element_mae({"PerElementRidge": metrics},
                          run_dir / "plots" / "per_element_mae.png")

    print(f"\n[done] Results saved to {run_dir.relative_to(PROJECT_ROOT)}")


from src.models.regression.metrics import evaluate_all


def run_cnn(args: argparse.Namespace):
    d = data.load_or_generate(
        args.config, args.n_train, args.n_val, args.n_test,
        args.seed, args.workers, args.force_data,
    )
    X_train, y_train = d["X_train"], d["y_train"]
    X_val,   y_val   = d["X_val"],   d["y_val"]
    X_test,  y_test  = d["X_test"],  d["y_test"]
    element_names    = d["element_names"]

    clf_probs_train = clf_probs_val = clf_probs_test = None
    if args.use_classifier:
        from scripts.regression.classifier_cache import load_or_train, predict_proba, CLF_DIR
        clf_trainer = None
        for clf_preset in ["thin_window_high_quality", "high_quality"]:
            for clf_seed in [args.seed, 123, 42]:
                p = CLF_DIR / f"{clf_preset}_n{args.n_train}_s{clf_seed}" / "clf_model.pt"
                if p.exists():
                    _, clf_trainer = load_or_train(clf_preset, args.n_train, clf_seed)
                    break
            if clf_trainer:
                break
        if clf_trainer:
            clf_probs_train = predict_proba(clf_trainer, X_train)
            clf_probs_val   = predict_proba(clf_trainer, X_val)
            clf_probs_test  = predict_proba(clf_trainer, X_test)
            print("[clf] Using classifier probabilities.")

    neighbor_map = build_neighbor_map(element_names, args.overlap_radius)

    run_dir = runs.make_run_dir("per_element_cnn", args.run_name)
    runs.save_config(run_dir, vars(args))

    model = PerElementCNNRegressor(
        element_names=element_names,
        neighbor_map=neighbor_map,
        patch_half=args.patch_half,
        max_patches=args.max_patches,
        use_clf=(clf_probs_train is not None),
    )
    print(f"[train] Training PerElementCNNRegressor "
          f"(patch_half={args.patch_half}, max_patches={args.max_patches})...")
    history = train_per_element_cnn(
        model, X_train, y_train, X_val, y_val,
        clf_probs_train=clf_probs_train,
        clf_probs_val=clf_probs_val,
        lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience,
    )

    preds   = predict_per_element_cnn(model, X_test, clf_probs_test)
    metrics = evaluate_all(preds, y_test, element_names=element_names)
    runs.print_metrics(metrics, label="per_element_cnn")
    runs.save_metrics(run_dir, metrics)

    plots.training_curve(history, "Per-Element CNN",
                         run_dir / "plots" / "training_curve.png")
    plots.predictions(y_test, preds, X_test, element_names,
                      "Per-Element CNN", run_dir / "plots" / "predictions.png")
    plots.per_element_mae({"PerElementCNN": metrics},
                          run_dir / "plots" / "per_element_mae.png")
    print(f"\n[done] Results saved to {run_dir.relative_to(PROJECT_ROOT)}")


def _load_data_and_clf(args):
    d = data.load_or_generate(
        args.config, args.n_train, args.n_val, args.n_test,
        args.seed, args.workers, getattr(args, "force_data", False),
    )
    clf_tr = clf_va = clf_te = None
    if args.use_classifier:
        from scripts.regression.classifier_cache import load_or_train, predict_proba as _pp, CLF_DIR
        clf_trainer = None
        for preset in ["thin_window_high_quality", "high_quality"]:
            for seed in [args.seed, 123, 42]:
                p = CLF_DIR / f"{preset}_n{args.n_train}_s{seed}" / "clf_model.pt"
                if p.exists():
                    _, clf_trainer = load_or_train(preset, args.n_train, seed)
                    break
            if clf_trainer: break
        if clf_trainer:
            clf_tr = _pp(clf_trainer, d["X_train"])
            clf_va = _pp(clf_trainer, d["X_val"])
            clf_te = _pp(clf_trainer, d["X_test"])
            print("[clf] Using classifier probabilities.")
    return d, clf_tr, clf_va, clf_te


def run_shared_cnn(args: argparse.Namespace):
    d, clf_tr, clf_va, clf_te = _load_data_and_clf(args)
    element_names = d["element_names"]
    neighbor_map  = build_neighbor_map(element_names, args.overlap_radius)
    run_dir = runs.make_run_dir("shared_element_cnn", args.run_name)
    runs.save_config(run_dir, vars(args))

    model = SharedPerElementCNN(
        element_names, neighbor_map,
        patch_half=args.patch_half, max_patches=args.max_patches,
        use_clf=(clf_tr is not None),
    )
    print("[train] Training SharedPerElementCNN...")
    history = train_generic(
        model, d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        clf_probs_train=clf_tr, clf_probs_val=clf_va,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, mixup_alpha=args.mixup_alpha,
    )
    preds = predict_generic(model, d["X_test"], clf_probs=clf_te)
    if args.hard_mask > 0 and clf_te is not None:
        preds = apply_hard_mask(preds, clf_te, threshold=args.hard_mask)
    metrics = evaluate_all(preds, d["y_test"], element_names=element_names)
    runs.print_metrics(metrics, label="shared_element_cnn")
    runs.save_metrics(run_dir, metrics)
    plots.training_curve(history, "Shared Per-Element CNN",
                         run_dir / "plots" / "training_curve.png")
    print(f"\n[done] Results saved to {run_dir.relative_to(PROJECT_ROOT)}")


def _make_transformer(args, element_names, use_clf, v2=False):
    cls = ElementTransformerV2 if v2 else ElementTransformer
    return cls(
        element_names,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        use_clf=use_clf,
    )


def run_transformer(args: argparse.Namespace, v2: bool = False):
    d, clf_tr, clf_va, clf_te = _load_data_and_clf(args)
    element_names = d["element_names"]
    label = "element_transformer_v2" if v2 else "element_transformer"
    run_dir = runs.make_run_dir(label, args.run_name)
    runs.save_config(run_dir, vars(args))

    use_clf = clf_tr is not None
    n_seeds = getattr(args, "n_seeds", 1)
    curriculum = getattr(args, "curriculum", False)

    # V2 needs combined (spectrum + peak features) input; V1 needs peak features only
    if v2:
        def transform_fn(X):
            return np.hstack([X, extract_multi_line_features(X, element_names)])
    else:
        def transform_fn(X):
            return extract_multi_line_features(X, element_names)

    X_tr = transform_fn(d["X_train"])
    X_va = transform_fn(d["X_val"])
    X_te = transform_fn(d["X_test"])

    all_preds = []
    for seed_offset in range(n_seeds):
        import torch as _torch
        _torch.manual_seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)
        model = _make_transformer(args, element_names, use_clf, v2=v2)
        name  = f"{label} seed={42+seed_offset}"
        print(f"[train] Training {name} ({seed_offset+1}/{n_seeds})...")
        history = train_generic(
            model, X_tr, d["y_train"], X_va, d["y_val"],
            clf_probs_train=clf_tr, clf_probs_val=clf_va,
            lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, mixup_alpha=args.mixup_alpha,
            curriculum=curriculum,
        )
        preds = predict_generic(model, X_te, clf_probs=clf_te)
        all_preds.append(preds)
        if n_seeds == 1:
            plots.training_curve(history, name,
                                 run_dir / "plots" / "training_curve.png")

    # Ensemble: average and renormalise
    final_preds = np.stack(all_preds, axis=0).mean(axis=0)
    final_preds = np.clip(final_preds, 0, None)
    final_preds /= final_preds.sum(axis=1, keepdims=True).clip(min=1e-8)

    if args.hard_mask > 0 and clf_te is not None:
        final_preds = apply_hard_mask(final_preds, clf_te, threshold=args.hard_mask)

    metrics = evaluate_all(final_preds, d["y_test"], element_names=element_names)
    runs.print_metrics(metrics, label=label + (f" ×{n_seeds}" if n_seeds > 1 else ""))
    runs.save_metrics(run_dir, metrics)
    print(f"\n[done] Results saved to {run_dir.relative_to(PROJECT_ROOT)}")


def _get_clf(args, n_train):
    """Helper: load cached classifier and return (clf_trainer, predict_proba) or None."""
    from scripts.regression.classifier_cache import load_or_train, predict_proba as _pp, CLF_DIR
    for preset in ["thin_window_high_quality", "high_quality"]:
        for seed in [getattr(args, "seed", 42), 123, 42]:
            p = CLF_DIR / f"{preset}_n{n_train}_s{seed}" / "clf_model.pt"
            if p.exists():
                _, t = load_or_train(preset, n_train, seed)
                print(f"[clf] Loaded {preset} seed={seed}")
                return t, _pp
    print("[warn] No cached classifier found.")
    return None, None


def _shared_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config",          default="regression",
                        choices=data.VALID_PRESETS)
    parser.add_argument("--n-train",         type=int, default=10000)
    parser.add_argument("--n-val",           type=int, default=4000)
    parser.add_argument("--n-test",          type=int, default=4000)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--workers",         type=int, default=12)
    parser.add_argument("--force-data",      action="store_true")
    parser.add_argument("--overlap-radius",  type=int, default=10)
    parser.add_argument("--use-classifier",  action="store_true")
    parser.add_argument("--hard-mask",       type=float, default=0.0,
                        help="Zero-out elements with clf_prob < threshold, then renorm")
    parser.add_argument("--mixup-alpha",     type=float, default=0.0,
                        help="Beta distribution alpha for mixup (0=off)")
    parser.add_argument("--run-name",        default=None)


def add_args(parser: argparse.ArgumentParser):
    _shared_args(parser)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularisation strength")


def _dl_args(parser):
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--patience",    type=int,   default=15)


def add_cnn_args(parser: argparse.ArgumentParser):
    _shared_args(parser)
    parser.add_argument("--patch-half",  type=int, default=10)
    parser.add_argument("--max-patches", type=int, default=8)
    _dl_args(parser)


def add_shared_cnn_args(parser: argparse.ArgumentParser):
    _shared_args(parser)
    parser.add_argument("--patch-half",  type=int, default=10)
    parser.add_argument("--max-patches", type=int, default=8)
    _dl_args(parser)


def add_transformer_args(parser: argparse.ArgumentParser):
    _shared_args(parser)
    parser.add_argument("--d-model",    type=int,   default=64)
    parser.add_argument("--n-heads",    type=int,   default=4)
    parser.add_argument("--n-layers",   type=int,   default=2)
    parser.add_argument("--n-seeds",    type=int,   default=1,
                        help="Number of seeds for ensembling")
    parser.add_argument("--curriculum", action="store_true",
                        help="Curriculum: train easy→hard samples")
    _dl_args(parser)
