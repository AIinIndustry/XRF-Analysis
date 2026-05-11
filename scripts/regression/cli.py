#!/usr/bin/env python3
"""
Regression CLI — run from the project root:

  python -m scripts.regression.cli <command> [options]

Commands:
  generate          Generate and cache a dataset
  compare-models    Train all models and compare (notebook 4)
  pipeline-compare  Raw vs denoised pipeline (notebook 5)
  full-pipeline     3-stage complete pipeline (notebook 6)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.regression import data
from scripts.regression import cmd_train, cmd_compare_models, cmd_pipeline_compare, cmd_full_pipeline
from scripts.regression import cmd_per_element, classifier_cache


def _cmd_generate(args: argparse.Namespace):
    data.generate(
        preset=args.config,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        workers=args.workers,
        force=args.force,
    )


def main():
    parser = argparse.ArgumentParser(
        description="XRF Regression CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate ─────────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Generate and cache a dataset")
    p_gen.add_argument("--config",   default="regression", choices=data.VALID_PRESETS,
                       help="Generator preset (default: regression)")
    p_gen.add_argument("--n-train",  type=int, default=2000)
    p_gen.add_argument("--n-val",    type=int, default=400)
    p_gen.add_argument("--n-test",   type=int, default=400)
    p_gen.add_argument("--seed",     type=int, default=42)
    p_gen.add_argument("--workers",  type=int, default=12)
    p_gen.add_argument("--force",    action="store_true",
                       help="Regenerate even if cached data exists")

    # ── train-classifier ──────────────────────────────────────────────────────
    p_clf = sub.add_parser("train-classifier",
                            help="Train and cache CNNClassifier for feature extraction")
    p_clf.add_argument("--config",   default="high_quality", choices=data.VALID_PRESETS)
    p_clf.add_argument("--n-train",  type=int, default=10000)
    p_clf.add_argument("--n-val",    type=int, default=4000)
    p_clf.add_argument("--seed",     type=int, default=123)
    p_clf.add_argument("--workers",  type=int, default=12)
    p_clf.add_argument("--force",    action="store_true",
                       help="Retrain even if cached weights exist")

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train a single model")
    cmd_train.add_args(p_train)

    # ── per-element ───────────────────────────────────────────────────────────
    p_per = sub.add_parser("per-element",
                            help="Per-element Ridge with overlap-aware features")
    cmd_per_element.add_args(p_per)

    # ── per-element-cnn ───────────────────────────────────────────────────────
    p_per_cnn = sub.add_parser("per-element-cnn",
                                help="41 tiny CNNs, one per element, on spectral patches")
    cmd_per_element.add_cnn_args(p_per_cnn)

    # ── shared-element-cnn ────────────────────────────────────────────────────
    p_shared = sub.add_parser("shared-element-cnn",
                               help="Shared backbone CNN applied per-element (fewer params)")
    cmd_per_element.add_shared_cnn_args(p_shared)

    # ── element-transformer ───────────────────────────────────────────────────
    p_trans = sub.add_parser("element-transformer",
                              help="Transformer over element tokens with self-attention")
    cmd_per_element.add_transformer_args(p_trans)

    # ── element-transformer-v2 ────────────────────────────────────────────────
    p_trans2 = sub.add_parser("element-transformer-v2",
                               help="Transformer V2 with cross-attention to raw spectrum")
    cmd_per_element.add_transformer_args(p_trans2)

    # ── compare-models ────────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compare-models",
                            help="Train all models and compare (notebook 4)")
    cmd_compare_models.add_args(p_cmp)

    # ── pipeline-compare ──────────────────────────────────────────────────────
    p_pip = sub.add_parser("pipeline-compare",
                            help="Raw vs denoised pipeline (notebook 5)")
    cmd_pipeline_compare.add_args(p_pip)

    # ── full-pipeline ─────────────────────────────────────────────────────────
    p_full = sub.add_parser("full-pipeline",
                             help="3-stage complete pipeline (notebook 6)")
    cmd_full_pipeline.add_args(p_full)

    args = parser.parse_args()

    if args.command == "per-element":
        cmd_per_element.run(args)
    elif args.command == "per-element-cnn":
        cmd_per_element.run_cnn(args)
    elif args.command == "shared-element-cnn":
        cmd_per_element.run_shared_cnn(args)
    elif args.command == "element-transformer":
        cmd_per_element.run_transformer(args, v2=False)
    elif args.command == "element-transformer-v2":
        cmd_per_element.run_transformer(args, v2=True)
    elif args.command == "train-classifier":
        save_path = None
        if args.force:
            import shutil
            from pathlib import Path
            sp = Path(PROJECT_ROOT) / "data" / "classifiers" / f"{args.config}_n{args.n_train}_s{args.seed}"
            if sp.exists():
                shutil.rmtree(sp)
        classifier_cache.train_and_save(args.config, args.n_train, args.n_val,
                                         args.seed, args.workers)
    elif args.command == "train":
        cmd_train.run(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "compare-models":
        cmd_compare_models.run(args)
    elif args.command == "pipeline-compare":
        cmd_pipeline_compare.run(args)
    elif args.command == "full-pipeline":
        cmd_full_pipeline.run(args)


if __name__ == "__main__":
    main()
