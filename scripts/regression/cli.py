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
from scripts.regression import cmd_compare_models, cmd_pipeline_compare, cmd_full_pipeline


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

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "compare-models":
        cmd_compare_models.run(args)
    elif args.command == "pipeline-compare":
        cmd_pipeline_compare.run(args)
    elif args.command == "full-pipeline":
        cmd_full_pipeline.run(args)


if __name__ == "__main__":
    main()
