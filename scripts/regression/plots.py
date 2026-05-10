"""All plotting functions for regression experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def training_curve(history: dict, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"],   label="Val")
    ax.set_title(title)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_spectra: np.ndarray,
    element_names: list,
    title: str,
    path: Path,
    n_samples: int = 3,
):
    energies = np.arange(0, 30, 0.05)
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples))

    for i in range(n_samples):
        ax_spec, ax_bar = axes[i]

        ax_spec.plot(energies, X_spectra[i], lw=1, color="steelblue")
        ax_spec.set_title(f"Sample {i} — Spectrum")
        ax_spec.set_xlabel("Energy (keV)"); ax_spec.set_ylabel("Intensity")
        ax_spec.grid(alpha=0.3)

        active = y_true[i] > 0
        els = [element_names[j] for j in range(len(element_names)) if active[j]]
        x_pos = np.arange(len(els)); w = 0.35
        ax_bar.bar(x_pos - w/2, y_true[i][active], w, label="True",  color="steelblue", alpha=0.8)
        ax_bar.bar(x_pos + w/2, y_pred[i][active], w, label="Pred",  color="tomato",    alpha=0.8)
        ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels(els)
        ax_bar.set_ylabel("Relative Concentration")
        ax_bar.set_title(f"Sample {i} — Concentrations")
        ax_bar.legend(); ax_bar.grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def per_element_mae(results: dict, path: Path):
    per_el = pd.DataFrame({
        name: m.get("per_element_mae")
        for name, m in results.items()
        if m.get("per_element_mae") is not None
    }).dropna()
    if per_el.empty:
        return
    best = per_el.columns[0]
    per_el_sorted = per_el.sort_values(best, ascending=False)

    fig, ax = plt.subplots(figsize=(16, 5))
    per_el_sorted.plot(kind="bar", ax=ax, alpha=0.8)
    ax.set_title("Per-Element MAE")
    ax.set_ylabel("MAE"); ax.set_xlabel("Element")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def model_comparison(summary: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    summary["masked_mae"].plot(kind="bar", ax=axes[0], color="steelblue", alpha=0.8)
    axes[0].set_title("Masked MAE (lower is better)")
    axes[0].set_ylabel("Masked MAE"); axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.3)

    if "masked_r2" in summary.columns:
        summary["masked_r2"].plot(kind="bar", ax=axes[1], color="seagreen", alpha=0.8)
        axes[1].set_title("Masked R² (higher is better)")
        axes[1].set_ylabel("Masked R²"); axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def noise_ablation(noise_levels: list, mae_dict: dict, path: Path):
    colors = ["tomato", "steelblue", "seagreen", "goldenrod"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, maes), color in zip(mae_dict.items(), colors):
        ax.plot(noise_levels, maes, marker="o", label=label, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Noise counts (n_counts)")
    ax.set_ylabel("Masked MAE (active elements)")
    ax.set_title("MAE vs Noise Level")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pipeline_comparison_bars(results: dict, path: Path):
    metrics = ["masked_mae", "masked_mse", "masked_r2"]
    available = [m for m in metrics if all(m in v for v in results.values())]
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    colors = ["tomato", "steelblue", "seagreen", "goldenrod"]
    names = list(results.keys())
    for ax, metric in zip(axes, available):
        vals = [results[n][metric] for n in names]
        ax.bar(names, vals, color=colors[:len(names)], alpha=0.85)
        ax.set_title(metric); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
