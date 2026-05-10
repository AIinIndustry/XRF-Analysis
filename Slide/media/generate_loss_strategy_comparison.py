import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

architectures = ["CNN", "UNet1D", "Transformer", "Conformer", "ResNet1D", "LSTM", "MLP", "Linear"]

# ── Data per scaler ────────────────────────────────────────────────────────────

mse_standard = {
    "Standard MSE":      [4.05e-05, 3.68e-05, 4.37e-05, 3.52e-05, 4.31e-05, 3.71e-05, 4.18e-05, 4.45e-05],
    "Standard MSE + L1": [3.58e-05, 4.88e-05, 4.36e-05, 3.93e-05, 4.94e-05, 4.13e-05, 4.37e-05, 4.19e-05],
    "Masked MSE + L1":   [6.97e-05, 6.86e-05, 6.73e-05, 5.99e-05, 6.03e-05, 6.21e-05, 7.13e-05, 6.79e-05],
    "Masked MSE":        [7.57e-05, 7.01e-05, 7.20e-05, 6.88e-05, 8.00e-05, 7.59e-05, 8.48e-05, 7.60e-05],
}
mae_standard = {
    "Standard MSE":      [1.23e-03, 1.20e-03, 1.32e-03, 1.16e-03, 1.26e-03, 1.33e-03, 1.68e-03, 1.64e-03],
    "Standard MSE + L1": [1.25e-03, 1.20e-03, 1.44e-03, 1.22e-03, 1.32e-03, 1.34e-03, 1.72e-03, 1.66e-03],
    "Masked MSE + L1":   [2.09e-03, 1.81e-03, 2.05e-03, 1.92e-03, 2.00e-03, 2.10e-03, 2.49e-03, 2.36e-03],
    "Masked MSE":        [2.13e-03, 1.92e-03, 2.09e-03, 1.93e-03, 2.17e-03, 2.23e-03, 2.54e-03, 2.48e-03],
}

mse_logminmax = {
    "Standard MSE":      [4.35e-05, 3.96e-05, 4.07e-05, 4.02e-05, 4.31e-05, 3.96e-05, 4.50e-05, 4.80e-05],
    "Standard MSE + L1": [4.38e-05, 4.63e-05, 4.54e-05, 4.45e-05, 4.74e-05, 4.40e-05, 4.92e-05, 4.77e-05],
    "Masked MSE + L1":   [6.72e-05, 6.50e-05, 6.69e-05, 6.16e-05, 6.93e-05, 6.74e-05, 7.35e-05, 6.92e-05],
    "Masked MSE":        [6.89e-05, 7.29e-05, 7.77e-05, 7.27e-05, 7.78e-05, 7.45e-05, 8.57e-05, 7.25e-05],
}
mae_logminmax = {
    "Standard MSE":      [1.42e-03, 1.40e-03, 1.21e-03, 1.47e-03, 1.52e-03, 1.36e-03, 1.62e-03, 1.63e-03],
    "Standard MSE + L1": [1.43e-03, 1.39e-03, 1.40e-03, 1.30e-03, 1.51e-03, 1.27e-03, 1.68e-03, 1.64e-03],
    "Masked MSE + L1":   [2.08e-03, 2.09e-03, 1.90e-03, 2.27e-03, 2.42e-03, 2.29e-03, 2.37e-03, 2.25e-03],
    "Masked MSE":        [2.01e-03, 2.12e-03, 1.85e-03, 1.97e-03, 2.38e-03, 2.10e-03, 2.50e-03, 2.30e-03],
}

mse_minmax = {
    "Standard MSE":      [4.30e-05, 4.42e-05, 4.09e-05, 4.18e-05, 4.30e-05, 3.91e-05, 5.16e-05, 5.15e-05],
    "Standard MSE + L1": [4.71e-05, 4.46e-05, 4.55e-05, 1.17e-04, 4.86e-05, 4.24e-05, 5.21e-05, 5.12e-05],
    "Masked MSE + L1":   [7.08e-05, 7.17e-05, 6.74e-05, 6.75e-05, 6.95e-05, 6.55e-05, 7.41e-05, 6.96e-05],
    "Masked MSE":        [6.94e-05, 7.62e-05, 7.72e-05, 7.54e-05, 7.89e-05, 7.47e-05, 7.92e-05, 7.39e-05],
}
mae_minmax = {
    "Standard MSE":      [1.40e-03, 1.39e-03, 1.14e-03, 1.52e-03, 1.67e-03, 1.35e-03, 1.71e-03, 1.73e-03],
    "Standard MSE + L1": [1.56e-03, 1.38e-03, 1.35e-03, 2.40e-03, 1.56e-03, 1.29e-03, 1.70e-03, 1.71e-03],
    "Masked MSE + L1":   [2.31e-03, 2.25e-03, 2.19e-03, 2.08e-03, 2.42e-03, 2.07e-03, 2.45e-03, 2.30e-03],
    "Masked MSE":        [2.18e-03, 2.15e-03, 1.95e-03, 2.39e-03, 2.35e-03, 2.01e-03, 2.46e-03, 2.39e-03],
}

# ── Plot ───────────────────────────────────────────────────────────────────────

colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
strategies = list(mse_standard.keys())
n = len(architectures)
n_strategies = len(strategies)
width = 0.18
x = np.arange(n)
base = "/Users/enricoferraiolo/Desktop/XRF-Analysis/Slide/media"


def plot_single(data, ylabel, scale, title, output):
    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = np.linspace(-(n_strategies - 1) / 2, (n_strategies - 1) / 2, n_strategies) * width
    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        vals = np.array(data[strategy]) * scale
        ax.bar(x + offsets[i], vals, width, label=strategy, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    best_vals = np.array(list(data.values())) * scale
    for arch_idx in range(n):
        col_vals = best_vals[:, arch_idx]
        best_strat_idx = np.argmin(col_vals)
        best_val = col_vals[best_strat_idx]
        bar_x = x[arch_idx] + offsets[best_strat_idx]
        ax.annotate("★", xy=(bar_x, best_val), ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color=c, label=s, alpha=0.85) for s, c in zip(strategies, colors)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


for mse_data, mae_data, tag, label in [
    (mse_standard,  mae_standard,  "standard",  "Standard Scaler"),
    (mse_logminmax, mae_logminmax, "logminmax", "LogMinMax Scaler"),
    (mse_minmax,    mae_minmax,    "minmax",    "MinMax Scaler"),
]:
    plot_single(mse_data, "MSE (×10⁻⁵)", 1e5,
                f"Loss Strategy Comparison — MSE ({label})",
                f"{base}/denoising_cs_mse_{tag}.png")
    plot_single(mae_data, "MAE (×10⁻³)", 1e3,
                f"Loss Strategy Comparison — MAE ({label})",
                f"{base}/denoising_cs_mae_{tag}.png")
