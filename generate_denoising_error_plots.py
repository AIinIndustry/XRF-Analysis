import numpy as np
import matplotlib.pyplot as plt
import os

# Consolidated Results Data
architectures = ["CNN", "Conformer", "LSTM", "Linear", "Transformer", "MLP", "UNet", "ResNet1D"]

# MAE data (Original Scale * 1000)
mae_data = {
    "Standard": [1.25, 1.22, 1.34, 1.66, 1.44, 1.72, 1.20, 1.32],
    "MinMax": [1.56, 2.40, 1.29, 1.71, 1.35, 1.70, 1.38, 1.56],
    "LogMinMax": [1.43, 1.30, 1.27, 1.64, 1.40, 1.68, 1.39, 1.51]
}

# MSE data (Original Scale * 10^5)
mse_data = {
    "Standard": [3.58, 3.93, 4.13, 4.19, 4.36, 4.37, 4.88, 4.94],
    "MinMax": [4.71, 11.70, 4.24, 5.12, 4.55, 5.21, 4.46, 4.86],
    "LogMinMax": [4.38, 4.45, 4.40, 4.77, 4.54, 4.92, 4.63, 4.74]
}

def create_grouped_bar(data, architectures, title, ylabel, save_path, reverse=True):
    # Sort architectures based on Standard values (DESC)
    std_vals = data["Standard"]
    sort_idx = np.argsort(std_vals)
    if reverse:
        sort_idx = sort_idx[::-1]
    
    sorted_archs = [architectures[i] for i in sort_idx]
    
    x = np.arange(len(sorted_archs))
    width = 0.25
    
    plt.figure(figsize=(14, 8))
    
    plt.bar(x - width, [data["Standard"][i] for i in sort_idx], width, label='Standard', color='#3498db', edgecolor='navy', alpha=0.8)
    plt.bar(x, [data["MinMax"][i] for i in sort_idx], width, label='MinMax', color='#e74c3c', edgecolor='darkred', alpha=0.8)
    plt.bar(x + width, [data["LogMinMax"][i] for i in sort_idx], width, label='LogMinMax', color='#2ecc71', edgecolor='darkgreen', alpha=0.8)
    
    plt.xlabel('Architecture', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, sorted_archs, rotation=45, fontsize=10)
    plt.legend(frameon=True, shadow=True, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Generate Grouped Plots
create_grouped_bar(mae_data, architectures, 
                  'Denoising MAE Comparison (Standard DESC)', 
                  'MAE (Original Scale, x10⁻³)', 
                  'media/denoising_mae_comparison.png')

create_grouped_bar(mse_data, architectures, 
                  'Denoising MSE Comparison (Standard DESC)', 
                  'MSE (Original Scale, x10⁻⁵)', 
                  'media/denoising_mse_comparison.png')

print("Grouped denoising error comparison plots (no labels) generated successfully.")
