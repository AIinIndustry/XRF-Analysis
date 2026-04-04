# XRF Spectra Denoising: Technical Decisions & Experimental Results

This document tracks the architectural decisions, preprocessing strategies, and loss function motivations for the XRF denoising task. These notes are structured to support the final project presentation.

---

## 1. Project Goal: The Denoising Challenge
X-ray Fluorescence (XRF) spectra are characterized by **discrete energy peaks** superimposed on a **stochastic background**. 
*   **The Problem:** Noise (Poisson/Gaussian) and sensor artifacts can obscure low-intensity peaks, leading to incorrect elemental identification.
*   **The Objective:** Map a noisy input spectrum $x$ to a clean ground-truth spectrum $y$, effectively performing "chemical signal recovery."

---

## 2. Data Generation Strategy
We generate a synthetic dataset of **2000 paired samples** (noisy/clean).
*   **Method:** Using `DenoisingDataGenerator` with high-noise presets.
*   **Logic:** Synthetic data allows us to have a "perfect" ground truth to calculate objective metrics (MSE/MAE) that are impossible to obtain with real-world unlabelled data.
*   **Split:** 70% Train / 15% Val / 15% Test.

---

## 3. Preprocessing & Scaling: Why it Matters
XRF data has a "Peak vs. Baseline" problem—peaks can be orders of magnitude larger than the background.

| Strategy | Technical Motivation | When to Choose |
| :--- | :--- | :--- |
| **StandardScaler** | Centers data and scales to unit variance. | Good for models sensitive to activation ranges (like MLPs). |
| **MinMaxScaler** | Compresses features to [0, 1]. | Best for image-like architectures (CNNs) and stability. |
| **LogMinMaxScaler** | **(Top Choice)** Applies $log(1+x)$ before MinMax. | **Best for XRF.** It compresses the dynamic range of massive peaks, allowing the loss function to "see" and learn small peaks that would otherwise be ignored. |

---

## 4. Architectural Inductive Biases
We evaluated 8 architectures to find the best structural fit for 1D spectral data:

### A. Spatial & Local Focus
*   **CNN (1D):** Uses local kernels to detect peak shapes. Effective for "smoothing" local noise.
*   **UNet1D:** Uses skip-connections to concatenate high-resolution features with global context. **Choice:** Excellent for maintaining precise peak positions while denoising.

### B. Sequence & Global Focus
*   **Transformer:** Uses self-attention across all 600 channels. **Choice:** Best for capturing relationships between related element lines (e.g., $K\alpha$ and $K\beta$ peaks) regardless of distance.
*   **Conformer:** Combines Conv1D (local) with Transformer (global). **Choice:** The most robust architecture for complex spectra.
*   **LSTM:** Sequential processing. Often slower and less effective for fixed-width spectra than CNNs.

### C. Complexity & Depth
*   **ResNet1D:** Residual blocks allow for deeper feature extraction without signal degradation.
*   **MLP/Linear:** Baseline models to quantify the "performance gain" of deep learning.

---

## 5. Loss Function: Standard vs. Masked
This is the most critical decision in the denoising pipeline.

### Strategy 1: Standard MSE (Unmasked)
*   **Motivation:** Penalize error equally across all 600 channels.
*   **Outcome:** The model learns to denoise the *entire* range, including the flat baseline. 
*   **Pros:** Clean-looking visual results.
*   **Cons:** The model spends too much "gradient budget" on flattening the zero-background instead of fixing peak heights.

### Strategy 2: Masked MSE
*   **Motivation:** Loss is only calculated where the ground truth peaks exist ($y > \epsilon$).
*   **Outcome:** The model is "blind" to the background and focuses exclusively on **peak intensity and position accuracy**.
*   **Pros:** Significantly higher chemical accuracy. The model recovers signal where it matters for analysis.
*   **Cons:** Background might still contain some residual artifacts (since the model isn't penalized for them).

---

## 6. Top Results Summary (Empirical Findings)

Based on our experimental grid, the following configurations emerged as leaders:

| Evaluation Regime | Top Configuration | MSE (Original Scale) | Key Observation |
| :--- | :--- | :--- | :--- |
| **Standard (Full Range)** | **ResNet1D + Standard Scaling** | ~3.68e-05 | Best at overall "aesthetic" denoising and baseline flattening. |
| **Standard (Full Range)** | **UNet1D + Standard Scaling** | ~3.79e-05 | Extremely stable and fast to converge. |
| **Masked (Peak Focus)** | **Conformer + Standard Scaling** | ~6.00e-05 | Best at preserving peak intensity and resolving overlapping peaks. |
| **Masked (Peak Focus)** | **Linear + MinMax Scaling** | ~6.34e-05 | Surprising baseline performance; suggests a strong linear component to peak recovery. |

