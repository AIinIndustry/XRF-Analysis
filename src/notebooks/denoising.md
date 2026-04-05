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
*   **Method:** Using `DenoisingDataGenerator` with a specific high-noise configuration:
    ```python
    GeneratorConfig(
        s_counts_range=(10000, 30000),  # Signal (peaks)
        n_counts_range=(10000, 30000),  # Poisson noise
        b_counts_range=(3000, 8000),    # Background
        c_counts_range=(3000, 3000),    # Compton scattering
        # Fixed parameters: kvp=30, angle=46, target="Mo", filters=[Be, Air]
    )
    ```
*   **Why Ranges?** By using a range (e.g., `n_counts_range=(10000, 30000)`) instead of fixed values, the simulator draws a random noise level for every single sample. This forces the model to generalize across a variety of realistic signal-to-noise ratios rather than overfitting to a single fixed noise profile.
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

### Strategy 3: Standard MSE + L1 Sparsity (Hybrid)
*   **Motivation:** Penalize error equally across all channels (MSE) but add an **L1 Penalty** specifically to the non-peak regions (where ground truth ≈ 0).
*   **Outcome:** Forces the background to be sparse (near-zero) while maintaining a standard reconstruction for the peaks.
*   **Pros:** Combines global reconstruction with background denoising.

### Strategy 4: Masked MSE + L1 Sparsity (Hybrid)
*   **Motivation:** Focus MSE exclusively on the peaks and use the **L1 Penalty** to zero out everything else.
*   **Outcome:** The model ignores noise on peaks but is strictly penalized for "hallucinating" peaks in the background.
*   **Pros:** **The most rigorous approach.** Achieves peak fidelity without sacrificing baseline flatness.

---

## 6. Experimental Grid Results (Summary)

We evaluated 32 configurations (8 Architectures x 4 Loss Regimes). **Standard Scaling** emerged as the most robust preprocessing strategy for MSE-based metrics across most architectures.

| Loss Regime | Top Architecture | MSE (Original) | Key Observation |
| :--- | :--- | :--- | :--- |
| **Standard MSE** | **ResNet1D** | 3.68e-05 | Balanced reconstruction; "hallucinates" noise into the baseline. |
| **Masked MSE** | **Conformer** | 6.00e-05 | Best peak preservation; zero background penalty results in noisy baselines. |
| **Standard + L1** | **Conformer** | **3.63e-05** | **Overall Winner.** Best balance of global accuracy and baseline suppression. |
| **Masked + L1** | **CNN** | 6.75e-05 | Aggressive sparsity; forces strict near-zero baseline at a small cost to peak fidelity. |

---

## 7. Technical Decision: The L1 Sparsity Penalty
We introduced a configurable `l1_lambda` (set to 1e-3 in these experiments) to control the "pressure" on the background.
*   **Implementation:** Total Loss = `MSE(Target)` + `l1_lambda * L1(Background)`.
*   **The "Standard + L1" Advantage:** This configuration uses the entire spectrum for MSE (maintaining structural consistency) while the L1 penalty specifically cleans the regions where no peaks are expected. This resolved the "baseline hallucination" issue observed in the original standard pipeline.

