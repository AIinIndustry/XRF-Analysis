# Regression Models — Elemental Concentration Prediction

## Problem Statement

Given a simulated XRF spectrum (`X`: shape `[N, 600]`, energy range 0–30 keV at 0.05 keV bins),
predict the relative concentration of each of the 41 supported elements (`y`: shape `[N, 41]`).

Concentrations are generated via a Dirichlet distribution, so they are non-negative and **sum to 1.0**.
Per sample, only 2–5 elements are active; the rest are exactly 0. This makes the target **sparse**.

This is harder than classification: the model must not only detect which elements are present,
but also estimate *how much* of each — a continuous, structured output problem.

---

## Data Configuration

Training data is generated using `GeneratorConfig.Presets.regression()`:

| Parameter         | Value         | Notes                                      |
|-------------------|---------------|--------------------------------------------|
| `s_counts_range`  | (50000–150000) | High signal → clean, well-resolved peaks   |
| `n_counts_range`  | (1000–5000)   | Mild Poisson noise                          |
| `b_counts_range`  | (3000–3000)   | Fixed Bremsstrahlung background             |
| `decal_prob`      | 0.0           | No energy calibration drift                 |
| `target_materials`| ["Mo"]        | Fixed tube anode material                   |

For robustness experiments, also consider `GeneratorConfig.Presets.fast_scan()` (high noise,
possible calibration drift) to simulate a handheld XRF device.

---

## Proposed Architecture

A **1D CNN regressor** is the primary approach. It shares the same backbone as the
classification model, differing only in the output head.

```
Input: [batch, 600]
  │
  ▼
Conv1D(64, k=7) → BN → ReLU → MaxPool(2)   # [batch, 64, 300]
  │
Conv1D(128, k=5) → BN → ReLU → MaxPool(2)  # [batch, 128, 150]
  │
Conv1D(256, k=3) → BN → ReLU → MaxPool(2)  # [batch, 256, 75]
  │
GlobalAvgPool                               # [batch, 256]
  │
Linear(256, 128) → ReLU → Dropout(0.3)
  │
Linear(128, 41) → Softmax                  # [batch, 41] — sums to 1
```

> **Why Softmax?** It enforces the physical constraint that concentrations must sum to 1,
> matching the Dirichlet-distributed targets. This is better than a sigmoid head + normalization.

**Loss function:** a combination works best:
- `MSE` on active elements only (masked by true label > 0)
- or `KL-divergence` treating the output as a probability distribution

**Metric:** `MAE` and `R²` computed **only on non-zero ground-truth elements**,
since predicting 0 for absent elements is trivial and would inflate the score.

---

## Baselines

| Model              | Notes                                                     |
|--------------------|-----------------------------------------------------------|
| **PLS** (n=20)     | Classical chemometrics baseline. Fast, interpretable.     |
| **Ridge regression**| Linear baseline with L2 regularization.                  |
| **CNN regressor**  | Primary proposed model (see above).                       |
| **Two-stage model**| Classifier → Regressor on predicted active elements only. |

The two-stage model is particularly interesting: it uses the classification output as a mask,
then runs a second network to estimate concentrations only for detected elements.

---

## Key Experiments

### Experiment 1 — Classification vs. Regression difficulty

Train both models on identical data from `GeneratorConfig.Presets.balanced()`.
Compare:
- Classification: macro F1-score per element
- Regression: MAE on active elements

**Expected result:** classification should reach F1 > 0.90, while regression will show
higher error — demonstrating that quantification is fundamentally harder than detection.

### Experiment 2 — Denoising improves regression

Two pipelines:
1. **Raw:** noisy spectrum → regressor (trained on noisy data)
2. **Denoised:** noisy spectrum → best denoising model → regressor (trained on clean data)

Generate test spectra with `GeneratorConfig.Presets.fast_scan()` (high noise) and compare
MAE for both pipelines.

**Expected result:** the denoised pipeline should show meaningfully lower MAE, especially
at high noise levels. This is the central claim of the project.

### Experiment 3 — Denoising improves classification less

Repeat Experiment 2 for the classification model.
The improvement should be smaller because peak detection is more robust to noise
than peak area estimation (which drives concentration accuracy).

**Expected result:** denoising gives a smaller relative improvement for F1 than for MAE.
This contrast is the core narrative of the project.

### Experiment 4 — Noise level ablation

Vary `n_counts_range` from (500, 500) to (30000, 30000) while keeping s_counts fixed.
For each noise level, measure:
- Classification F1 (with and without denoising)
- Regression MAE (with and without denoising)

Plot performance vs. SNR (signal-to-noise ratio). This shows at what noise level each
task degrades and how much denoising helps.

---

## Implementation Plan

```
src/models/regression/
├── README.md              ← this file
├── architectures.py       ← CNN regressor, two-stage model
├── losses.py              ← masked MSE, KL-divergence loss
├── metrics.py             ← MAE/R² on active elements only
├── trainer.py             ← training loop (mirrors denoising/trainer.py)
└── baselines.py           ← PLS and Ridge regression wrappers
```

Experiments should be run from `notebooks/`:
- `2_regression_preview.ipynb` — data exploration (already exists)
- `4_regression_training.ipynb` — train and evaluate models
- `5_pipeline_comparison.ipynb` — combined denoising + regression pipeline vs. raw

---

## Notes

- The denoising models (`src/models/denoising/`) are already implemented with several
  architectures (MLP, CNN, UNet1D, ResNet1D, Transformer, Conformer). The regression
  backbone should mirror the CNN/ResNet style for consistency.
- The regression model trained on **clean** spectra (from `high_quality` config) is the
  target for the denoised pipeline — it never sees raw noisy input during training.
- Keep the architecture shared between regression and classification so that differences
  in performance can be attributed to the task difficulty, not the model capacity.
