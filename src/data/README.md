# XRF Data Generation & Preprocessing Library

This library provides utilities for generating synthetic X-ray Fluorescence (XRF) datasets using the `multiel-spectra` physics engine. It is designed to support the development of Machine Learning (ML) and Deep Learning (DL) models for various XRF analysis tasks.

## Supported Training Goals

The library is organized into submodules based on the primary analytical objective:

### 1. Element Classification (`src/data/classification/`)
**Goal:** Multi-label classification.
**Task:** Predict which elements are present in a given spectrum.
**Output:** A binary matrix (one-hot encoded) where each column represents an element and `1` indicates its presence in the sample.

### 2. Regression (`src/data/regression/`)
**Goal:** Regression.
**Task:** Predict the relative concentration or mass fraction of each element.
**Output:** A continuous matrix where each value (0.0 to 1.0) represents the estimated concentration of an element. Concentrations are generated using a Dirichlet distribution to ensure they sum to 1.0.

### 3. Spectrum Denoising (`src/data/denoising/`)
**Goal:** Signal Reconstruction / Denoising Autoencoders.
**Task:** Remove noise and background from low-quality or short-exposure spectra.
**Output:** Paired datasets consisting of a "noisy" input spectrum and a "clean" target spectrum (generated with high signal counts and zero Poisson noise) for the same sample composition.

## Supported Elements

The library currently supports a fixed set of **41 elements** defined by the `multiel-spectra` physics engine's `reduced_list`. This list includes:
`O, F, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ge, As, Br, Sr, Zr, Mo, Rh, Pd, Ag, Cd, Sn, Sb, Te, Cs, Ba, Pt, Au, Hg, Pb, Bi`.

Attempting to simulate elements outside this list (e.g., Tungsten 'W' or Uranium 'U') will result in a `ValueError` from the underlying simulation engine.

## Architecture

- **`common/base_generator.py`**: Contains the `BaseXRFGenerator` class which wraps the `XRFSimulator`. It automatically imports the `reduced_list` from `multiel_spectra.xrf_utils` to ensure all generated samples use supported elements. It handles random element selection and randomizing simulation parameters (counts, noise, etc.).
- **Sub-generators**: Each goal-specific generator inherits from `BaseXRFGenerator` and implements a `generate_dataset` method tailored to its output format.

## Usage Example

```python
from src.data.classification.generator import ClassificationDataGenerator

# Initialize the generator
gen = ClassificationDataGenerator(seed=42)

# Generate a dataset of 1000 samples
X, y = gen.generate_dataset(num_samples=1000, min_elements=1, max_elements=5)

# X: (1000, n_channels) - Simulated spectra
# y: (1000, n_elements) - Binary labels
```
