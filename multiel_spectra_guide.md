# Multiel Spectra Library Guide

The `multiel-spectra` library is a specialized tool for simulating X-ray Fluorescence (XRF) spectra. It allows for the creation of synthetic data by modeling the physics of X-ray tube emission, photon interaction with matter (sample), and detector response.

## Core Concepts

The simulation process typically follows two main steps:
1. **Primary Spectrum Generation**: Modeling the source (X-ray tube).
2. **XRF Spectrum Simulation**: Modeling the interaction with a sample and the detector.

---

## 1. Primary Spectrum Generation (`Primary_gen`)

This function generates the incident X-ray spectrum from the tube.

```python
import multiel_spectra as ms

# Example parameters
kvp = 30.0      # Tube potential in keV
angle = 46.0    # Anode angle in degrees
dk = 0.1        # Spectrum bin width in keV
physics = "casim"
mu_source = "nist"
z = 1.0         # Focus-to-detector distance
mas = 9.0       # Milli-Ampere-seconds (exposure)
target = "Mo"   # Anode target material ("W", "Mo", "Rh")
filters = [('Be', 0.127), ('Air', 10.0)] # (Material, Thickness in mm)

Prim, brem = ms.Primary_gen(
    k=kvp, theta=angle, d=dk, phys=physics, 
    mu_source=mu_source, z=z, mas=mas, 
    target=target, filters=filters
)
```

- **`Prim`**: The total primary spectrum (energies and probabilities).
- **`brem`**: The Bremsstrahlung component specifically.

---

## 2. XRF Spectrum Simulation (`spectra_gen`)

This function simulates the final spectrum as seen by a detector after interacting with a sample.

```python
final_spectrum, peaks, elements, decal_params = ms.spectra_gen(
    a=['Fe', 'Cu'],         # List of elements or array of concentrations
    Prim=Prim,              # From Primary_gen
    brems=brem,             # From Primary_gen
    s_counts=30000,         # Total signal counts
    n_counts=2000,          # Noise counts
    b_counts=3000,          # Bremsstrahlung background counts
    c_counts=3000,          # Characteristic background counts
    escape=True,            # Simulate escape peaks
    sum=True,               # Simulate sum peaks
    decal=True              # Simulate energy decalibration
)
```

### Key Parameters:
- **`a`**: Sample composition. 
    - Can be a list of chemical symbols: `['Fe', 'Cu', 'Zn']`.
    - Can be a `numpy` array of relative concentrations if the library is configured with a default element list.
- **Counts (`s_counts`, `n_counts`, etc.)**: Control the statistical quality and noise level of the synthetic data. Increasing `n_counts` relative to `s_counts` results in a noisier spectrum.
- **Physics Flags**:
    - `escape`: Adds peaks resulting from X-ray photons escaping the detector crystal.
    - `sum`: Adds coincidence peaks (two photons detected as one).
- **`decal`**: If `True`, the library applies a random or specified decalibration to the energy axis, simulating real-world detector drift. It returns `decal_params` which are useful for training models to be robust to calibration errors.

---

## Creating Synthetic Datasets

To create a synthetic dataset for Machine Learning, you can wrap these calls in a loop, varying the parameters:

1. **Vary Concentrations**: Randomize the input array `a` to represent different material compositions.
2. **Vary Noise**: Randomize `n_counts` and `s_counts` to simulate different acquisition times or source intensities.
3. **Vary Calibration**: Use `decal=True` to generate spectra with shifted energy axes.
4. **Vary Tube Settings**: Change `kvp` or `filters` to simulate different measurement conditions.

### Usage in this Project

The project provides a `XRFSimulator` class in `src/common/spectrum_utils.py` that simplifies this process:

```python
from src.common.spectrum_utils import XRFSimulator

sim = XRFSimulator(kvp=30, target="Mo")
# Generate a spectrum for Iron and Copper
spectrum, peaks, els, decal = sim.simulate_xrf_spectrum(['Fe', 'Cu'], decal=True)
```

This wrapper handles the interaction between `Primary_gen` and `spectra_gen` automatically.


The multiel-spectra library provides a high-level API for simulating X-ray fluorescence, built on top of physical databases like XrayDB and specialized physics models.

  Below is a detailed guide to the library's core components, derived from the library's own documentation and observed usage in the project.


  ---
Here is the fully formatted Markdown for the complete guide, including the new sections and return values:

***

## Full Multiel-Spectra Library Guide

### 1. High-Level Functions

---

#### `Primary_gen(...)`
Generates the incident X-ray spectrum from a tube using various physics models (via the `spekpy` package internally).

**Parameters**

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `k` | `float` | Tube potential in keV (e.g., 30.0). |
| `theta` | `float` | Anode angle in degrees (e.g., 46.0). |
| `d` | `float` | Spectrum bin width in keV (e.g., 0.1). |
| `phys` | `str` | Physics model for spekpy (`"legacy"`, `"spekcalc"`, `"spekpy-v1"`, `"casim"`, `"kqp"`). |
| `mu_source` | `str` | Source of photon coefficients (`"pene"`, `"nist"`). |
| `z` | `float` | Focus-to-detector distance. |
| `mas` | `float` | Exposure setting in milli-Ampere-seconds. |
| `target` | `str` | Anode target material (`"W"`, `"Mo"`, `"Rh"`). |
| `filters` | `list` | List of tuples (`material`, `thickness_mm`) for filtration. |

> **Returns:** A tuple of `(primary_spectrum, bremsstrahlung_spectrum)`, where each is a tuple of two arrays: `(energies, intensities)`.

---

#### `spectra_gen(...)`
Simulates the interaction of the primary beam with a sample to produce a detected XRF spectrum.

**Parameters**

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `a` | `str` / `list` / `arr`| Sample composition. Chemical symbols (e.g., `['Fe', 'Cu']`) or numerical concentration array. |
| `Prim` | `tuple` | Primary spectrum (from `Primary_gen`). |
| `brems` | `tuple` | Bremsstrahlung component (from `Primary_gen`). |
| `s_counts` | `int` | Total counts for the main characteristic signal. |
| `n_counts` | `int` | Counts used for generating Poisson noise. |
| `b_counts` | `int` | Intensity of the Bremsstrahlung background. |
| `c_counts` | `int` | Intensity of the characteristic background. |
| `escape` | `bool` | If `True`, simulates detector escape peaks. |
| `sum` | `bool` | If `True`, simulates sum (pile-up) peaks. |
| `decal` | `bool` | If `True`, applies random energy shift/gain (decalibration). |
| `noise_f` | `int` | Factor controlling noise granularity. |

> **Returns:** A tuple containing:
> * `final_spectrum`: The simulated intensities (y-axis).
> * `peaks`: A dictionary of calculated peak positions and intensities.
> * `elements`: The list of elements detected in the simulation.
> * `decal_params` *(optional)*: Calibration parameters (if `decal=True`).

---

### 2. Utility Constants and Databases

The library exports several useful constants and internal utilities:

* `ATOM_SYMS`: A list of chemical symbols mapping to atomic numbers.
* `XrayLine` / `XrayEdge`: Classes for representing specific atomic transitions and absorption edges.
* `material_mu(material, energy)`: Calculates the mass attenuation coefficient for a given material and energy.
* `fluor_yield(z, shell)`: Returns the fluorescence yield for a specific atomic number and shell (e.g., `'K'`, `'L'`).

---

### 3. Advanced Physics Components

The library handles the following effects internally, but they can be observed in the resulting spectra:

* **Detector Efficiency** (`detector_eff`): Models how the detector crystal's sensitivity changes across the energy range.
* **Escape Peaks** (`escape_peaks`): Simulates the loss of energy when a characteristic X-ray of the detector crystal (often Silicon or Germanium) escapes.
* **Sum Peaks** (`sum_peaks`): Simulates the effect of two photons hitting the detector simultaneously, appearing as a single peak at the sum of their energies.
* **Decalibration** (`decalibration`): Adds variations to the linear relationship between channel number and energy (Energy = Offset + Gain * Channel), crucial for building robust ML models.

---

### 4. Usage Tips for Synthetic Data Generation

1.  **Normalization:** The output of `spectra_gen` is usually a count-based intensity array. For machine learning, it is often beneficial to normalize these by total area or maximum intensity.
2.  **Calibration Mapping:** The library assumes a standard energy grid (usually 0.05 keV bins). If `decal=True`, the energy-to-index mapping is effectively changed.
3.  **Complex Backgrounds:** By adjusting the ratio of `b_counts` and `c_counts` relative to `s_counts`, you can simulate everything from high-purity laboratory measurements to low-signal field measurements with high background interference.