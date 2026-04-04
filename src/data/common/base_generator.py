import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from src.common.spectrum_utils import XRFSimulator

@dataclass
class GeneratorConfig:
    """
    Configuration for dataset generation specifying the ranges for simulation parameters.
    
    Guidelines & Recommended Ranges:
    - s_counts_range: (Signal) Controls the total counts of the main elemental peaks.
        * High Quality (e.g. Denoising Target): (80000, 150000)
        * Standard Regression/Classification: (30000, 80000)
        * Low Quality/Fast Scan: (10000, 30000)
    - n_counts_range: (Noise) Controls the Poisson noise (statistical fluctuations).
        * Perfectly Clean (Ground Truth): (1, 1)
        * Mild Noise: (500, 5000)
        * Very Noisy Spectrum: (10000, 30000)
    - b_counts_range: (Bremsstrahlung) The continuous background curve from the X-ray tube.
        * Low Background: (1000, 2000)
        * Typical Background: (3000, 3000)
        * High Background: (5000, 10000)
    - c_counts_range: (Characteristic Background) Additional background counts from scattering.
        * Typical Background: (3000, 3000)
    - escape_prob: Probability to simulate escape peaks. Default 1.0.
    - sum_peaks_prob: Probability to simulate sum (pile-up) peaks. Default 1.0.
    - decal_prob: Probability to apply random energy calibration drift. Default 0.0.
    - kvp_range: Tube potential in keV. Default (30.0, 30.0).
    - angle_range: Anode angle in degrees. Default (46.0, 46.0).
    - mas_range: Exposure setting in milli-Ampere-seconds. Default (9.0, 9.0).
    - target_materials: Anode target material ("W", "Mo", "Rh").
    - filters: Tube filtration as (material, thickness_mm) pairs.
    """
    s_counts_range: Tuple[int, int] = (30000, 80000)
    n_counts_range: Tuple[int, int] = (1000, 5000)
    b_counts_range: Tuple[int, int] = (3000, 3000)
    c_counts_range: Tuple[int, int] = (3000, 3000)
    escape_prob: float = 1.0
    sum_peaks_prob: float = 1.0
    decal_prob: float = 0.0
    kvp_range: Tuple[float, float] = (30.0, 30.0)
    angle_range: Tuple[float, float] = (46.0, 46.0)
    mas_range: Tuple[float, float] = (9.0, 9.0)
    target_materials: List[str] = field(default_factory=lambda: ["Mo"])
    filters: List[Tuple[str, float]] = field(default_factory=lambda: [('Be', 0.127), ('Air', 10.0)])

    class Presets:
        """Namespace for preconfigured GeneratorConfig presets."""
        
        @staticmethod
        def balanced() -> 'GeneratorConfig':
            """Standard, clean laboratory-like configuration."""
            return GeneratorConfig(
                s_counts_range=(30000, 80000),
                n_counts_range=(1000, 5000),
                b_counts_range=(3000, 3000),
                c_counts_range=(3000, 3000),
                escape_prob=1.0,
                sum_peaks_prob=1.0,
                decal_prob=0.0,
                kvp_range=(30.0, 30.0),
                angle_range=(46.0, 46.0),
                mas_range=(9.0, 9.0),
                target_materials=["Mo"],
                filters=[('Be', 0.127), ('Air', 10.0)]
            )

        @staticmethod
        def high_quality() -> 'GeneratorConfig':
            """Ideal for generating 'ground truth' denoising targets or high-precision training data."""
            return GeneratorConfig(
                s_counts_range=(80000, 150000),
                n_counts_range=(1, 1),
                b_counts_range=(1000, 2000),
                c_counts_range=(1000, 2000),
                escape_prob=1.0,
                sum_peaks_prob=1.0,
                decal_prob=0.0,
                kvp_range=(30.0, 30.0),
                angle_range=(46.0, 46.0),
                mas_range=(9.0, 9.0),
                target_materials=["Mo"],
                filters=[('Be', 0.127), ('Air', 10.0)]
            )

        @staticmethod
        def fast_scan() -> 'GeneratorConfig':
            """Simulates low-quality, noisy data (e.g. handheld XRF device)."""
            return GeneratorConfig(
                s_counts_range=(10000, 30000),
                n_counts_range=(10000, 30000),
                b_counts_range=(3000, 8000),
                c_counts_range=(3000, 3000),
                escape_prob=1.0,
                sum_peaks_prob=1.0,
                decal_prob=0.2,
                kvp_range=(30.0, 30.0),
                angle_range=(46.0, 46.0),
                mas_range=(9.0, 9.0),
                target_materials=["Mo"],
                filters=[('Be', 0.127), ('Air', 10.0)]
            )

        @staticmethod
        def robust_training() -> 'GeneratorConfig':
            """Varies all physical parameters to help models generalize across different hardware."""
            return GeneratorConfig(
                s_counts_range=(20000, 100000),
                n_counts_range=(1000, 20000),
                b_counts_range=(3000, 3000),
                c_counts_range=(3000, 3000),
                escape_prob=1.0,
                sum_peaks_prob=1.0,
                decal_prob=0.5,
                kvp_range=(20.0, 50.0),
                angle_range=(46.0, 46.0),
                mas_range=(9.0, 9.0),
                target_materials=["Mo", "W", "Rh"],
                filters=[('Be', 0.127), ('Air', 10.0)]
            )

        @staticmethod
        def classification() -> 'GeneratorConfig':
            """Configuration optimized for training element identification models."""
            return GeneratorConfig(
                s_counts_range=(10000, 80000),
                n_counts_range=(500, 10000),
                b_counts_range=(3000, 5000),
                c_counts_range=(3000, 3000),
                escape_prob=1.0,
                sum_peaks_prob=1.0,
                decal_prob=0.1,
                kvp_range=(30.0, 30.0),
                angle_range=(46.0, 46.0),
                mas_range=(9.0, 9.0),
                target_materials=["Mo", "W", "Rh"],
                filters=[('Be', 0.127), ('Air', 10.0)]
            )

        @staticmethod
        def regression() -> 'GeneratorConfig':
            """Configuration optimized for training elemental quantification models."""
            return GeneratorConfig(
                s_counts_range=(50000, 150000),
                n_counts_range=(1000, 5000),
                b_counts_range=(3000, 3000),
                c_counts_range=(3000, 3000),
                escape_prob=1.0,
                sum_peaks_prob=1.0,
                decal_prob=0.0,
                kvp_range=(30.0, 30.0),
                angle_range=(46.0, 46.0),
                mas_range=(9.0, 9.0),
                target_materials=["Mo"],
                filters=[('Be', 0.127), ('Air', 10.0)]
            )

        @staticmethod
        def denoising() -> 'GeneratorConfig':
            """Configuration optimized for generating noisy inputs for denoising models."""
            return GeneratorConfig(
                s_counts_range=(10000, 30000),
                n_counts_range=(10000, 30000),
                b_counts_range=(3000, 8000),
                c_counts_range=(3000, 3000),
                escape_prob=1.0,
                sum_peaks_prob=1.0,
                decal_prob=0.0,
                kvp_range=(30.0, 30.0),
                angle_range=(46.0, 46.0),
                mas_range=(9.0, 9.0),
                target_materials=["Mo"],
                filters=[('Be', 0.127), ('Air', 10.0)]
            )

    def summary(self):
        """Prints a summary of the configuration settings."""
        print("-" * 40)
        print(f"{'XRF Generator Configuration':^40}")
        print("-" * 40)
        for key, val in asdict(self).items():
            print(f"{key:<20}: {val}")
        print("-" * 40)

def preview_dataset(generator: Any, config: GeneratorConfig, num_samples: int = 3):
    """
    Utility to quickly generate and plot a few spectra from a config for validation.
    """
    print(f"Generating {num_samples} preview samples...")
    # BaseXRFGenerator doesn't have generate_dataset, so we rely on Duck Typing
    res = generator.generate_dataset(num_samples=num_samples, config=config)
    spectra = res[0]
    
    energies = np.arange(0, 30, 0.05)
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.plot(energies, spectra[i], label=f"Sample {i}", alpha=0.7)
    
    plt.title(f"Dataset Preview - {generator.__class__.__name__}")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

class BaseXRFGenerator:
    """
    Base class for generating synthetic XRF datasets.
    """
    def __init__(self, simulator: Optional[XRFSimulator] = None, seed: Optional[int] = None):
        if simulator is None:
            self.simulator = XRFSimulator()
        else:
            self.simulator = simulator
            
        if seed is not None:
            np.random.seed(seed)
            
        # Common elements supported by multiel-spectra library
        from multiel_spectra.xrf_utils import reduced_list
        self.common_elements = reduced_list

    def _generate_random_elements(self, min_elements: int = 1, max_elements: int = 5) -> List[str]:
        """Selects a random subset of elements."""
        num_els = np.random.randint(min_elements, max_elements + 1)
        return np.random.choice(self.common_elements, size=num_els, replace=False).tolist()
