import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
from src.data.common.base_generator import BaseXRFGenerator, GeneratorConfig

class DenoisingDataGenerator(BaseXRFGenerator):
    """
    Generates paired datasets (clean and noisy spectra) for Denoising models.
    """
    def generate_dataset(self, num_samples: int, min_elements: int = 1, max_elements: int = 5, config: Optional[GeneratorConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates pairs of (noisy_spectrum, clean_spectrum).
        
        Args:
            num_samples: Number of samples to generate.
            min_elements: Minimum number of elements per sample.
            max_elements: Maximum number of elements per sample.
            config: Optional Configuration. Defines ranges for the noisy inputs.
                    The clean targets will use perfectly noiseless settings (n_counts=1).
            
        Returns:
            noisy_spectra: (num_samples, num_channels) array
            clean_spectra: (num_samples, num_channels) array
        """
        if config is None:
            config = GeneratorConfig.Presets.denoising()

        noisy_list = []
        clean_list = []
        
        for _ in tqdm(range(num_samples), desc="Generating Denoising Dataset"):
            elements = self._generate_random_elements(min_elements, max_elements)
            
            # Common physical parameters for both clean and noisy samples
            s_counts = np.random.randint(config.s_counts_range[0], config.s_counts_range[1] + 1)
            b_counts = np.random.randint(config.b_counts_range[0], config.b_counts_range[1] + 1)
            c_counts = np.random.randint(config.c_counts_range[0], config.c_counts_range[1] + 1)
            
            escape = bool(np.random.random() < config.escape_prob)
            sum_peaks = bool(np.random.random() < config.sum_peaks_prob)
            decal = bool(np.random.random() < config.decal_prob)
            kvp = np.random.uniform(config.kvp_range[0], config.kvp_range[1])
            angle = np.random.uniform(config.angle_range[0], config.angle_range[1])
            mas = np.random.uniform(config.mas_range[0], config.mas_range[1])
            target = np.random.choice(config.target_materials)
            
            # Clean parameters
            clean_n_counts = 1 # Set to 1 to bypass divide-by-zero bug in multiel_spectra
            
            # Noisy parameters
            noisy_n_counts = np.random.randint(config.n_counts_range[0], config.n_counts_range[1] + 1)
            
            # 1. Generate clean spectrum
            res_clean = self.simulator.simulate_xrf_spectrum(
                elements=elements, 
                s_counts=s_counts, 
                n_counts=clean_n_counts,
                b_counts=b_counts,
                c_counts=c_counts,
                plot=False,
                escape=escape,
                sum_peaks=sum_peaks,
                decal=decal,
                kvp=kvp,
                angle=angle,
                mas=mas,
                target=target,
                filters=config.filters
            )
            
            # 2. Generate noisy spectrum
            res_noisy = self.simulator.simulate_xrf_spectrum(
                elements=elements, 
                s_counts=s_counts, 
                n_counts=noisy_n_counts,
                b_counts=b_counts,
                c_counts=c_counts,
                plot=False,
                escape=escape,
                sum_peaks=sum_peaks,
                decal=decal,
                kvp=kvp,
                angle=angle,
                mas=mas,
                target=target,
                filters=config.filters
            )
            
            clean_list.append(res_clean[0])
            noisy_list.append(res_noisy[0])
            
        return np.array(noisy_list), np.array(clean_list)
