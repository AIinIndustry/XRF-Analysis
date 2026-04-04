import numpy as np
import multiel_spectra
from tqdm import tqdm
from typing import Tuple, Optional, List
from concurrent.futures import ProcessPoolExecutor
from src.data.common.base_generator import BaseXRFGenerator, GeneratorConfig
from src.common.spectrum_utils import XRFSimulator, suppress_stdout

# Global worker simulator to persist cache across calls in the same process
_worker_simulator = None

def _get_simulator() -> XRFSimulator:
    """Lazily initializes or returns the global worker simulator."""
    global _worker_simulator
    if _worker_simulator is None:
        _worker_simulator = XRFSimulator()
    return _worker_simulator

def _generate_single_sample(
    elements: List[str],
    s_counts: int,
    b_counts: int,
    c_counts: int,
    escape: bool,
    sum_peaks: bool,
    decal: bool,
    kvp: float,
    angle: float,
    mas: float,
    target: str,
    config_filters: List[Tuple[str, float]],
    noisy_n_counts: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone worker function for parallel generation.
    It uses a global simulator instance per process to leverage caching.
    """
    simulator = _get_simulator()
    
    # 1. Generate/Fetch Primary Spectrum
    Prim, brems = simulator.generate_primary_spectrum(
        filters=config_filters, 
        kvp=kvp, 
        angle=angle, 
        mas=mas, 
        target=target
    )
    
    # Clean parameters
    clean_n_counts = 1 
    
    with suppress_stdout():
        # 2. Generate clean spectrum
        res_clean = multiel_spectra.spectra_gen(
            a=elements,
            Prim=Prim,
            brems=brems,
            s_counts=s_counts,
            n_counts=clean_n_counts,
            b_counts=b_counts,
            c_counts=c_counts,
            plot=False,
            escape=escape,
            sum=sum_peaks,
            decal=decal
        )
        
        # 3. Generate noisy spectrum
        res_noisy = multiel_spectra.spectra_gen(
            a=elements,
            Prim=Prim,
            brems=brems,
            s_counts=s_counts,
            n_counts=noisy_n_counts,
            b_counts=b_counts,
            c_counts=c_counts,
            plot=False,
            escape=escape,
            sum=sum_peaks,
            decal=decal
        )
    
    return res_noisy[0], res_clean[0]

class DenoisingDataGenerator(BaseXRFGenerator):
    """
    Generates paired datasets (clean and noisy spectra) for Denoising models.
    Supports parallel multi-core generation.
    """
    def generate_dataset(
        self, 
        num_samples: int, 
        min_elements: int = 1, 
        max_elements: int = 5, 
        config: Optional[GeneratorConfig] = None,
        num_workers: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates pairs of (noisy_spectrum, clean_spectrum).
        """
        if config is None:
            config = GeneratorConfig.Presets.denoising()

        if num_workers <= 1:
            # Fallback to sequential generation
            return self._generate_sequential(num_samples, min_elements, max_elements, config)

        # 1. Pre-generate all random parameters
        tasks = []
        for _ in range(num_samples):
            elements = self._generate_random_elements(min_elements, max_elements)
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
            noisy_n_counts = np.random.randint(config.n_counts_range[0], config.n_counts_range[1] + 1)
            
            tasks.append((
                elements, s_counts, b_counts, c_counts, escape, sum_peaks, decal,
                kvp, angle, mas, target, config.filters, noisy_n_counts
            ))

        # 2. Run in Parallel
        noisy_list = []
        clean_list = []
        
        print(f"Starting parallel generation with {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Using map to maintain order and show progress
            results = list(tqdm(
                executor.map(_generate_single_sample, *zip(*tasks)), 
                total=num_samples, 
                desc="Parallel Generation"
            ))
            
        for res_noisy, res_clean in results:
            noisy_list.append(res_noisy)
            clean_list.append(res_clean)
            
        return np.array(noisy_list), np.array(clean_list)

    def _generate_sequential(self, num_samples, min_elements, max_elements, config):
        """Standard sequential generation loop with caching."""
        self.simulator.clear_cache()
        noisy_list = []
        clean_list = []
        
        for _ in tqdm(range(num_samples), desc="Sequential Generation"):
            elements = self._generate_random_elements(min_elements, max_elements)
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
            noisy_n_counts = np.random.randint(config.n_counts_range[0], config.n_counts_range[1] + 1)

            res_noisy, res_clean = _generate_single_sample(
                elements, s_counts, b_counts, c_counts, escape, sum_peaks, decal,
                kvp, angle, mas, target, config.filters, noisy_n_counts
            )
            noisy_list.append(res_noisy)
            clean_list.append(res_clean)
            
        return np.array(noisy_list), np.array(clean_list)
