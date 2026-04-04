import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
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

def _generate_regression_sample(
    active_elements: List[str],
    conc_dict: Dict[str, float],
    s_counts: int,
    n_counts: int,
    b_counts: int,
    c_counts: int,
    escape: bool,
    sum_peaks: bool,
    decal: bool,
    kvp: float,
    angle: float,
    mas: float,
    target: str,
    config_filters: List[Tuple[str, float]]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Worker function for parallel regression sample generation."""
    simulator = _get_simulator()
    
    with suppress_stdout():
        res = simulator.simulate_xrf_spectrum(
            elements=active_elements, 
            s_counts=s_counts, 
            n_counts=n_counts,
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
            filters=config_filters
        )
    
    return res[0], conc_dict

class RegressionDataGenerator(BaseXRFGenerator):
    """
    Generates datasets for Regression (Element Concentration Prediction).
    """
    def generate_dataset(
        self, 
        num_samples: int, 
        min_elements: int = 1, 
        max_elements: int = 5, 
        config: Optional[GeneratorConfig] = None,
        num_workers: int = 4
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generates spectra and continuous target concentrations.
        """
        if config is None:
            config = GeneratorConfig.Presets.regression()

        # 1. Pre-generate all random parameters
        tasks = []
        for _ in range(num_samples):
            # Random number of elements to be present
            num_els = np.random.randint(min_elements, max_elements + 1)
            active_elements = np.random.choice(self.common_elements, size=num_els, replace=False).tolist()
            
            # Generate random concentrations that sum to 1.0
            concs = np.random.dirichlet(np.ones(num_els))
            conc_dict = {el: c for el, c in zip(active_elements, concs)}
            
            s_counts = np.random.randint(config.s_counts_range[0], config.s_counts_range[1] + 1)
            n_counts = np.random.randint(config.n_counts_range[0], config.n_counts_range[1] + 1)
            b_counts = np.random.randint(config.b_counts_range[0], config.b_counts_range[1] + 1)
            c_counts = np.random.randint(config.c_counts_range[0], config.c_counts_range[1] + 1)
            escape = bool(np.random.random() < config.escape_prob)
            sum_peaks = bool(np.random.random() < config.sum_peaks_prob)
            decal = bool(np.random.random() < config.decal_prob)
            kvp = np.random.uniform(config.kvp_range[0], config.kvp_range[1])
            angle = np.random.uniform(config.angle_range[0], config.angle_range[1])
            mas = np.random.uniform(config.mas_range[0], config.mas_range[1])
            target = np.random.choice(config.target_materials)
            
            tasks.append((
                active_elements, conc_dict, s_counts, n_counts, b_counts, c_counts,
                escape, sum_peaks, decal, kvp, angle, mas, target, config.filters
            ))

        # 2. Run Generation
        spectra_list = []
        concentrations_list = []
        
        if num_workers <= 1:
            for task in tqdm(tasks, desc="Sequential Regression Generation"):
                spectrum, conc_dict = _generate_regression_sample(*task)
                spectra_list.append(spectrum)
                concentrations_list.append(conc_dict)
        else:
            print(f"Starting parallel regression generation with {num_workers} workers...")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(
                    executor.map(_generate_regression_sample, *zip(*tasks)), 
                    total=num_samples, 
                    desc="Parallel Regression Generation"
                ))
            for spectrum, conc_dict in results:
                spectra_list.append(spectrum)
                concentrations_list.append(conc_dict)
            
        # 3. Compile into DataFrame
        concs_df = pd.DataFrame(concentrations_list).fillna(0.0)
        
        # Ensure all common elements are columns
        for el in self.common_elements:
            if el not in concs_df.columns:
                concs_df[el] = 0.0
                
        concs_df = concs_df[self.common_elements]
        
        return np.array(spectra_list), concs_df
