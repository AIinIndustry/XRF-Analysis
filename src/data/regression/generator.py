import numpy as np
import pandas as pd
from typing import Tuple, Optional
from tqdm import tqdm
from src.data.common.base_generator import BaseXRFGenerator, GeneratorConfig

class RegressionDataGenerator(BaseXRFGenerator):
    """
    Generates datasets for Regression (Element Concentration Prediction).
    """
    def generate_dataset(self, num_samples: int, min_elements: int = 1, max_elements: int = 5, config: Optional[GeneratorConfig] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generates spectra and continuous target concentrations.
        
        Args:
            num_samples: Number of samples to generate.
            min_elements: Minimum number of elements per sample.
            max_elements: Maximum number of elements per sample.
            config: Optional Configuration for noise, signal and background ranges.
            
        Returns:
            spectra: (num_samples, num_channels) array
            concentrations: DataFrame containing relative concentrations for each element
        """
        if config is None:
            config = GeneratorConfig.Presets.regression()

        spectra_list = []
        concentrations_list = []
        
        for _ in tqdm(range(num_samples), desc="Generating Regression Dataset"):
            # Random number of elements to be present
            num_els = np.random.randint(min_elements, max_elements + 1)
            active_elements = np.random.choice(self.common_elements, size=num_els, replace=False)
            
            # Generate random concentrations that sum to 1.0
            concs = np.random.dirichlet(np.ones(num_els))
            
            # Build concentration dictionary for ground truth
            conc_dict = {el: c for el, c in zip(active_elements, concs)}
            
            escape = bool(np.random.random() < config.escape_prob)
            sum_peaks = bool(np.random.random() < config.sum_peaks_prob)
            decal = bool(np.random.random() < config.decal_prob)
            kvp = np.random.uniform(config.kvp_range[0], config.kvp_range[1])
            angle = np.random.uniform(config.angle_range[0], config.angle_range[1])
            mas = np.random.uniform(config.mas_range[0], config.mas_range[1])
            target = np.random.choice(config.target_materials)
            
            # Assuming elements argument handles list of symbols for now.
            res = self.simulator.simulate_xrf_spectrum(
                elements=active_elements.tolist(), 
                s_counts=np.random.randint(config.s_counts_range[0], config.s_counts_range[1] + 1), 
                n_counts=np.random.randint(config.n_counts_range[0], config.n_counts_range[1] + 1),
                b_counts=np.random.randint(config.b_counts_range[0], config.b_counts_range[1] + 1),
                c_counts=np.random.randint(config.c_counts_range[0], config.c_counts_range[1] + 1),
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
            
            spectra_list.append(res[0])
            concentrations_list.append(conc_dict)
            
        # Compile into DataFrame
        concs_df = pd.DataFrame(concentrations_list).fillna(0.0)
        
        # Ensure all common elements are columns
        for el in self.common_elements:
            if el not in concs_df.columns:
                concs_df[el] = 0.0
                
        concs_df = concs_df[self.common_elements]
        
        return np.array(spectra_list), concs_df
