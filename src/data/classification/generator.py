import numpy as np
import pandas as pd
from typing import Tuple, Optional
from tqdm import tqdm
from src.data.common.base_generator import BaseXRFGenerator, GeneratorConfig

class ClassificationDataGenerator(BaseXRFGenerator):
    """
    Generates datasets for Multi-Label Classification (Element Identification).
    """
    def generate_dataset(self, num_samples: int, min_elements: int = 1, max_elements: int = 5, config: Optional[GeneratorConfig] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generates spectra and a one-hot encoded matrix of elements present.
        
        Args:
            num_samples: Number of samples to generate.
            min_elements: Minimum number of elements per sample.
            max_elements: Maximum number of elements per sample.
            config: Optional Configuration for noise, signal and background ranges.
            
        Returns:
            spectra: (num_samples, num_channels) array of spectra
            labels: DataFrame containing the one-hot encoded labels for each element
        """
        if config is None:
            config = GeneratorConfig.Presets.classification()

        spectra_list = []
        labels_list = []
        
        for _ in tqdm(range(num_samples), desc="Generating Classification Dataset"):
            # Random elements
            elements = self._generate_random_elements(min_elements, max_elements)
            
            # Randomize simulation parameters (noise, counts) based on config
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
            
            # Simulate
            res = self.simulator.simulate_xrf_spectrum(
                elements=elements, 
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
                filters=config.filters
            )
            spectrum = res[0]
            
            # Record labels
            label_dict = {el: 1 for el in elements}
            
            spectra_list.append(spectrum)
            labels_list.append(label_dict)
            
        # Compile labels into a structured format
        labels_df = pd.DataFrame(labels_list).fillna(0).astype(int)
        
        # Ensure all common elements are present in the columns for consistency
        for el in self.common_elements:
            if el not in labels_df.columns:
                labels_df[el] = 0
                
        # Reorder columns to match common_elements
        labels_df = labels_df[self.common_elements]
        
        return np.array(spectra_list), labels_df
