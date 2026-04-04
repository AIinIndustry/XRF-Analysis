import os
import sys
import contextlib
import numpy as np
import multiel_spectra
from multiel_spectra import ATOM_SYMS
from typing import List, Tuple, Union, Optional, Dict

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class XRFSimulator:
    """
    A utility class to simulate X-ray Fluorescence (XRF) spectra using the multiel-spectra library.
    """
    def __init__(
        self, 
        kvp: float = 30.0, 
        angle: float = 46.0, 
        dk: float = 0.1, 
        physics: str = "casim", 
        mu_source: str = "nist", 
        target: str = "Mo",
        z: float = 1.0,
        mas: float = 9.0
    ):
        """
        Initialize the simulator with tube parameters.
        
        Args:
            kvp: Tube potential in keV.
            angle: Anode angle in degrees.
            dk: Spectrum bin width in keV.
            physics: Physics model for Spekpy ("legacy", "spekcalc", "spekpy-v1", "casim", "kqp").
            mu_source: Source of photon coefficients ("pene", "nist").
            target: Anode target material ("W", "Mo", "Rh").
            z: Focus-to-detector distance.
            mas: Exposure setting in milli-Ampere-seconds.
        """
        self.kvp = kvp
        self.angle = angle
        self.dk = dk
        self.physics = physics
        self.mu_source = mu_source
        self.target = target
        self.z = z
        self.mas = mas
        self.default_filters = [('Be', 0.127), ('Air', 10.0)]

    def generate_primary_spectrum(
        self, 
        filters: Optional[List[Tuple[str, float]]] = None,
        kvp: Optional[float] = None,
        angle: Optional[float] = None,
        mas: Optional[float] = None,
        target: Optional[str] = None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Generates the primary X-ray tube spectrum.
        
        Args:
            filters: List of tuples (element_name, thickness_mm) for filtration.
        
        Returns:
            A tuple of (primary_spectrum, bremsstrahlung_spectrum), where each is a tuple of (energies, probabilities).
        """
        if filters is None:
            filters = self.default_filters
            
        with suppress_stdout():
            return multiel_spectra.Primary_gen(
                k=kvp if kvp is not None else self.kvp,
                theta=angle if angle is not None else self.angle,
                d=self.dk,
                phys=self.physics,
                mu_source=self.mu_source,
                z=self.z,
                mas=mas if mas is not None else self.mas,
                target=target if target is not None else self.target,
                filters=filters
            )

    def simulate_xrf_spectrum(
        self, 
        elements: Union[str, List[str]], 
        s_counts: int = 30000, 
        n_counts: int = 2000,
        b_counts: int = 3000,
        c_counts: int = 3000,
        plot: bool = False,
        escape: bool = True,
        sum_peaks: bool = True,
        decal: bool = True,
        filters: Optional[List[Tuple[str, float]]] = None,
        kvp: Optional[float] = None,
        angle: Optional[float] = None,
        mas: Optional[float] = None,
        target: Optional[str] = None
    ) -> Union[Tuple[np.ndarray, Dict, List], Tuple[np.ndarray, Dict, List, Dict]]:
        """
        Simulates an XRF spectrum for a given sample composition.
        
        Args:
            elements: String or list of elements in the sample.
            s_counts: Number of counts in the main spectrum.
            n_counts: Number of counts in the noise spectrum.
            b_counts: Number of counts in the bremsstrahlung spectrum.
            c_counts: Number of counts in the characteristic spectrum.
            plot: Whether to generate a Bokeh plot (if in a notebook).
            escape: Whether to apply escape peak correction.
            sum_peaks: Whether to apply sum peak correction.
            decal: Whether to apply decalibration (returns extra params).
            filters: Custom filters for the primary source.
            kvp: Tube potential in keV override.
            angle: Anode angle in degrees override.
            mas: Exposure setting in milli-Ampere-seconds override.
            target: Anode target material override.
            
        Returns:
            Depends on decal parameter:
            - If decal=False: (spectrum, peaks, elements)
            - If decal=True: (spectrum, peaks, elements, decal_params)
        """
        Prim, brems = self.generate_primary_spectrum(
            filters=filters, kvp=kvp, angle=angle, mas=mas, target=target
        )
        
        with suppress_stdout():
            return multiel_spectra.spectra_gen(
                a=elements,
                Prim=Prim,
                brems=brems,
                s_counts=s_counts,
                n_counts=n_counts,
                b_counts=b_counts,
                c_counts=c_counts,
                plot=plot,
                escape=escape,
                sum=sum_peaks,
                decal=decal
            )
