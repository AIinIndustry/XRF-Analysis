import numpy as np

class BaseScaler:
    def fit(self, data: np.ndarray):
        raise NotImplementedError
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

class StandardScaler(BaseScaler):
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        # Avoid division by zero
        self.std[self.std == 0] = 1e-8
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean

class MinMaxScaler(BaseScaler):
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.range_min, self.range_max = feature_range
        
    def fit(self, data: np.ndarray):
        self.min = np.min(data, axis=0, keepdims=True)
        self.max = np.max(data, axis=0, keepdims=True)
        self.range = self.max - self.min
        self.range[self.range == 0] = 1e-8
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        scaled = (data - self.min) / self.range
        return scaled * (self.range_max - self.range_min) + self.range_min
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        scaled = (data - self.range_min) / (self.range_max - self.range_min)
        return scaled * self.range + self.min

class LogMinMaxScaler(BaseScaler):
    """
    Applies log1p to compress dynamic range, then MinMax scaling.
    Excellent for XRF spectra with very large peaks and small backgrounds.
    """
    def __init__(self, feature_range=(0, 1)):
        self.minmax_scaler = MinMaxScaler(feature_range)
        
    def fit(self, data: np.ndarray):
        log_data = np.log1p(data)
        self.minmax_scaler.fit(log_data)
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        log_data = np.log1p(data)
        return self.minmax_scaler.transform(log_data)
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        log_data = self.minmax_scaler.inverse_transform(data)
        # expm1 handles exp(x) - 1, inverse of log1p
        return np.expm1(log_data)
