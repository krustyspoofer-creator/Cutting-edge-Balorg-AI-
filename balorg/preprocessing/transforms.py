"""Data transformation utilities for Balorg AI."""

from typing import Tuple, Optional
import numpy as np


def normalize(x: np.ndarray, axis: Optional[int] = None, 
              min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Normalize data to a specified range.
    
    Args:
        x: Input data
        axis: Axis along which to normalize (None for global)
        min_val: Minimum value for normalized data
        max_val: Maximum value for normalized data
        
    Returns:
        Normalized data
    """
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    
    # Avoid division by zero
    x_range = x_max - x_min
    x_range = np.where(x_range == 0, 1, x_range)
    
    normalized = (x - x_min) / x_range
    return normalized * (max_val - min_val) + min_val


def standardize(x: np.ndarray, axis: Optional[int] = None, 
                epsilon: float = 1e-8) -> np.ndarray:
    """
    Standardize data to zero mean and unit variance.
    
    Args:
        x: Input data
        axis: Axis along which to standardize (None for global)
        epsilon: Small constant for numerical stability
        
    Returns:
        Standardized data
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    
    return (x - mean) / (std + epsilon)


def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert class labels to one-hot encoded vectors.
    
    Args:
        y: Class labels (integers)
        num_classes: Number of classes (inferred if None)
        
    Returns:
        One-hot encoded array
    """
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    
    n_samples = len(y)
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y.astype(int)] = 1
    
    return one_hot


def train_test_split(x: np.ndarray, y: np.ndarray, 
                     test_size: float = 0.2, 
                     shuffle: bool = True,
                     random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        x: Input data
        y: Target data
        test_size: Fraction of data to use for testing
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (x_train, x_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(x)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def reshape_image_data(x: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reshape image data to target shape.
    
    Args:
        x: Input image data
        target_shape: Target shape for images
        
    Returns:
        Reshaped data
    """
    # Simple reshape - in practice would use proper image resizing
    return x.reshape((-1,) + target_shape)


def flatten(x: np.ndarray) -> np.ndarray:
    """
    Flatten multidimensional data.
    
    Args:
        x: Input data
        
    Returns:
        Flattened data
    """
    return x.reshape(x.shape[0], -1)


def add_noise(x: np.ndarray, noise_factor: float = 0.1, 
              noise_type: str = 'gaussian') -> np.ndarray:
    """
    Add noise to data for augmentation or regularization.
    
    Args:
        x: Input data
        noise_factor: Amount of noise to add
        noise_type: Type of noise ('gaussian' or 'uniform')
        
    Returns:
        Noisy data
    """
    if noise_type == 'gaussian':
        noise = np.random.randn(*x.shape) * noise_factor
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_factor, noise_factor, size=x.shape)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return x + noise


def clip_values(x: np.ndarray, min_val: Optional[float] = None, 
                max_val: Optional[float] = None) -> np.ndarray:
    """
    Clip values to a specified range.
    
    Args:
        x: Input data
        min_val: Minimum value (None for no lower bound)
        max_val: Maximum value (None for no upper bound)
        
    Returns:
        Clipped data
    """
    return np.clip(x, min_val, max_val)


class Scaler:
    """
    Scaler class for fitting and transforming data.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize scaler.
        
        Args:
            method: Scaling method ('standard' or 'minmax')
        """
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.is_fitted = False
    
    def fit(self, x: np.ndarray, axis: Optional[int] = None):
        """
        Fit scaler to data.
        
        Args:
            x: Input data
            axis: Axis along which to fit
        """
        if self.method == 'standard':
            self.mean = np.mean(x, axis=axis, keepdims=True)
            self.std = np.std(x, axis=axis, keepdims=True)
        elif self.method == 'minmax':
            self.min = np.min(x, axis=axis, keepdims=True)
            self.max = np.max(x, axis=axis, keepdims=True)
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        self.is_fitted = True
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            x: Input data
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        
        if self.method == 'standard':
            return (x - self.mean) / (self.std + 1e-8)
        elif self.method == 'minmax':
            x_range = self.max - self.min
            x_range = np.where(x_range == 0, 1, x_range)
            return (x - self.min) / x_range
    
    def fit_transform(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """
        Fit scaler and transform data in one step.
        
        Args:
            x: Input data
            axis: Axis along which to fit
            
        Returns:
            Transformed data
        """
        self.fit(x, axis=axis)
        return self.transform(x)
    
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            x: Scaled data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse transform")
        
        if self.method == 'standard':
            return x * self.std + self.mean
        elif self.method == 'minmax':
            x_range = self.max - self.min
            return x * x_range + self.min
