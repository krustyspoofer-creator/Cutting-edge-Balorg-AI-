"""Data augmentation utilities for Balorg AI."""

from typing import Optional, List, Callable
import numpy as np


class DataAugmentor:
    """
    Data augmentation class for increasing dataset diversity.
    
    This class provides various augmentation techniques to help
    models generalize better and prevent overfitting.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize data augmentor.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.augmentation_pipeline: List[Callable] = []
    
    def add_augmentation(self, augmentation_fn: Callable):
        """
        Add an augmentation function to the pipeline.
        
        Args:
            augmentation_fn: Function that takes data and returns augmented data
        """
        self.augmentation_pipeline.append(augmentation_fn)
    
    def augment(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> tuple:
        """
        Apply all augmentations in the pipeline.
        
        Args:
            x: Input data
            y: Optional target data
            
        Returns:
            Tuple of (augmented_x, y)
        """
        augmented_x = x.copy()
        
        for augmentation_fn in self.augmentation_pipeline:
            augmented_x = augmentation_fn(augmented_x)
        
        return augmented_x, y
    
    def augment_batch(self, x_batch: np.ndarray, 
                      y_batch: Optional[np.ndarray] = None) -> tuple:
        """
        Augment a batch of data.
        
        Args:
            x_batch: Batch of input data
            y_batch: Optional batch of target data
            
        Returns:
            Tuple of (augmented_x_batch, y_batch)
        """
        augmented_batch = []
        
        for i in range(len(x_batch)):
            augmented_x, _ = self.augment(x_batch[i])
            augmented_batch.append(augmented_x)
        
        augmented_x_batch = np.array(augmented_batch)
        return augmented_x_batch, y_batch


# Specific augmentation functions
def random_flip_horizontal(x: np.ndarray, probability: float = 0.5) -> np.ndarray:
    """
    Randomly flip images horizontally.
    
    Args:
        x: Input images (assumes last axis is channels)
        probability: Probability of flipping
        
    Returns:
        Flipped or original images
    """
    if np.random.random() < probability:
        return np.flip(x, axis=-2)  # Flip horizontal axis
    return x


def random_flip_vertical(x: np.ndarray, probability: float = 0.5) -> np.ndarray:
    """
    Randomly flip images vertically.
    
    Args:
        x: Input images
        probability: Probability of flipping
        
    Returns:
        Flipped or original images
    """
    if np.random.random() < probability:
        return np.flip(x, axis=-3)  # Flip vertical axis
    return x


def random_rotation(x: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Randomly rotate images.
    
    Args:
        x: Input images
        max_angle: Maximum rotation angle in degrees
        
    Returns:
        Rotated images (placeholder implementation)
    """
    # Placeholder - full implementation would use proper rotation
    angle = np.random.uniform(-max_angle, max_angle)
    # In practice, would apply rotation matrix
    return x


def random_crop(x: np.ndarray, crop_size: tuple, padding: int = 4) -> np.ndarray:
    """
    Randomly crop images.
    
    Args:
        x: Input images
        crop_size: Size of crop (height, width)
        padding: Padding to add before cropping
        
    Returns:
        Cropped images (placeholder implementation)
    """
    # Placeholder - full implementation would properly crop images
    return x


def random_brightness(x: np.ndarray, max_delta: float = 0.2) -> np.ndarray:
    """
    Randomly adjust image brightness.
    
    Args:
        x: Input images
        max_delta: Maximum brightness adjustment
        
    Returns:
        Brightness-adjusted images
    """
    delta = np.random.uniform(-max_delta, max_delta)
    return np.clip(x + delta, 0, 1)


def random_contrast(x: np.ndarray, lower: float = 0.8, upper: float = 1.2) -> np.ndarray:
    """
    Randomly adjust image contrast.
    
    Args:
        x: Input images
        lower: Lower bound for contrast factor
        upper: Upper bound for contrast factor
        
    Returns:
        Contrast-adjusted images
    """
    factor = np.random.uniform(lower, upper)
    mean = np.mean(x)
    return np.clip((x - mean) * factor + mean, 0, 1)


def random_noise(x: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
    """
    Add random noise to images.
    
    Args:
        x: Input images
        noise_factor: Standard deviation of Gaussian noise
        
    Returns:
        Noisy images
    """
    noise = np.random.randn(*x.shape) * noise_factor
    return np.clip(x + noise, 0, 1)


def mixup(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray,
          alpha: float = 0.2) -> tuple:
    """
    Apply mixup augmentation.
    
    Args:
        x1: First batch of images
        x2: Second batch of images
        y1: First batch of labels
        y2: Second batch of labels
        alpha: Mixup hyperparameter
        
    Returns:
        Tuple of (mixed_x, mixed_y)
    """
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    
    return mixed_x, mixed_y


def cutout(x: np.ndarray, mask_size: int = 16) -> np.ndarray:
    """
    Apply cutout augmentation (randomly mask square regions).
    
    Args:
        x: Input images
        mask_size: Size of square mask
        
    Returns:
        Images with cutout applied
    """
    if len(x.shape) < 3:
        return x
    
    h, w = x.shape[-3:-1]
    y_pos = np.random.randint(0, h)
    x_pos = np.random.randint(0, w)
    
    y1 = np.clip(y_pos - mask_size // 2, 0, h)
    y2 = np.clip(y_pos + mask_size // 2, 0, h)
    x1 = np.clip(x_pos - mask_size // 2, 0, w)
    x2 = np.clip(x_pos + mask_size // 2, 0, w)
    
    x_cutout = x.copy()
    x_cutout[..., y1:y2, x1:x2, :] = 0
    
    return x_cutout
