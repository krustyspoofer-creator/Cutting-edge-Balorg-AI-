"""Tests for data preprocessing tools."""

import pytest
import numpy as np
from balorg.preprocessing import (
    DataLoader,
    normalize,
    standardize,
    one_hot_encode,
    train_test_split,
    DataAugmentor
)
from balorg.preprocessing.transforms import Scaler
from balorg.preprocessing.augmentation import (
    random_flip_horizontal,
    random_brightness,
    random_noise
)


def test_data_loader_initialization():
    """Test data loader initialization."""
    loader = DataLoader(batch_size=32, shuffle=True)
    
    assert loader.batch_size == 32
    assert loader.shuffle == True


def test_load_dataset():
    """Test loading dataset."""
    loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = loader.load_dataset('mnist')
    
    assert x_train.shape[0] > 0
    assert y_train.shape[0] > 0
    assert x_test.shape[0] > 0
    assert y_test.shape[0] > 0


def test_create_batches():
    """Test creating batches."""
    loader = DataLoader(batch_size=10)
    x = np.random.randn(50, 10)
    y = np.random.randn(50, 1)
    
    batches = loader.create_batches(x, y)
    
    assert len(batches) == 5  # 50 samples / 10 batch_size
    assert batches[0][0].shape == (10, 10)


def test_normalize():
    """Test data normalization."""
    x = np.array([0, 5, 10])
    normalized = normalize(x, min_val=0, max_val=1)
    
    assert np.min(normalized) >= 0
    assert np.max(normalized) <= 1


def test_standardize():
    """Test data standardization."""
    x = np.array([1, 2, 3, 4, 5])
    standardized = standardize(x)
    
    assert np.abs(np.mean(standardized)) < 1e-6
    assert np.abs(np.std(standardized) - 1.0) < 1e-6


def test_one_hot_encode():
    """Test one-hot encoding."""
    y = np.array([0, 1, 2, 1])
    one_hot = one_hot_encode(y, num_classes=3)
    
    assert one_hot.shape == (4, 3)
    assert np.allclose(one_hot[0], [1, 0, 0])
    assert np.allclose(one_hot[1], [0, 1, 0])


def test_train_test_split():
    """Test train-test split."""
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 1)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    assert x_train.shape[0] == 80
    assert x_test.shape[0] == 20
    assert y_train.shape[0] == 80
    assert y_test.shape[0] == 20


def test_scaler_standard():
    """Test standard scaler."""
    scaler = Scaler(method='standard')
    x = np.array([[1, 2], [3, 4], [5, 6]])
    
    scaled = scaler.fit_transform(x)
    assert scaler.is_fitted
    
    # Check standardization
    assert np.abs(np.mean(scaled)) < 1e-6


def test_scaler_minmax():
    """Test min-max scaler."""
    scaler = Scaler(method='minmax')
    x = np.array([[1, 2], [3, 4], [5, 6]])
    
    scaled = scaler.fit_transform(x)
    assert scaler.is_fitted
    
    # Check range
    assert np.min(scaled) >= 0
    assert np.max(scaled) <= 1


def test_scaler_inverse_transform():
    """Test inverse transform."""
    scaler = Scaler(method='standard')
    x = np.array([[1, 2], [3, 4], [5, 6]])
    
    scaled = scaler.fit_transform(x)
    reconstructed = scaler.inverse_transform(scaled)
    
    assert np.allclose(x, reconstructed)


def test_data_augmentor_initialization():
    """Test data augmentor initialization."""
    augmentor = DataAugmentor(seed=42)
    assert augmentor.seed == 42
    assert len(augmentor.augmentation_pipeline) == 0


def test_data_augmentor_add_augmentation():
    """Test adding augmentation to pipeline."""
    augmentor = DataAugmentor()
    augmentor.add_augmentation(random_brightness)
    
    assert len(augmentor.augmentation_pipeline) == 1


def test_data_augmentor_augment():
    """Test data augmentation."""
    augmentor = DataAugmentor(seed=42)
    x = np.random.randn(10, 10)
    
    augmented_x, _ = augmentor.augment(x)
    assert augmented_x.shape == x.shape


def test_random_flip_horizontal():
    """Test random horizontal flip."""
    x = np.random.randn(32, 32, 3)
    flipped = random_flip_horizontal(x, probability=1.0)
    
    assert flipped.shape == x.shape


def test_random_brightness():
    """Test random brightness adjustment."""
    x = np.random.rand(32, 32, 3) * 0.5
    adjusted = random_brightness(x, max_delta=0.1)
    
    assert adjusted.shape == x.shape
    assert np.min(adjusted) >= 0
    assert np.max(adjusted) <= 1


def test_random_noise():
    """Test adding random noise."""
    x = np.random.rand(32, 32, 3)
    noisy = random_noise(x, noise_factor=0.05)
    
    assert noisy.shape == x.shape
