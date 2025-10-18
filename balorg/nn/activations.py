"""Activation functions for neural networks."""

from typing import Callable, Optional
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit activation function.
    
    Args:
        x: Input array
        
    Returns:
        ReLU activation applied to input
    """
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid activation applied to input
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation function.
    
    Args:
        x: Input array
        
    Returns:
        Tanh activation applied to input
    """
    return np.tanh(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax activation function.
    
    Args:
        x: Input array
        axis: Axis along which to apply softmax
        
    Returns:
        Softmax activation applied to input
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU activation function.
    
    Args:
        x: Input array
        alpha: Slope for negative values
        
    Returns:
        Leaky ReLU activation applied to input
    """
    return np.where(x > 0, x, alpha * x)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Exponential Linear Unit activation function.
    
    Args:
        x: Input array
        alpha: Scale for negative values
        
    Returns:
        ELU activation applied to input
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def swish(x: np.ndarray) -> np.ndarray:
    """
    Swish activation function (x * sigmoid(x)).
    
    Args:
        x: Input array
        
    Returns:
        Swish activation applied to input
    """
    return x * sigmoid(x)


def get_activation(name: Optional[str]) -> Optional[Callable]:
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function
    """
    if name is None:
        return None
        
    activations = {
        'relu': relu,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'softmax': softmax,
        'leaky_relu': leaky_relu,
        'elu': elu,
        'swish': swish
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activations[name.lower()]


# Activation derivatives (for backpropagation)
def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return np.where(x > 0, 1, 0)


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh."""
    return 1 - np.tanh(x) ** 2
