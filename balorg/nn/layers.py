"""Layer implementations for neural networks."""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np
from balorg.nn.activations import get_activation


class Layer(ABC):
    """
    Abstract base class for neural network layers.
    """
    
    def __init__(self):
        """Initialize the layer."""
        self.input = None
        self.output = None
        
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            x: Input data
            
        Returns:
            Layer output
        """
        pass
    
    @abstractmethod
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.
        
        Args:
            gradient: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        pass


class DenseLayer(Layer):
    """
    Fully connected (dense) layer implementation.
    """
    
    def __init__(self, input_size: int, output_size: int,
                 activation: Optional[str] = None,
                 use_bias: bool = True):
        """
        Initialize the dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            activation: Activation function name
            use_bias: Whether to use bias term
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.01
        if self.use_bias:
            self.bias = np.zeros((1, output_size))
        else:
            self.bias = None
            
        # Get activation function
        self.activation = get_activation(activation) if activation else None
        self.activation_name = activation
        
        # For gradient computation
        self.weights_gradient = None
        self.bias_gradient = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the dense layer.
        
        Args:
            x: Input data of shape (batch_size, input_size)
            
        Returns:
            Output of shape (batch_size, output_size)
        """
        self.input = x
        
        # Linear transformation
        output = np.dot(x, self.weights)
        if self.use_bias:
            output += self.bias
            
        # Apply activation function if specified
        if self.activation:
            output = self.activation(output)
            
        self.output = output
        return output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through the dense layer.
        
        Args:
            gradient: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        # If activation function is used, apply its derivative
        if self.activation:
            # Simplified - in practice would need activation derivative
            gradient = gradient
            
        # Compute gradients
        self.weights_gradient = np.dot(self.input.T, gradient)
        if self.use_bias:
            self.bias_gradient = np.sum(gradient, axis=0, keepdims=True)
            
        # Gradient with respect to input
        input_gradient = np.dot(gradient, self.weights.T)
        
        return input_gradient
    
    def get_num_parameters(self) -> int:
        """
        Get the number of parameters in this layer.
        
        Returns:
            Total number of parameters
        """
        params = self.input_size * self.output_size
        if self.use_bias:
            params += self.output_size
        return params
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return (f"DenseLayer(input_size={self.input_size}, "
                f"output_size={self.output_size}, "
                f"activation={self.activation_name})")


class DropoutLayer(Layer):
    """
    Dropout layer for regularization.
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        """
        Initialize the dropout layer.
        
        Args:
            dropout_rate: Fraction of inputs to drop (0 to 1)
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with dropout.
        
        Args:
            x: Input data
            
        Returns:
            Output with dropout applied
        """
        self.input = x
        
        if self.training:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
            # Apply mask and scale
            output = x * self.mask / (1 - self.dropout_rate)
        else:
            output = x
            
        self.output = output
        return output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through dropout.
        
        Args:
            gradient: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        if self.training and self.mask is not None:
            return gradient * self.mask / (1 - self.dropout_rate)
        return gradient
    
    def get_num_parameters(self) -> int:
        """Dropout has no parameters."""
        return 0


class BatchNormLayer(Layer):
    """
    Batch normalization layer.
    """
    
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.9):
        """
        Initialize batch normalization layer.
        
        Args:
            num_features: Number of features in input
            epsilon: Small constant for numerical stability
            momentum: Momentum for moving average
        """
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        # Running statistics
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        self.training = True
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through batch normalization.
        
        Args:
            x: Input data
            
        Returns:
            Normalized output
        """
        self.input = x
        
        if self.training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean + 
                                (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var + 
                               (1 - self.momentum) * batch_var)
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            # Use running statistics
            x_norm = ((x - self.running_mean) / 
                     np.sqrt(self.running_var + self.epsilon))
        
        # Scale and shift
        output = self.gamma * x_norm + self.beta
        self.output = output
        return output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through batch normalization.
        
        Args:
            gradient: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        # Simplified backward pass
        # Full implementation would compute proper gradients
        return gradient
    
    def get_num_parameters(self) -> int:
        """Get number of parameters (gamma and beta)."""
        return 2 * self.num_features
