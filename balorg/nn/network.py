"""Neural Network implementation for Balorg AI."""

from typing import List, Optional
import numpy as np
from balorg.framework.model import Model
from balorg.nn.layers import Layer


class NeuralNetwork(Model):
    """
    Neural Network model implementation.
    
    This class provides a sequential neural network implementation
    that can stack multiple layers to create deep learning models.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the neural network.
        
        Args:
            name: Optional name for the network
        """
        super().__init__(name=name)
        self.layers: List[Layer] = []
        
    def add_layer(self, layer: Layer):
        """
        Add a layer to the network.
        
        Args:
            layer: Layer instance to add
        """
        self.layers.append(layer)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            x: Input data
            
        Returns:
            Network output
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through all layers.
        
        Args:
            gradient: Gradient from loss function
            
        Returns:
            Gradient with respect to input
        """
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    
    def summary(self):
        """Print a summary of the network architecture."""
        print(f"Neural Network: {self.name}")
        print("=" * 60)
        print(f"{'Layer':<20} {'Type':<20} {'Parameters':<20}")
        print("=" * 60)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_type = layer.__class__.__name__
            params = layer.get_num_parameters() if hasattr(layer, 'get_num_parameters') else 0
            total_params += params
            print(f"Layer {i:<14} {layer_type:<20} {params:<20}")
        
        print("=" * 60)
        print(f"Total parameters: {total_params}")
        print("=" * 60)
        
    def get_weights(self) -> List[np.ndarray]:
        """
        Get all weights from the network.
        
        Returns:
            List of weight arrays
        """
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                weights.append(layer.weights)
        return weights
    
    def set_weights(self, weights: List[np.ndarray]):
        """
        Set all weights in the network.
        
        Args:
            weights: List of weight arrays
        """
        weight_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = weights[weight_idx]
                weight_idx += 1
