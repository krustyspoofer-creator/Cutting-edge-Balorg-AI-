"""
Core Balorg AI module with architecture components
"""

from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TransformerModel:
    """
    Transformer-based model architecture for Balorg AI
    Supports attention mechanisms and multi-head attention
    """
    
    def __init__(self, 
                 num_layers: int = 12,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 max_position_embeddings: int = 512):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        
        logger.info(f"Initialized TransformerModel with {num_layers} layers")
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings
        }


class NeuralNetwork:
    """
    Neural network architecture for Balorg AI
    Supports various layer types and activation functions
    """
    
    def __init__(self,
                 layer_sizes: List[int],
                 activation: str = 'relu',
                 dropout_rate: float = 0.1):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        logger.info(f"Initialized NeuralNetwork with {len(layer_sizes)} layers")
    
    def get_config(self) -> Dict[str, Any]:
        """Return network configuration"""
        return {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate
        }


class BalorgAI:
    """
    Main Balorg AI class integrating various architectures and optimizations
    """
    
    def __init__(self, 
                 model_type: str = 'transformer',
                 config: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self._initialize_model()
        
        logger.info(f"Initialized BalorgAI with model type: {model_type}")
    
    def _initialize_model(self):
        """Initialize the model based on type"""
        if self.model_type == 'transformer':
            self.model = TransformerModel(**self.config)
        elif self.model_type == 'neural_network':
            self.model = NeuralNetwork(**self.config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_type': self.model_type,
            'config': self.model.get_config() if self.model else None
        }
    
    def summary(self) -> str:
        """Return a summary of the model architecture"""
        info = self.get_model_info()
        summary_lines = [
            "Balorg AI Model Summary",
            "=" * 50,
            f"Model Type: {info['model_type']}",
            "",
            "Configuration:"
        ]
        
        if info['config']:
            for key, value in info['config'].items():
                summary_lines.append(f"  {key}: {value}")
        
        return "\n".join(summary_lines)
