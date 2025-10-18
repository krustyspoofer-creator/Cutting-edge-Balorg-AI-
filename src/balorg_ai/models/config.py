"""
Model configuration for Balorg AI system.

This module defines the configuration parameters for the transformer-based
architecture, including model size, complexity, and optimization settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class ModelConfig:
    """
    Configuration class for Balorg AI transformer model.
    
    Balances model size and complexity with computational resources and training time.
    """
    
    # Model Architecture
    vocab_size: int = 50000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    
    # Dropout and Regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Activation Function
    hidden_act: str = "gelu"
    
    # Model Type
    model_type: str = "balorg_transformer"
    
    # Multimodal Support
    use_vision: bool = False
    use_audio: bool = False
    vision_hidden_size: int = 768
    audio_hidden_size: int = 768
    
    # Training Configuration
    initializer_range: float = 0.02
    use_cache: bool = True
    
    # Advanced Features
    gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
    
    # Additional Parameters
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_json_file(cls, json_file: str) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_json_file(self, json_file: str):
        """Save configuration to JSON file."""
        with open(json_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class LargeModelConfig(ModelConfig):
    """Configuration for large-scale language model."""
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 4096


@dataclass
class MultimodalConfig(ModelConfig):
    """Configuration for multimodal learning model."""
    use_vision: bool = True
    use_audio: bool = True
    vision_hidden_size: int = 768
    audio_hidden_size: int = 768
