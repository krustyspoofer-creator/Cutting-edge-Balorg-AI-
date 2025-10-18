"""Configuration management for Balorg AI Framework."""

from typing import Any, Dict, Optional
import json


class Config:
    """
    Configuration manager for Balorg AI Framework.
    
    This class provides a centralized way to manage configuration
    parameters for models, training, and other framework components.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize configuration.
        
        Args:
            **kwargs: Configuration parameters as keyword arguments
        """
        self._config = {}
        self.update(**kwargs)
    
    def update(self, **kwargs):
        """
        Update configuration with new parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self._config.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return self._config.copy()
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Convert configuration to JSON.
        
        Args:
            filepath: Optional path to save JSON file
            
        Returns:
            JSON string representation of configuration
        """
        json_str = json.dumps(self._config, indent=2)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            Config instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """
        Create configuration from JSON file.
        
        Args:
            filepath: Path to JSON configuration file
            
        Returns:
            Config instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return json.dumps(self._config, indent=2)


# Default configurations
DEFAULT_TRAINING_CONFIG = Config(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

DEFAULT_MODEL_CONFIG = Config(
    name='balorg_model',
    verbose=1,
    seed=42
)
