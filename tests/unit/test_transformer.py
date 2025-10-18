"""Unit tests for Balorg AI transformer model."""

import pytest
import torch
from balorg_ai.models.transformer import (
    BalorgConfig,
    BalorgTransformer,
    MultiHeadAttention,
    FeedForward,
    TransformerLayer,
    create_balorg_model,
)


class TestBalorgConfig:
    """Tests for BalorgConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BalorgConfig()
        assert config.vocab_size == 50257
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.intermediate_size == 3072
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BalorgConfig(
            vocab_size=10000,
            hidden_size=256,
            num_layers=6,
            num_heads=4,
        )
        assert config.vocab_size == 10000
        assert config.hidden_size == 256
        assert config.num_layers == 6
        assert config.num_heads == 4


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""
    
    def test_attention_forward(self):
        """Test forward pass of attention."""
        config = BalorgConfig(hidden_size=256, num_heads=4)
        attention = MultiHeadAttention(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output, cache = attention(hidden_states, use_cache=False)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert cache is None
    
    def test_attention_with_cache(self):
        """Test attention with caching."""
        config = BalorgConfig(hidden_size=256, num_heads=4)
        attention = MultiHeadAttention(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output, cache = attention(hidden_states, use_cache=True)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert cache is not None
        assert len(cache) == 2  # key and value


class TestFeedForward:
    """Tests for FeedForward module."""
    
    def test_feedforward_forward(self):
        """Test forward pass of feed-forward network."""
        config = BalorgConfig(hidden_size=256, intermediate_size=1024)
        ff = FeedForward(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = ff(hidden_states)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestTransformerLayer:
    """Tests for TransformerLayer module."""
    
    def test_layer_forward(self):
        """Test forward pass of transformer layer."""
        config = BalorgConfig(hidden_size=256, num_heads=4)
        layer = TransformerLayer(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output, cache = layer(hidden_states, use_cache=False)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestBalorgTransformer:
    """Tests for BalorgTransformer model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = BalorgConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
        )
        model = BalorgTransformer(config)
        
        assert len(model.layers) == config.num_layers
        assert model.token_embedding.num_embeddings == config.vocab_size
    
    def test_model_forward(self):
        """Test forward pass of model."""
        config = BalorgConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
        )
        model = BalorgTransformer(config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_model_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        config = BalorgConfig(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=4)
        model = BalorgTransformer(config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_model_generate(self):
        """Test text generation."""
        config = BalorgConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
        )
        model = BalorgTransformer(config)
        
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        generated = model.generate(input_ids, max_length=10, temperature=1.0)
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] > seq_len
        assert generated.shape[1] <= seq_len + 10
    
    def test_parameter_count(self):
        """Test that model has expected number of parameters."""
        config = BalorgConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
        )
        model = BalorgTransformer(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have parameters (rough estimate)
        assert total_params > 1_000_000
        assert total_params < 10_000_000


class TestFactoryFunction:
    """Tests for factory functions."""
    
    def test_create_balorg_model(self):
        """Test model creation factory function."""
        model = create_balorg_model(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
        )
        
        assert isinstance(model, BalorgTransformer)
        assert model.config.vocab_size == 1000
        assert model.config.hidden_size == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
