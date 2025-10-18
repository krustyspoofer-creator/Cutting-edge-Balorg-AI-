"""Transformer-based model architecture for Balorg AI.

This module implements a state-of-the-art transformer architecture
optimized for multiple tasks including conversation, content generation,
and question answering.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class BalorgConfig:
    """Configuration class for Balorg Transformer model.
    
    Attributes:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of hidden layers
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: Dimension of feed-forward network
        max_position_embeddings: Maximum sequence length
        dropout_prob: Dropout probability
        layer_norm_eps: Layer normalization epsilon
        use_cache: Whether to use key-value caching
    """
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: BalorgConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        assert self.head_dim * config.num_heads == config.hidden_size, \
            "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout_prob)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """Forward pass of multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Mask tensor of shape (batch, 1, 1, seq_len)
            past_key_value: Cached key and value from previous forward pass
            use_cache: Whether to return key-value cache
            
        Returns:
            Tuple of output tensor and optional cache
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle cached keys/values
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        cache = (k, v) if use_cache else None
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.output(context)
        
        return output, cache


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config: BalorgConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single transformer layer with attention and feed-forward."""
    
    def __init__(self, config: BalorgConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, cache = self.attention(
            hidden_states, attention_mask, past_key_value, use_cache
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)
        
        return hidden_states, cache


class BalorgTransformer(nn.Module):
    """Main Balorg Transformer model.
    
    This is a state-of-the-art transformer-based language model designed
    for conversational AI, content generation, and question answering.
    """
    
    def __init__(self, config: BalorgConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embeddings and output projection
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            past_key_values: Cached key-values from previous forward passes
            use_cache: Whether to return key-value caches
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing logits and optional cached values
        """
        batch_size, seq_len = input_ids.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Get position IDs
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        
        position_ids = torch.arange(
            past_length, seq_len + past_length,
            dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert attention mask to causal mask
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Apply transformer layers
        new_cache = () if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, cache = layer(
                hidden_states, attention_mask, past_kv, use_cache
            )
            if use_cache:
                new_cache = new_cache + (cache,)
        
        # Final layer norm and projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            return (logits, new_cache) if use_cache else logits
        
        return {
            "logits": logits,
            "past_key_values": new_cache,
            "hidden_states": hidden_states,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Generate text using the model.
        
        Args:
            input_ids: Starting token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated token IDs
        """
        self.eval()
        device = input_ids.device
        generated = input_ids
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model outputs
                outputs = self.forward(
                    generated if past_key_values is None else generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(generated[0].tolist()):
                        logits[0, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == self.config.eos_token_id:
                    break
        
        return generated


def create_balorg_model(
    vocab_size: int = 50257,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
) -> BalorgTransformer:
    """Factory function to create a Balorg Transformer model.
    
    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        
    Returns:
        Initialized BalorgTransformer model
    """
    config = BalorgConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=hidden_size * 4,
    )
    return BalorgTransformer(config)
