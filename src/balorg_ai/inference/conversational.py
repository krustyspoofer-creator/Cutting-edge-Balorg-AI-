"""Conversational AI interface for Balorg AI.

This module provides a user-friendly interface for conversational
interactions with Balorg AI models.
"""

import torch
from typing import List, Dict, Optional, Any
from balorg_ai.models.transformer import BalorgTransformer


class ConversationalAI:
    """Conversational AI system for natural dialogue.
    
    This class provides an interface for engaging in natural-sounding
    conversations with Balorg AI models.
    """
    
    def __init__(
        self,
        model: BalorgTransformer,
        tokenizer: Optional[Any] = None,
        max_history: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize conversational AI.
        
        Args:
            model: Balorg transformer model
            tokenizer: Tokenizer for encoding/decoding text
            max_history: Maximum conversation history to maintain
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.device = device
        
        # Conversation history
        self.history: List[Dict[str, str]] = []
    
    def chat(
        self,
        user_input: str,
        max_length: int = 100,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """Generate a response to user input.
        
        Args:
            user_input: User's message
            max_length: Maximum response length
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response
        """
        # Add user input to history
        self.history.append({"role": "user", "content": user_input})
        
        # Build prompt from history
        prompt = self._build_prompt()
        
        # Tokenize
        if self.tokenizer:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        else:
            # Simple tokenization
            tokens = prompt.lower().split()
            input_ids = torch.tensor([[hash(token) % 50000 for token in tokens]])
        
        input_ids = input_ids.to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        
        # Decode response
        if self.tokenizer:
            response = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
        else:
            # Simple decoding (placeholder)
            response = "This is a generated response from Balorg AI."
        
        # Add response to history
        self.history.append({"role": "assistant", "content": response})
        
        # Trim history if needed
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
        
        return response
    
    def _build_prompt(self) -> str:
        """Build prompt from conversation history.
        
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for turn in self.history[-self.max_history * 2:]:
            role = turn["role"]
            content = turn["content"]
            
            if role == "user":
                prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def reset(self):
        """Reset conversation history."""
        self.history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.history.copy()
    
    def set_system_prompt(self, system_prompt: str):
        """Set a system prompt for the conversation.
        
        Args:
            system_prompt: System-level instruction for the model
        """
        self.history.insert(0, {"role": "system", "content": system_prompt})


def create_conversational_ai(
    model: BalorgTransformer,
    system_prompt: Optional[str] = None,
) -> ConversationalAI:
    """Factory function to create a conversational AI instance.
    
    Args:
        model: Balorg transformer model
        system_prompt: Optional system prompt
        
    Returns:
        ConversationalAI instance
    """
    ai = ConversationalAI(model)
    
    if system_prompt:
        ai.set_system_prompt(system_prompt)
    
    return ai
