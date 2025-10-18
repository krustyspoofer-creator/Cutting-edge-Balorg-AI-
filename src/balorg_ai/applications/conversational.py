"""
Conversational AI application for Balorg AI.

Implements a natural-sounding conversational AI system capable of engaging
in human-like conversations.
"""

import torch
from typing import List, Dict, Optional, Any
from ..models.transformer import BalorgTransformer
from ..models.config import ModelConfig


class ConversationalAI:
    """
    Conversational AI system using Balorg Transformer.
    
    Provides an interface for engaging in natural conversations with
    context awareness and coherent responses.
    """
    
    def __init__(
        self,
        model: BalorgTransformer,
        tokenizer: Any,
        max_history: int = 10,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize conversational AI.
        
        Args:
            model: Balorg Transformer model
            tokenizer: Tokenizer instance
            max_history: Maximum conversation history to maintain
            max_length: Maximum response length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = device
        
        # Conversation history
        self.history: List[Dict[str, str]] = []
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: User's message
            
        Returns:
            AI-generated response
        """
        # Add user input to history
        self.history.append({"role": "user", "content": user_input})
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Build context from history
        context = self._build_context()
        
        # Tokenize context
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=self.max_length,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Add response to history
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def _build_context(self) -> str:
        """Build conversation context from history."""
        context_parts = []
        for turn in self.history:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                context_parts.append(f"User: {content}")
            else:
                context_parts.append(f"Assistant: {content}")
        
        context_parts.append("Assistant:")
        return "\n".join(context_parts)
    
    def reset_history(self):
        """Reset conversation history."""
        self.history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history.copy()
    
    def chat(self):
        """
        Start an interactive chat session.
        
        Continues until user types 'exit', 'quit', or 'bye'.
        """
        print("Balorg AI Conversational System")
        print("Type 'exit', 'quit', or 'bye' to end the conversation")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nBalorg AI: Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue
            
            response = self.generate_response(user_input)
            print(f"\nBalorg AI: {response}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> "ConversationalAI":
        """
        Load conversational AI from pretrained model.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer: Tokenizer instance
            device: Device to run inference on
            **kwargs: Additional arguments for ConversationalAI
            
        Returns:
            ConversationalAI instance
        """
        # Load config
        config = ModelConfig.from_json_file(f"{model_path}/config.json")
        
        # Create model
        model = BalorgTransformer(config)
        
        # Load weights
        state_dict = torch.load(f"{model_path}/model.pt", map_location=device)
        model.load_state_dict(state_dict)
        
        return cls(model, tokenizer, device=device, **kwargs)
