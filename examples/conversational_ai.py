"""
Example script for using Balorg AI conversational system.

Demonstrates how to use the conversational AI for interactive chat.
"""

from balorg_ai.applications import ConversationalAI
from balorg_ai.models import BalorgTransformer, ModelConfig


def main():
    print("Balorg AI Conversational Example")
    print("=" * 50)
    
    # Create model configuration
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    )
    
    # Create model
    model = BalorgTransformer(config)
    print("Model created successfully!")
    
    # Dummy tokenizer (replace with actual tokenizer)
    class DummyTokenizer:
        def encode(self, text, return_tensors):
            import torch
            return torch.randint(0, config.vocab_size, (1, 100))
        
        def decode(self, tokens, skip_special_tokens=False):
            # Return sample responses
            responses = [
                "Hello! I'm Balorg AI. How can I help you today?",
                "That's an interesting question! Let me think about it.",
                "I understand. Can you tell me more about that?",
                "Thank you for sharing! I appreciate your input.",
                "That's a great point. Here's what I think...",
            ]
            import random
            return random.choice(responses)
    
    tokenizer = DummyTokenizer()
    
    # Create conversational AI
    conv_ai = ConversationalAI(
        model=model,
        tokenizer=tokenizer,
        max_history=10,
        max_length=100,
        temperature=0.8,
    )
    
    print("\nConversational AI initialized!")
    print("\nTry asking questions or having a conversation.")
    print("Type 'exit', 'quit', or 'bye' to end.\n")
    
    # Start interactive chat
    conv_ai.chat()


if __name__ == "__main__":
    main()
