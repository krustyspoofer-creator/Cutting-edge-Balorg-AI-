"""
Example script for content generation with Balorg AI.

Demonstrates various content generation capabilities.
"""

from balorg_ai.applications import ContentGenerator
from balorg_ai.models import BalorgTransformer, ModelConfig


def main():
    print("Balorg AI Content Generation Example")
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
    
    # Dummy tokenizer
    class DummyTokenizer:
        def encode(self, text, return_tensors):
            import torch
            return torch.randint(0, config.vocab_size, (1, 100))
        
        def decode(self, tokens, skip_special_tokens=False):
            return "Generated content based on your prompt..."
    
    tokenizer = DummyTokenizer()
    
    # Create content generator
    generator = ContentGenerator(
        model=model,
        tokenizer=tokenizer,
    )
    
    print("\nContent Generator initialized!\n")
    
    # Generate article
    print("1. Generating Article")
    print("-" * 50)
    article = generator.generate_article(
        title="The Future of Artificial Intelligence",
        keywords=["AI", "machine learning", "deep learning"],
        max_length=500,
    )
    print(f"Article: {article}\n")
    
    # Generate story
    print("2. Generating Story")
    print("-" * 50)
    story = generator.generate_story(
        genre="science fiction",
        characters=["Alex", "Robot Assistant"],
        setting="A futuristic city in 2150",
        max_length=800,
    )
    print(f"Story: {story}\n")
    
    # Generate dialogue
    print("3. Generating Dialogue")
    print("-" * 50)
    dialogue = generator.generate_dialogue(
        context="Two scientists discussing a breakthrough discovery",
        num_turns=5,
        max_length=300,
    )
    print(f"Dialogue: {dialogue}\n")
    
    print("=" * 50)
    print("Content generation examples completed!")


if __name__ == "__main__":
    main()
