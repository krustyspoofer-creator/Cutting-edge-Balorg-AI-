"""Example: Content generation with Balorg AI.

This example demonstrates various content generation capabilities
including articles, stories, dialogue, and more.
"""

from balorg_ai.models.transformer import create_balorg_model
from balorg_ai.inference.content_generator import ContentGenerator


def main():
    """Main function demonstrating content generation."""
    print("=" * 50)
    print("Balorg AI - Content Generation Demo")
    print("=" * 50)
    
    # Create model
    print("\n1. Creating model...")
    model = create_balorg_model(
        vocab_size=50257,
        hidden_size=384,
        num_layers=6,
        num_heads=6,
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create content generator
    print("\n2. Initializing content generator...")
    generator = ContentGenerator(model)
    
    print("\n" + "=" * 50)
    print("Content Generation Examples")
    print("=" * 50)
    
    # Example 1: Generate an article
    print("\n" + "-" * 50)
    print("Example 1: Article Generation")
    print("-" * 50)
    
    article = generator.generate_article(
        topic="the future of artificial intelligence",
        max_length=300,
        temperature=0.8,
        style="informative"
    )
    print(f"\nGenerated Article:\n{article}")
    
    # Example 2: Generate a story
    print("\n" + "-" * 50)
    print("Example 2: Story Generation")
    print("-" * 50)
    
    story = generator.generate_story(
        premise="A scientist discovers a way to communicate with AI",
        max_length=400,
        temperature=1.0,
        genre="science fiction"
    )
    print(f"\nGenerated Story:\n{story}")
    
    # Example 3: Generate dialogue
    print("\n" + "-" * 50)
    print("Example 3: Dialogue Generation")
    print("-" * 50)
    
    dialogue = generator.generate_dialogue(
        characters=["Alice", "Bob"],
        situation="discussing the capabilities of AI systems",
        num_turns=6,
        temperature=0.9
    )
    print(f"\nGenerated Dialogue:\n{dialogue}")
    
    # Example 4: Generate code
    print("\n" + "-" * 50)
    print("Example 4: Code Generation")
    print("-" * 50)
    
    code = generator.generate_code(
        description="implements a simple neural network layer",
        language="python",
        max_length=200,
        temperature=0.7
    )
    print(f"\nGenerated Code:\n{code}")
    
    # Example 5: Text continuation
    print("\n" + "-" * 50)
    print("Example 5: Text Continuation")
    print("-" * 50)
    
    start_text = "The rapid advancement of AI technology has led to"
    continuations = generator.continue_text(
        text=start_text,
        max_length=100,
        temperature=0.9,
        num_continuations=3
    )
    
    print(f"\nOriginal text: {start_text}")
    print("\nContinuations:")
    for i, cont in enumerate(continuations, 1):
        print(f"\n{i}. {cont}")
    
    # Example 6: Batch generation
    print("\n" + "-" * 50)
    print("Example 6: Batch Generation")
    print("-" * 50)
    
    prompts = [
        "Write a haiku about technology:",
        "Describe the color blue:",
        "Explain machine learning in one sentence:",
    ]
    
    results = generator.batch_generate(prompts, max_length=50, temperature=0.8)
    
    print("\nBatch Generation Results:")
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Result: {result}")
    
    print("\n" + "=" * 50)
    print("Content generation demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
