#!/usr/bin/env python3
"""Quick start script for Balorg AI.

This script provides a simple menu-driven interface to explore
Balorg AI capabilities without writing code.
"""

import sys


def print_header():
    """Print the header."""
    print("=" * 60)
    print(" " * 15 + "BALORG AI")
    print(" " * 10 + "Cutting-Edge Connectionist AI")
    print("=" * 60)


def print_menu():
    """Print the main menu."""
    print("\nWhat would you like to do?\n")
    print("1. Create a small model and test it")
    print("2. Run conversational AI demo")
    print("3. Run content generation demo")
    print("4. Run question answering demo")
    print("5. View model architecture info")
    print("6. Exit")
    print()


def create_and_test_model():
    """Create and test a small model."""
    print("\n" + "=" * 60)
    print("Creating a small Balorg AI model...")
    print("=" * 60)
    
    try:
        from balorg_ai.models.transformer import create_balorg_model
        
        model = create_balorg_model(
            vocab_size=10000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n✓ Model created successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Hidden size: 256")
        print(f"  Number of layers: 4")
        print(f"  Number of heads: 4")
        
        # Test forward pass
        import torch
        input_ids = torch.randint(0, 10000, (1, 10))
        outputs = model(input_ids)
        print(f"\n✓ Forward pass test successful!")
        print(f"  Output shape: {outputs['logits'].shape}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure dependencies are installed:")
        print("  pip install -r requirements.txt")
        return False


def run_conversational_demo():
    """Run conversational AI demo."""
    print("\n" + "=" * 60)
    print("Running Conversational AI Demo...")
    print("=" * 60)
    
    try:
        import subprocess
        subprocess.run([sys.executable, "examples/conversational_ai.py", "--demo"])
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def run_content_generation_demo():
    """Run content generation demo."""
    print("\n" + "=" * 60)
    print("Running Content Generation Demo...")
    print("=" * 60)
    
    try:
        import subprocess
        subprocess.run([sys.executable, "examples/content_generation.py"])
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def run_qa_demo():
    """Run question answering demo."""
    print("\n" + "=" * 60)
    print("Running Question Answering Demo...")
    print("=" * 60)
    
    try:
        import subprocess
        subprocess.run([sys.executable, "examples/question_answering.py"])
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def show_architecture_info():
    """Show model architecture information."""
    print("\n" + "=" * 60)
    print("Balorg AI Model Architecture")
    print("=" * 60)
    
    print("""
Balorg AI uses a transformer-based architecture with the following components:

1. Token Embedding Layer
   - Maps token IDs to dense vectors
   - Dimension: vocab_size → hidden_size

2. Position Embedding Layer
   - Encodes position information
   - Max sequence length: configurable (default 2048)

3. Transformer Layers (stacked)
   Each layer contains:
   - Multi-Head Self-Attention
     * Enables parallel attention to different positions
     * Number of heads: configurable (default 12)
   - Feed-Forward Network
     * Two linear layers with GELU activation
     * Intermediate size: 4x hidden size
   - Layer Normalization (applied before each sub-layer)
   - Residual Connections (for gradient flow)

4. Output Layer
   - Projects hidden states to vocabulary
   - Weight tying with token embeddings

Key Features:
• Mixed-precision training for faster computation
• Gradient accumulation for larger effective batch sizes
• Advanced text generation (top-k, nucleus sampling)
• Conversation history management
• Multiple inference modes (chat, QA, content generation)

Model Sizes:
┌────────┬────────┬─────────────┬───────┬────────────┐
│ Size   │ Layers │ Hidden Size │ Heads │ Parameters │
├────────┼────────┼─────────────┼───────┼────────────┤
│ Small  │ 6      │ 384         │ 6     │ ~40M       │
│ Base   │ 12     │ 768         │ 12    │ ~125M      │
│ Large  │ 24     │ 1024        │ 16    │ ~355M      │
└────────┴────────┴─────────────┴───────┴────────────┘
    """)


def main():
    """Main function."""
    print_header()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "1":
                create_and_test_model()
            elif choice == "2":
                run_conversational_demo()
            elif choice == "3":
                run_content_generation_demo()
            elif choice == "4":
                run_qa_demo()
            elif choice == "5":
                show_architecture_info()
            elif choice == "6":
                print("\nThank you for using Balorg AI!")
                print("Visit https://github.com/krustyspoofer-creator/Cutting-edge-Balorg-AI-")
                sys.exit(0)
            else:
                print("\n✗ Invalid choice. Please enter 1-6.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        except EOFError:
            print("\n\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    main()
