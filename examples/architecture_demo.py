#!/usr/bin/env python3
"""
Example script demonstrating different model architectures
"""

from balorg_ai import BalorgAI


def demonstrate_transformer():
    """Demonstrate Transformer model"""
    print("\n" + "=" * 60)
    print("Transformer Model Example")
    print("=" * 60)
    
    # Create transformer model
    model = BalorgAI(model_type='transformer', config={
        'num_layers': 12,
        'hidden_size': 768,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'max_position_embeddings': 512
    })
    
    print(model.summary())
    
    print("\nKey Features:")
    print("  - Multi-head self-attention")
    print("  - Parallel sequence processing")
    print("  - Long-range dependency capture")
    print("  - State-of-the-art NLP performance")


def demonstrate_neural_network():
    """Demonstrate Neural Network model"""
    print("\n" + "=" * 60)
    print("Neural Network Example")
    print("=" * 60)
    
    # Create neural network model
    model = BalorgAI(model_type='neural_network', config={
        'layer_sizes': [1024, 512, 256, 128, 64],
        'activation': 'relu',
        'dropout_rate': 0.2
    })
    
    print(model.summary())
    
    print("\nKey Features:")
    print("  - Flexible layer configuration")
    print("  - Various activation functions")
    print("  - Dropout regularization")
    print("  - Transfer learning support")


def main():
    """Run all architecture demonstrations"""
    print("\n" + "=" * 60)
    print("Balorg AI - Model Architecture Demonstration")
    print("=" * 60)
    
    demonstrate_transformer()
    demonstrate_neural_network()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
