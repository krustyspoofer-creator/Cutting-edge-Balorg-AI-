"""Example: Training a Balorg AI model.

This example demonstrates how to train a Balorg AI transformer model
using the training utilities and sample data.
"""

import torch
from balorg_ai.models.transformer import create_balorg_model, BalorgConfig
from balorg_ai.training.trainer import Trainer, TrainingConfig
from balorg_ai.data.dataset import TextDataset, create_sample_dataset


def main():
    """Main training function."""
    print("=" * 50)
    print("Balorg AI - Training Example")
    print("=" * 50)
    
    # Create model
    print("\n1. Creating model...")
    config = BalorgConfig(
        vocab_size=50257,
        hidden_size=384,  # Smaller for faster training
        num_layers=6,
        num_heads=6,
        max_position_embeddings=512,
    )
    
    from balorg_ai.models.transformer import BalorgTransformer
    model = BalorgTransformer(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample dataset
    print("\n2. Creating dataset...")
    texts = create_sample_dataset(num_samples=200)
    train_texts = texts[:150]
    eval_texts = texts[150:]
    
    train_dataset = TextDataset(train_texts, max_length=128)
    eval_dataset = TextDataset(eval_texts, max_length=128)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Configure training
    print("\n3. Configuring training...")
    training_config = TrainingConfig(
        output_dir="./outputs/balorg_training",
        num_epochs=2,
        batch_size=4,
        learning_rate=3e-4,
        warmup_steps=100,
        save_steps=500,
        logging_steps=50,
        eval_steps=500,
        mixed_precision=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"Training on: {training_config.device}")
    print(f"Mixed precision: {training_config.mixed_precision}")
    
    # Create trainer
    print("\n4. Initializing trainer...")
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train model
    print("\n5. Starting training...")
    print("-" * 50)
    
    try:
        results = trainer.train()
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Total steps: {results['total_steps']}")
        print(f"Epochs: {results['epochs']}")
        print("=" * 50)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted")
    
    # Evaluate final model
    print("\n6. Final evaluation...")
    eval_metrics = trainer.evaluate()
    print("\nEvaluation metrics:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
