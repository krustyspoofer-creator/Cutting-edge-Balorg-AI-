"""
Example script for training Balorg AI model.

Demonstrates how to train a Balorg AI transformer model on text data.
"""

import torch
from torch.utils.data import DataLoader
from balorg_ai.models import BalorgTransformer, ModelConfig
from balorg_ai.training import BalorgTrainer
from balorg_ai.data import TextDataset, DataProcessor
from balorg_ai.utils import get_linear_schedule_with_warmup, get_parameter_groups


def main():
    # Configuration
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
        use_mixed_precision=True,
    )
    
    print("Balorg AI Training Example")
    print("=" * 50)
    print(f"Model Configuration:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention Heads: {config.num_attention_heads}")
    print("=" * 50)
    
    # Create model
    model = BalorgTransformer(config)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Dummy tokenizer (replace with actual tokenizer)
    class DummyTokenizer:
        def __call__(self, text, max_length, padding, truncation, return_tensors):
            # This is a placeholder - use a real tokenizer in production
            tokens = torch.randint(0, config.vocab_size, (1, max_length))
            return {
                'input_ids': tokens,
                'attention_mask': torch.ones_like(tokens)
            }
        
        def encode(self, text, return_tensors):
            return torch.randint(0, config.vocab_size, (1, 100))
        
        def decode(self, tokens, skip_special_tokens=False):
            return "Generated text"
    
    tokenizer = DummyTokenizer()
    
    # Create dummy dataset (replace with actual data)
    dummy_texts = [
        "This is a sample text for training the model.",
        "Balorg AI is a cutting-edge AI system.",
        "Deep learning with transformers is powerful.",
    ] * 100  # Repeat for demonstration
    
    dataset = TextDataset(
        texts=dummy_texts,
        tokenizer=tokenizer,
        max_length=512,
    )
    
    # Split dataset
    train_dataset, val_dataset, _ = DataProcessor.split_dataset(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    
    # Create dataloaders
    train_dataloader = DataProcessor.create_dataloader(
        train_dataset,
        batch_size=4,
        shuffle=True,
    )
    
    val_dataloader = DataProcessor.create_dataloader(
        val_dataset,
        batch_size=4,
        shuffle=False,
    )
    
    # Setup optimizer
    param_groups = get_parameter_groups(model, weight_decay=0.01)
    optimizer = torch.optim.AdamW(param_groups, lr=5e-5)
    
    # Setup scheduler
    num_training_steps = len(train_dataloader) * 3  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    
    # Create trainer
    trainer = BalorgTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir="./output/balorg_model",
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train(num_epochs=3)
    
    print("\nTraining completed!")
    print(f"Model saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
