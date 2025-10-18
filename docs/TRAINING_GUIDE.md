# Training Guide for Balorg AI

This guide covers everything you need to know about training Balorg AI models.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create a text file with your training data (one example per line):

```
This is the first training example.
This is the second training example.
...
```

### 3. Run Training

```bash
python examples/train_model.py
```

## Detailed Training Process

### Step 1: Define Model Configuration

```python
from balorg_ai.models import ModelConfig

# Choose a configuration
config = ModelConfig(
    vocab_size=50000,           # Match your tokenizer
    hidden_size=768,            # Model dimension
    num_hidden_layers=12,       # Number of layers
    num_attention_heads=12,     # Attention heads
    max_position_embeddings=2048,  # Max sequence length
    use_mixed_precision=True,   # Enable mixed precision
)
```

### Step 2: Create Model

```python
from balorg_ai.models import BalorgTransformer

model = BalorgTransformer(config)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
```

### Step 3: Prepare Dataset

```python
from balorg_ai.data import TextDataset, DataProcessor

# Load data
texts = DataProcessor.load_text_data("data.txt")

# Create dataset
dataset = TextDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_length=512,
)

# Split into train/val/test
train_dataset, val_dataset, test_dataset = DataProcessor.split_dataset(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)

# Create dataloaders
train_loader = DataProcessor.create_dataloader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

val_loader = DataProcessor.create_dataloader(
    val_dataset,
    batch_size=8,
    shuffle=False,
)
```

### Step 4: Setup Optimizer and Scheduler

```python
from balorg_ai.utils import get_parameter_groups, get_linear_schedule_with_warmup
import torch.optim as optim

# Get parameter groups (different weight decay for different params)
param_groups = get_parameter_groups(model, weight_decay=0.01)

# Create optimizer
optimizer = optim.AdamW(
    param_groups,
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
)

# Create scheduler
num_training_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)
```

### Step 5: Create Trainer

```python
from balorg_ai.training import BalorgTrainer

trainer = BalorgTrainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    output_dir="./output/balorg_model",
    logging_steps=100,      # Log every 100 steps
    eval_steps=500,         # Evaluate every 500 steps
    save_steps=1000,        # Save checkpoint every 1000 steps
    max_grad_norm=1.0,      # Gradient clipping
    use_amp=True,           # Mixed precision training
)
```

### Step 6: Train Model

```python
# Train for multiple epochs
trainer.train(num_epochs=10)
```

## Advanced Training Techniques

### Distributed Training

For multi-GPU training:

```python
# Set up distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group("nccl")

# Wrap model with DDP
model = DDP(model)

# Train as usual
trainer.train(num_epochs=10)
```

### Gradient Accumulation

To simulate larger batch sizes:

```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    loss = compute_loss(model, batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

Automatically enabled in BalorgTrainer with `use_amp=True`:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        loss = compute_loss(model, batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Gradient Checkpointing

To reduce memory usage:

```python
from balorg_ai.utils import enable_gradient_checkpointing

# Enable gradient checkpointing
enable_gradient_checkpointing(model)

# Train as usual
trainer.train(num_epochs=10)
```

## Hyperparameter Tuning

### Learning Rate

- **Small models**: 1e-4 to 5e-4
- **Large models**: 1e-5 to 5e-5
- **Fine-tuning**: 1e-5 to 1e-4

### Batch Size

- Limited by GPU memory
- Larger is generally better
- Use gradient accumulation if needed

### Warmup Steps

- Typically 5-10% of total steps
- Helps stabilize early training
- Linear warmup is most common

### Weight Decay

- Prevents overfitting
- Typical values: 0.01 to 0.1
- Not applied to biases and layer norms

### Gradient Clipping

- Prevents gradient explosion
- Typical values: 0.5 to 1.0
- Clip by norm, not by value

## Monitoring Training

### Metrics to Track

1. **Training Loss**: Should decrease over time
2. **Validation Loss**: Should decrease, watch for overfitting
3. **Perplexity**: exp(loss), lower is better
4. **Learning Rate**: Monitor schedule
5. **Gradient Norm**: Check for explosion/vanishing

### Using TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/balorg_training")

# Log metrics
writer.add_scalar("Loss/train", train_loss, step)
writer.add_scalar("Loss/val", val_loss, step)
writer.add_scalar("Learning_rate", lr, step)

# View with: tensorboard --logdir=runs
```

### Using Weights & Biases

```python
import wandb

# Initialize
wandb.init(project="balorg-ai", config=config)

# Log metrics
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "learning_rate": lr,
})
```

## Troubleshooting

### Out of Memory (OOM)

Solutions:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Reduce sequence length
5. Use smaller model

### Loss Not Decreasing

Solutions:
1. Increase learning rate
2. Check data preprocessing
3. Increase model capacity
4. Add more training data
5. Adjust warmup steps

### Loss Exploding

Solutions:
1. Decrease learning rate
2. Enable gradient clipping
3. Check data for anomalies
4. Use mixed precision training
5. Increase warmup steps

### Overfitting

Solutions:
1. Add more training data
2. Increase dropout
3. Add weight decay
4. Use data augmentation
5. Early stopping

## Best Practices

### Data Preparation

1. Clean and normalize text
2. Remove duplicates
3. Balance classes if applicable
4. Use consistent tokenization
5. Shuffle training data

### Training Strategy

1. Start with small model
2. Validate on held-out data
3. Save multiple checkpoints
4. Monitor validation loss
5. Use early stopping

### Resource Management

1. Use GPU if available
2. Enable mixed precision
3. Optimize batch size
4. Use multiple workers for data loading
5. Pin memory for faster transfers

### Reproducibility

```python
import torch
import numpy as np
import random

# Set seeds
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# For CUDA
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Example Training Script

See `examples/train_model.py` for a complete example.

```bash
python examples/train_model.py \
    --data_path data.txt \
    --output_dir ./output \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --max_length 512
```

## Next Steps

After training:

1. **Evaluate**: Test on held-out data
2. **Fine-tune**: Adapt to specific tasks
3. **Deploy**: Use for inference
4. **Monitor**: Track performance in production

## Resources

- [Architecture Guide](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Examples](../examples/)
- PyTorch Documentation
- Hugging Face Tutorials
