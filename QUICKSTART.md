# Balorg AI Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/krustyspoofer-creator/Cutting-edge-Balorg-AI-.git
cd Cutting-edge-Balorg-AI-

# Install in development mode
pip install -e .
```

## 5-Minute Tutorial

### 1. Explore Technical Mode

The fastest way to learn about Balorg AI is through the interactive technical mode:

```bash
python examples/technical_mode_demo.py
```

You'll see a menu with three main options:
- `1`: Model Architecture
- `2`: Optimization Techniques
- `3`: Applications

Try entering different numbers and keywords like `transformer`, `mixed_precision`, or `nlp`.

### 2. Create Your First Model

```python
from balorg_ai import BalorgAI

# Create a transformer model
model = BalorgAI(
    model_type='transformer',
    config={
        'num_layers': 12,
        'hidden_size': 768,
        'num_attention_heads': 12
    }
)

# View model summary
print(model.summary())
```

**Output:**
```
Balorg AI Model Summary
==================================================
Model Type: transformer

Configuration:
  num_layers: 12
  hidden_size: 768
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 512
```

### 3. Apply Mixed Precision Training

```python
from balorg_ai import MixedPrecisionTrainer

# Initialize mixed precision trainer
trainer = MixedPrecisionTrainer(
    enabled=True,
    opt_level='O1',  # Recommended for best balance
    loss_scale=1024.0
)

# Apply to your model
model = trainer.apply(model)

print("Mixed Precision Training enabled!")
print(f"Configuration: {trainer.get_config()}")
```

### 4. Setup Distributed Training

```python
from balorg_ai import DistributedTrainer

# Initialize distributed trainer for 4 GPUs
trainer = DistributedTrainer(
    backend='nccl',  # Best for NVIDIA GPUs
    world_size=4,
    rank=0
)

# Setup distributed environment
trainer.setup()
print("Distributed training ready!")
```

### 5. Compress Model with Knowledge Distillation

```python
from balorg_ai import BalorgAI, KnowledgeDistillation

# Teacher model (large)
teacher = BalorgAI(
    model_type='transformer',
    config={'num_layers': 12, 'hidden_size': 768}
)

# Student model (small)
student = BalorgAI(
    model_type='transformer',
    config={'num_layers': 6, 'hidden_size': 384}
)

# Setup knowledge distillation
kd = KnowledgeDistillation(
    temperature=3.0,
    alpha=0.5,
    teacher_model=teacher
)

print(f"Teacher: {teacher.summary()}")
print(f"Student: {student.summary()}")
print("Ready for distillation!")
```

### 6. Prune Model for Efficiency

```python
from balorg_ai import ModelPruner

# Create pruner with 50% sparsity
pruner = ModelPruner(
    pruning_method='magnitude',
    sparsity=0.5,
    structured=False
)

# Prune the model
pruned_model = pruner.prune(model)

# Get statistics
stats = pruner.get_sparsity_stats(pruned_model)
print(f"Pruning complete! Sparsity: {stats['global_sparsity']*100}%")
```

### 7. Quantize for Deployment

```python
from balorg_ai import ModelQuantizer

# Create quantizer
quantizer = ModelQuantizer(
    quantization_type='dynamic',
    dtype='int8',
    calibration_method='minmax'
)

# Quantize the model
quantized_model = quantizer.quantize(model)

# Get size reduction
reduction = quantizer.get_size_reduction(model, quantized_model)
print(f"Model size reduced by {reduction['reduction_ratio']*100}%")
```

## Run All Examples

```bash
# Model architectures
python examples/architecture_demo.py

# Optimization techniques
python examples/optimization_demo.py

# Interactive technical mode
python examples/technical_mode_demo.py

# Complete non-interactive demonstration
python demo_technical_mode_full.py
```

## Common Workflows

### Workflow 1: Training a Large Model

```python
from balorg_ai import BalorgAI, MixedPrecisionTrainer, DistributedTrainer

# Create model
model = BalorgAI(model_type='transformer', config={'num_layers': 24})

# Apply mixed precision
mp = MixedPrecisionTrainer(opt_level='O1')
model = mp.apply(model)

# Setup distributed training
dist = DistributedTrainer(world_size=8)
dist.setup()

# Train your model...
```

### Workflow 2: Model Compression Pipeline

```python
from balorg_ai import (
    BalorgAI,
    KnowledgeDistillation,
    ModelPruner,
    ModelQuantizer
)

# Large model
teacher = BalorgAI(model_type='transformer', config={'num_layers': 12})

# Step 1: Knowledge Distillation
student = BalorgAI(model_type='transformer', config={'num_layers': 6})
kd = KnowledgeDistillation(teacher_model=teacher)
student = kd.distill(student, data)

# Step 2: Pruning
pruner = ModelPruner(sparsity=0.5)
student = pruner.prune(student)

# Step 3: Quantization
quantizer = ModelQuantizer(dtype='int8')
student = quantizer.quantize(student)

# Now you have a highly compressed model!
```

### Workflow 3: Exploring Technical Details

```python
from balorg_ai import TechnicalMode

tech_mode = TechnicalMode()

# Show welcome message
print(tech_mode.get_welcome_message())

# Get architecture details
print(tech_mode.get_architecture_details('transformer'))

# Get optimization details
print(tech_mode.get_optimization_details('mixed_precision'))
print(tech_mode.get_optimization_details('quantization'))

# Get application details
print(tech_mode.get_application_details('nlp'))
```

## Next Steps

1. **Read the Technical Guide**: [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)
2. **Check API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
3. **Explore Examples**: Browse the `examples/` directory
4. **Run Tests**: 
   - Unit tests: `python test_technical_mode.py`
   - Integration tests: `python test_integration.py`

## Tips

- Start with mixed precision training for immediate speedup
- Use distributed training for large datasets
- Apply knowledge distillation for deployment models
- Combine pruning and quantization for maximum compression
- Use technical mode to explore all capabilities interactively

## Troubleshooting

**Import Error?**
Make sure you're in the project directory and use:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Need more examples?**
Check the `examples/` directory for comprehensive demonstrations.

**Questions?**
Open an issue on GitHub!

## Resources

- **Documentation**: All `.md` files in the repository
- **Examples**: `examples/` directory
- **Tests**: `test_technical_mode.py`
- **Demo**: `demo_technical_mode_full.py`

Happy optimizing with Balorg AI! 🚀
