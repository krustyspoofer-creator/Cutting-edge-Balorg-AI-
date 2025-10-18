# Balorg AI Quick Reference

## Installation

```bash
git clone https://github.com/krustyspoofer-creator/Cutting-edge-Balorg-AI-.git
cd Cutting-edge-Balorg-AI-
pip install -e .
```

## Basic Usage

### 1. Create a Model

```python
from balorg_ai.models import BalorgTransformer, ModelConfig

config = ModelConfig(vocab_size=50000, hidden_size=768)
model = BalorgTransformer(config)
```

### 2. Train a Model

```python
from balorg_ai.training import BalorgTrainer
from balorg_ai.data import TextDataset, DataProcessor

# Prepare data
dataset = TextDataset(texts, tokenizer, max_length=512)
train_loader = DataProcessor.create_dataloader(dataset, batch_size=8)

# Train
trainer = BalorgTrainer(model, config, train_dataloader=train_loader)
trainer.train(num_epochs=10)
```

### 3. Use Conversational AI

```python
from balorg_ai.applications import ConversationalAI

conv_ai = ConversationalAI(model, tokenizer)
response = conv_ai.generate_response("Hello!")
```

### 4. Generate Content

```python
from balorg_ai.applications import ContentGenerator

generator = ContentGenerator(model, tokenizer)
article = generator.generate_article(
    title="AI in 2024",
    keywords=["AI", "machine learning"]
)
```

### 5. Answer Questions

```python
from balorg_ai.applications import QuestionAnsweringSystem

qa_system = QuestionAnsweringSystem(model, tokenizer)
answer = qa_system.answer_question("What is AI?")
```

## Model Configurations

### Base Model (Default)
```python
ModelConfig(
    vocab_size=50000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
)
```

### Large Model
```python
from balorg_ai.models import LargeModelConfig
config = LargeModelConfig()  # 1024 hidden, 24 layers
```

### Multimodal Model
```python
from balorg_ai.models import MultimodalConfig
config = MultimodalConfig(use_vision=True, use_audio=True)
```

## Common Tasks

### Load Pre-trained Model

```python
config = ModelConfig.from_json_file("path/config.json")
model = BalorgTransformer(config)
model.load_state_dict(torch.load("path/model.pt"))
```

### Save Model

```python
config.to_json_file("output/config.json")
torch.save(model.state_dict(), "output/model.pt")
```

### Generate Text

```python
input_ids = tokenizer.encode("Once upon a time", return_tensors='pt')
output_ids = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
text = tokenizer.decode(output_ids[0])
```

### Mixed Precision Training

```python
trainer = BalorgTrainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    use_amp=True,  # Enable mixed precision
)
```

### Gradient Checkpointing

```python
from balorg_ai.utils import enable_gradient_checkpointing
enable_gradient_checkpointing(model)
```

## Examples

Run example scripts:

```bash
# Training
python examples/train_model.py

# Conversational AI
python examples/conversational_ai.py

# Content Generation
python examples/content_generation.py

# Question Answering
python examples/question_answering.py
```

## Key Parameters

### Training
- `learning_rate`: 1e-5 to 5e-4 (default: 5e-5)
- `batch_size`: 4-64 (limited by memory)
- `num_epochs`: 3-20 (depends on data size)
- `warmup_steps`: 5-10% of total steps
- `weight_decay`: 0.01-0.1 (default: 0.01)

### Generation
- `temperature`: 0.1-2.0 (default: 0.8)
  - Lower = more deterministic
  - Higher = more creative
- `top_k`: 10-100 (default: 50)
  - Number of top tokens to consider
- `top_p`: 0.5-1.0 (default: 0.95)
  - Nucleus sampling threshold

### Model Size
- `hidden_size`: 256-2048 (default: 768)
- `num_layers`: 6-48 (default: 12)
- `num_heads`: 8-32 (default: 12)

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
batch_size = 4  # or 2

# Enable gradient checkpointing
enable_gradient_checkpointing(model)

# Use gradient accumulation
accumulation_steps = 4
```

### Slow Training
```python
# Use mixed precision
use_amp = True

# Increase batch size (if memory allows)
batch_size = 16

# Use multiple workers
num_workers = 4
```

### Poor Quality
```python
# Increase model size
hidden_size = 1024
num_layers = 24

# More training data
# Longer training
num_epochs = 20

# Better hyperparameters
learning_rate = 3e-5
warmup_steps = 1000
```

## Resources

- **Documentation**: `docs/`
- **Examples**: `examples/`
- **Source Code**: `src/balorg_ai/`

## Support

Open an issue on GitHub for questions and support.
