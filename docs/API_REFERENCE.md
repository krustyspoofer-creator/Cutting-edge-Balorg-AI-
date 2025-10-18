# Balorg AI API Reference

## Table of Contents

1. [Models](#models)
2. [Training](#training)
3. [Data](#data)
4. [Applications](#applications)
5. [Utilities](#utilities)

---

## Models

### ModelConfig

Configuration class for Balorg AI models.

```python
from balorg_ai.models import ModelConfig

config = ModelConfig(
    vocab_size=50000,              # Size of vocabulary
    hidden_size=768,               # Hidden dimension size
    num_hidden_layers=12,          # Number of transformer layers
    num_attention_heads=12,        # Number of attention heads
    intermediate_size=3072,        # Feed-forward intermediate size
    max_position_embeddings=2048,  # Maximum sequence length
    hidden_dropout_prob=0.1,       # Dropout probability
    attention_probs_dropout_prob=0.1,  # Attention dropout
    use_mixed_precision=True,      # Enable mixed precision
    gradient_checkpointing=False,  # Enable gradient checkpointing
)
```

**Methods:**
- `from_dict(config_dict)`: Create from dictionary
- `to_dict()`: Convert to dictionary
- `from_json_file(path)`: Load from JSON file
- `to_json_file(path)`: Save to JSON file

### LargeModelConfig

Pre-configured large model settings.

```python
from balorg_ai.models import LargeModelConfig

config = LargeModelConfig()  # 1024 hidden, 24 layers
```

### MultimodalConfig

Configuration for multimodal models.

```python
from balorg_ai.models import MultimodalConfig

config = MultimodalConfig(
    use_vision=True,
    use_audio=True,
    vision_hidden_size=768,
    audio_hidden_size=768,
)
```

### BalorgTransformer

Main transformer model class.

```python
from balorg_ai.models import BalorgTransformer

model = BalorgTransformer(config)
```

**Methods:**

#### forward()
```python
outputs = model.forward(
    input_ids,              # Token IDs (batch_size, seq_length)
    attention_mask=None,    # Attention mask (batch_size, seq_length)
    vision_features=None,   # Optional vision features
    audio_features=None,    # Optional audio features
)
# Returns: {"logits": tensor, "hidden_states": tensor}
```

#### generate()
```python
output_ids = model.generate(
    input_ids,          # Starting tokens
    max_length=100,     # Maximum generation length
    temperature=1.0,    # Sampling temperature
    top_k=50,          # Top-k sampling
    top_p=0.95,        # Nucleus sampling
)
```

---

## Training

### BalorgTrainer

Main training class with advanced features.

```python
from balorg_ai.training import BalorgTrainer

trainer = BalorgTrainer(
    model,                    # BalorgTransformer instance
    config,                   # ModelConfig instance
    train_dataloader=None,    # Training DataLoader
    eval_dataloader=None,     # Evaluation DataLoader
    optimizer=None,           # Optimizer (default: AdamW)
    scheduler=None,           # LR scheduler
    device="cuda",            # Device ("cuda" or "cpu")
    output_dir="./output",    # Output directory
    logging_steps=100,        # Log every N steps
    eval_steps=500,           # Evaluate every N steps
    save_steps=1000,          # Save checkpoint every N steps
    max_grad_norm=1.0,        # Gradient clipping
    use_amp=True,            # Automatic mixed precision
)
```

**Methods:**

#### train()
```python
trainer.train(num_epochs=10)
```

#### evaluate()
```python
eval_loss = trainer.evaluate()
```

#### save_checkpoint()
```python
trainer.save_checkpoint("checkpoint_name")
```

#### load_checkpoint()
```python
trainer.load_checkpoint("path/to/checkpoint")
```

---

## Data

### TextDataset

Dataset for text data.

```python
from balorg_ai.data import TextDataset

dataset = TextDataset(
    texts,                      # List of text strings
    tokenizer,                  # Tokenizer instance
    max_length=512,            # Maximum sequence length
    return_attention_mask=True, # Return attention mask
)
```

### MultimodalDataset

Dataset for multimodal data.

```python
from balorg_ai.data import MultimodalDataset

dataset = MultimodalDataset(
    texts,              # List of text strings
    images=None,        # Optional list of image arrays
    audio=None,         # Optional list of audio arrays
    tokenizer=None,     # Tokenizer instance
    max_length=512,     # Maximum sequence length
)
```

### DataProcessor

Utilities for data processing.

```python
from balorg_ai.data import DataProcessor

# Create DataLoader
dataloader = DataProcessor.create_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

# Load text data
texts = DataProcessor.load_text_data("data.txt")

# Split dataset
train, val, test = DataProcessor.split_dataset(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)
```

---

## Applications

### ConversationalAI

Interactive conversational AI system.

```python
from balorg_ai.applications import ConversationalAI

conv_ai = ConversationalAI(
    model,              # BalorgTransformer instance
    tokenizer,          # Tokenizer instance
    max_history=10,     # Maximum conversation history
    max_length=100,     # Maximum response length
    temperature=0.8,    # Sampling temperature
    top_k=50,          # Top-k sampling
    top_p=0.95,        # Nucleus sampling
    device="cuda",      # Device
)
```

**Methods:**

#### generate_response()
```python
response = conv_ai.generate_response("Hello, how are you?")
```

#### chat()
```python
conv_ai.chat()  # Start interactive chat
```

#### reset_history()
```python
conv_ai.reset_history()  # Clear conversation history
```

#### get_history()
```python
history = conv_ai.get_history()  # Get conversation history
```

#### from_pretrained()
```python
conv_ai = ConversationalAI.from_pretrained(
    model_path="./output/best_model",
    tokenizer=tokenizer,
    device="cuda",
)
```

### ContentGenerator

Generate various types of content.

```python
from balorg_ai.applications import ContentGenerator

generator = ContentGenerator(
    model,      # BalorgTransformer instance
    tokenizer,  # Tokenizer instance
    device="cuda",
)
```

**Methods:**

#### generate_article()
```python
article = generator.generate_article(
    title="Article Title",
    keywords=["keyword1", "keyword2"],
    max_length=500,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)
```

#### generate_story()
```python
story = generator.generate_story(
    genre="science fiction",
    characters=["Character1", "Character2"],
    setting="Future city",
    max_length=800,
    temperature=0.9,
)
```

#### generate_dialogue()
```python
dialogue = generator.generate_dialogue(
    context="Two friends meeting",
    num_turns=5,
    max_length=300,
)
```

#### generate_continuation()
```python
continuation = generator.generate_continuation(
    text="Once upon a time",
    max_length=200,
)
```

#### from_pretrained()
```python
generator = ContentGenerator.from_pretrained(
    model_path="./output/best_model",
    tokenizer=tokenizer,
)
```

### QuestionAnsweringSystem

Answer questions based on context or knowledge.

```python
from balorg_ai.applications import QuestionAnsweringSystem

qa_system = QuestionAnsweringSystem(
    model,      # BalorgTransformer instance
    tokenizer,  # Tokenizer instance
    device="cuda",
)
```

**Methods:**

#### answer_question()
```python
answer = qa_system.answer_question(
    question="What is AI?",
    context="Optional context...",
    max_length=150,
    temperature=0.7,
)
```

#### answer_multiple_questions()
```python
questions = ["Q1?", "Q2?", "Q3?"]
answers = qa_system.answer_multiple_questions(
    questions,
    context="Shared context",
)
```

#### extractive_qa()
```python
answer = qa_system.extractive_qa(
    question="Question?",
    context="Context containing answer",
    max_length=50,
)
```

#### open_domain_qa()
```python
answer = qa_system.open_domain_qa(
    question="General knowledge question?",
    max_length=200,
)
```

#### multi_hop_qa()
```python
answer = qa_system.multi_hop_qa(
    question="Complex question?",
    contexts=["Context 1", "Context 2"],
    max_length=200,
)
```

#### interactive_qa()
```python
qa_system.interactive_qa()  # Start interactive Q&A
```

#### from_pretrained()
```python
qa_system = QuestionAnsweringSystem.from_pretrained(
    model_path="./output/best_model",
    tokenizer=tokenizer,
)
```

---

## Utilities

### Optimizers

#### AdamWScheduleFree
```python
from balorg_ai.utils import AdamWScheduleFree

optimizer = AdamWScheduleFree(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)
```

### Learning Rate Schedulers

#### get_linear_schedule_with_warmup()
```python
from balorg_ai.utils import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000,
)
```

#### get_cosine_schedule_with_warmup()
```python
from balorg_ai.utils import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000,
    num_cycles=0.5,
)
```

### Helper Functions

#### enable_gradient_checkpointing()
```python
from balorg_ai.utils import enable_gradient_checkpointing

enable_gradient_checkpointing(model)
```

#### count_parameters()
```python
from balorg_ai.utils import count_parameters

num_params = count_parameters(model)
print(f"Model has {num_params:,} parameters")
```

#### get_parameter_groups()
```python
from balorg_ai.utils import get_parameter_groups

param_groups = get_parameter_groups(
    model,
    weight_decay=0.01,
    no_decay_params=["bias", "LayerNorm.weight"],
)
optimizer = torch.optim.AdamW(param_groups, lr=5e-5)
```

---

## Examples

See `examples/` directory for complete working examples:

- `train_model.py`: Complete training pipeline
- `conversational_ai.py`: Interactive chatbot
- `content_generation.py`: Content generation examples
- `question_answering.py`: Q&A system examples

## Notes

- All methods support PyTorch tensors and standard Python data types
- Models automatically move to specified device
- Mixed precision training is recommended for faster training
- Use gradient checkpointing for memory-constrained environments
- All configurations are serializable to JSON

## See Also

- [Quick Reference](QUICK_REFERENCE.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Training Guide](TRAINING_GUIDE.md)
