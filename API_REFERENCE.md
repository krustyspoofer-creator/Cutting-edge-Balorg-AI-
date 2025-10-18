# Balorg AI API Reference

## Table of Contents
- [Core Module](#core-module)
- [Technical Mode](#technical-mode)
- [Optimizations](#optimizations)

---

## Core Module

### `balorg_ai.core`

#### `class BalorgAI`

Main Balorg AI class integrating various architectures and optimizations.

**Constructor:**
```python
BalorgAI(model_type='transformer', config=None)
```

**Parameters:**
- `model_type` (str): Type of model architecture. Options: `'transformer'`, `'neural_network'`. Default: `'transformer'`
- `config` (dict, optional): Configuration dictionary for the model. Default: `None`

**Methods:**

##### `get_model_info() -> Dict[str, Any]`
Get information about the current model.

**Returns:**
- `dict`: Dictionary containing model type and configuration

**Example:**
```python
model = BalorgAI(model_type='transformer')
info = model.get_model_info()
print(info)
```

##### `summary() -> str`
Return a summary of the model architecture.

**Returns:**
- `str`: Human-readable model summary

**Example:**
```python
model = BalorgAI(model_type='transformer')
print(model.summary())
```

---

#### `class TransformerModel`

Transformer-based model architecture for Balorg AI.

**Constructor:**
```python
TransformerModel(
    num_layers=12,
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512
)
```

**Parameters:**
- `num_layers` (int): Number of transformer layers. Default: `12`
- `hidden_size` (int): Dimensionality of hidden states. Default: `768`
- `num_attention_heads` (int): Number of attention heads. Default: `12`
- `intermediate_size` (int): Size of feed-forward intermediate layer. Default: `3072`
- `max_position_embeddings` (int): Maximum sequence length. Default: `512`

**Methods:**

##### `get_config() -> Dict[str, Any]`
Return model configuration.

**Example:**
```python
model = TransformerModel(num_layers=6, hidden_size=512)
config = model.get_config()
```

---

#### `class NeuralNetwork`

Neural network architecture for Balorg AI.

**Constructor:**
```python
NeuralNetwork(
    layer_sizes,
    activation='relu',
    dropout_rate=0.1
)
```

**Parameters:**
- `layer_sizes` (List[int]): List of layer sizes from input to output
- `activation` (str): Activation function. Default: `'relu'`
- `dropout_rate` (float): Dropout probability. Default: `0.1`

**Methods:**

##### `get_config() -> Dict[str, Any]`
Return network configuration.

**Example:**
```python
model = NeuralNetwork(layer_sizes=[512, 256, 128])
config = model.get_config()
```

---

## Technical Mode

### `balorg_ai.technical_mode`

#### `class TechnicalMode`

Interactive Technical Mode for exploring and configuring Balorg AI.

**Constructor:**
```python
TechnicalMode()
```

**Class Attributes:**

- `MENU_OPTIONS` (dict): Main menu options mapping
- `ARCHITECTURE_OPTIONS` (dict): Architecture sub-menu options
- `OPTIMIZATION_OPTIONS` (dict): Optimization sub-menu options
- `APPLICATION_OPTIONS` (dict): Application sub-menu options

**Methods:**

##### `get_welcome_message() -> str`
Get the welcome message for technical mode.

**Returns:**
- `str`: Welcome message text

**Example:**
```python
tech_mode = TechnicalMode()
print(tech_mode.get_welcome_message())
```

##### `get_architecture_details(arch_type: str) -> str`
Get detailed information about a specific architecture.

**Parameters:**
- `arch_type` (str): Architecture type (`'transformer'`, `'neural_network'`, `'hybrid'`)

**Returns:**
- `str`: Detailed architecture information

**Example:**
```python
tech_mode = TechnicalMode()
details = tech_mode.get_architecture_details('transformer')
print(details)
```

##### `get_optimization_details(opt_type: str) -> str`
Get detailed information about a specific optimization technique.

**Parameters:**
- `opt_type` (str): Optimization type (`'mixed_precision'`, `'distributed'`, `'distillation'`, `'pruning'`, `'quantization'`)

**Returns:**
- `str`: Detailed optimization information

**Example:**
```python
tech_mode = TechnicalMode()
details = tech_mode.get_optimization_details('mixed_precision')
print(details)
```

##### `get_application_details(app_type: str) -> str`
Get detailed information about a specific application area.

**Parameters:**
- `app_type` (str): Application type (`'nlp'`, `'cv'`, `'rl'`, `'multimodal'`)

**Returns:**
- `str`: Detailed application information

**Example:**
```python
tech_mode = TechnicalMode()
details = tech_mode.get_application_details('nlp')
print(details)
```

##### `process_query(query: str) -> str`
Process a technical query and return appropriate response.

**Parameters:**
- `query` (str): User query (number or text)

**Returns:**
- `str`: Response string

**Example:**
```python
tech_mode = TechnicalMode()
response = tech_mode.process_query('1')
print(response)
```

##### `interactive_session()`
Run an interactive technical mode session.

**Example:**
```python
tech_mode = TechnicalMode()
tech_mode.interactive_session()
```

---

## Optimizations

### `balorg_ai.optimizations`

#### `class MixedPrecisionTrainer`

Mixed Precision Training implementation.

**Constructor:**
```python
MixedPrecisionTrainer(
    enabled=True,
    loss_scale=1024.0,
    opt_level='O1'
)
```

**Parameters:**
- `enabled` (bool): Whether to enable mixed precision. Default: `True`
- `loss_scale` (float): Initial loss scale. Default: `1024.0`
- `opt_level` (str): Optimization level (`'O0'`, `'O1'`, `'O2'`, `'O3'`). Default: `'O1'`

**Methods:**

##### `get_config() -> Dict[str, Any]`
Return configuration.

##### `apply(model: Any) -> Any`
Apply mixed precision to model.

**Example:**
```python
trainer = MixedPrecisionTrainer(opt_level='O1')
model = trainer.apply(model)
```

---

#### `class DistributedTrainer`

Distributed Training implementation.

**Constructor:**
```python
DistributedTrainer(
    backend='nccl',
    world_size=1,
    rank=0,
    local_rank=0
)
```

**Parameters:**
- `backend` (str): Backend to use (`'nccl'`, `'gloo'`, `'mpi'`). Default: `'nccl'`
- `world_size` (int): Total number of processes. Default: `1`
- `rank` (int): Global rank of current process. Default: `0`
- `local_rank` (int): Local rank within node. Default: `0`

**Methods:**

##### `get_config() -> Dict[str, Any]`
Return configuration.

##### `setup() -> bool`
Setup distributed training environment.

##### `cleanup()`
Cleanup distributed training.

**Example:**
```python
trainer = DistributedTrainer(world_size=4, rank=0)
trainer.setup()
# ... training code ...
trainer.cleanup()
```

---

#### `class KnowledgeDistillation`

Knowledge Distillation implementation.

**Constructor:**
```python
KnowledgeDistillation(
    temperature=3.0,
    alpha=0.5,
    teacher_model=None
)
```

**Parameters:**
- `temperature` (float): Temperature for softening distributions. Default: `3.0`
- `alpha` (float): Weight balance between losses. Default: `0.5`
- `teacher_model` (Any, optional): Pre-trained teacher model. Default: `None`

**Methods:**

##### `get_config() -> Dict[str, Any]`
Return configuration.

##### `distill(student_model: Any, data: Any) -> Dict[str, float]`
Perform knowledge distillation.

**Returns:**
- `dict`: Dictionary with loss components

**Example:**
```python
kd = KnowledgeDistillation(temperature=3.0, teacher_model=teacher)
losses = kd.distill(student_model, data)
```

---

#### `class ModelPruner`

Model Pruning implementation.

**Constructor:**
```python
ModelPruner(
    pruning_method='magnitude',
    sparsity=0.5,
    structured=False
)
```

**Parameters:**
- `pruning_method` (str): Method to use (`'magnitude'`, `'random'`, `'structured'`). Default: `'magnitude'`
- `sparsity` (float): Target sparsity level (0.0-1.0). Default: `0.5`
- `structured` (bool): Whether to use structured pruning. Default: `False`

**Methods:**

##### `get_config() -> Dict[str, Any]`
Return configuration.

##### `prune(model: Any) -> Any`
Prune the model.

##### `get_sparsity_stats(model: Any) -> Dict[str, float]`
Get sparsity statistics.

**Example:**
```python
pruner = ModelPruner(sparsity=0.5)
pruned_model = pruner.prune(model)
stats = pruner.get_sparsity_stats(pruned_model)
```

---

#### `class ModelQuantizer`

Model Quantization implementation.

**Constructor:**
```python
ModelQuantizer(
    quantization_type='dynamic',
    dtype='int8',
    calibration_method='minmax'
)
```

**Parameters:**
- `quantization_type` (str): Type of quantization (`'dynamic'`, `'static'`, `'qat'`). Default: `'dynamic'`
- `dtype` (str): Target data type (`'int8'`, `'int16'`, `'float16'`). Default: `'int8'`
- `calibration_method` (str): Calibration method (`'minmax'`, `'entropy'`, `'percentile'`). Default: `'minmax'`

**Methods:**

##### `get_config() -> Dict[str, Any]`
Return configuration.

##### `quantize(model: Any) -> Any`
Quantize the model.

##### `get_size_reduction(original_model: Any, quantized_model: Any) -> Dict[str, Any]`
Calculate size reduction from quantization.

**Example:**
```python
quantizer = ModelQuantizer(dtype='int8')
quantized_model = quantizer.quantize(model)
reduction = quantizer.get_size_reduction(model, quantized_model)
```

---

## Usage Examples

### Complete Workflow

```python
from balorg_ai import (
    BalorgAI,
    TechnicalMode,
    MixedPrecisionTrainer,
    DistributedTrainer,
    KnowledgeDistillation,
    ModelPruner,
    ModelQuantizer
)

# Create a model
model = BalorgAI(
    model_type='transformer',
    config={
        'num_layers': 12,
        'hidden_size': 768
    }
)

# Apply mixed precision training
mp_trainer = MixedPrecisionTrainer(opt_level='O1')
model = mp_trainer.apply(model)

# Setup distributed training
dist_trainer = DistributedTrainer(world_size=4)
dist_trainer.setup()

# Create smaller model for distillation
student = BalorgAI(
    model_type='transformer',
    config={'num_layers': 6, 'hidden_size': 384}
)

# Apply knowledge distillation
kd = KnowledgeDistillation(teacher_model=model)
kd.distill(student, training_data)

# Prune the model
pruner = ModelPruner(sparsity=0.5)
student = pruner.prune(student)

# Quantize for deployment
quantizer = ModelQuantizer(dtype='int8')
student = quantizer.quantize(student)

# Explore technical details
tech_mode = TechnicalMode()
print(tech_mode.get_welcome_message())
```

### Interactive Technical Mode

```python
from balorg_ai import TechnicalMode

# Start interactive session
tech_mode = TechnicalMode()
tech_mode.interactive_session()

# Or query programmatically
response = tech_mode.process_query('1')  # Model Architecture
print(response)

details = tech_mode.get_optimization_details('mixed_precision')
print(details)
```

---

## Type Hints

The library uses type hints throughout. Import types:

```python
from typing import Optional, Dict, Any, List
```

---

## Logging

The library uses Python's standard logging module:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('balorg_ai')
```

---

## Version Information

```python
import balorg_ai

print(balorg_ai.__version__)  # '1.0.0'
print(balorg_ai.__author__)   # 'Balorg AI Team'
```
