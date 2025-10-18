"""
Balorg AI - Advanced Deep Learning Framework
"""

__version__ = "1.0.0"
__author__ = "Balorg AI Team"

from .core import BalorgAI
from .technical_mode import TechnicalMode
from .optimizations import (
    MixedPrecisionTrainer,
    DistributedTrainer,
    KnowledgeDistillation,
    ModelPruner,
    ModelQuantizer
)

__all__ = [
    'BalorgAI',
    'TechnicalMode',
    'MixedPrecisionTrainer',
    'DistributedTrainer',
    'KnowledgeDistillation',
    'ModelPruner',
    'ModelQuantizer'
]
