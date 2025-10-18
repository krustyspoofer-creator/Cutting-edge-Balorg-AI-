"""
Balorg AI - A Cutting-edge Connectionist AI System

This package provides a comprehensive deep learning framework for building
advanced AI systems with transformer-based architectures, multimodal learning,
and large-scale neural networks.
"""

__version__ = "0.1.0"
__author__ = "Balorg AI Team"

from .models.transformer import BalorgTransformer
from .models.config import ModelConfig
from .training.trainer import BalorgTrainer
from .applications.conversational import ConversationalAI
from .applications.content_generator import ContentGenerator
from .applications.qa_system import QuestionAnsweringSystem

__all__ = [
    "BalorgTransformer",
    "ModelConfig",
    "BalorgTrainer",
    "ConversationalAI",
    "ContentGenerator",
    "QuestionAnsweringSystem",
]
