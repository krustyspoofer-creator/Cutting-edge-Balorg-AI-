"""Tests for Balorg AI Framework core components."""

import pytest
import numpy as np
from balorg.framework import Model, Config, Trainer


class SimpleModel(Model):
    """Simple model implementation for testing."""
    
    def forward(self, x):
        return x * 2
    
    def backward(self, gradient):
        return gradient * 2


def test_model_initialization():
    """Test model initialization."""
    model = SimpleModel(name="test_model")
    assert model.name == "test_model"
    assert not model.is_compiled
    assert not model.is_trained


def test_model_compile():
    """Test model compilation."""
    model = SimpleModel()
    model.compile(optimizer='adam', loss='mse')
    
    assert model.is_compiled
    assert model._config['optimizer'] == 'adam'
    assert model._config['loss'] == 'mse'


def test_model_fit():
    """Test model training."""
    model = SimpleModel()
    model.compile(optimizer='adam', loss='mse')
    
    x_train = np.random.randn(100, 10)
    y_train = np.random.randn(100, 10)
    
    history = model.fit(x_train, y_train, epochs=5, batch_size=32)
    
    assert model.is_trained
    assert 'loss' in history
    assert 'accuracy' in history


def test_model_predict():
    """Test model prediction."""
    model = SimpleModel()
    model.compile(optimizer='adam', loss='mse')
    
    x_train = np.random.randn(100, 10)
    y_train = np.random.randn(100, 10)
    model.fit(x_train, y_train, epochs=1, verbose=0)
    
    x_test = np.random.randn(20, 10)
    predictions = model.predict(x_test)
    
    assert predictions.shape == x_test.shape


def test_config_initialization():
    """Test config initialization."""
    config = Config(learning_rate=0.001, batch_size=32)
    
    assert config.get('learning_rate') == 0.001
    assert config.get('batch_size') == 32


def test_config_update():
    """Test config update."""
    config = Config(learning_rate=0.001)
    config.update(batch_size=64)
    
    assert config.get('batch_size') == 64


def test_config_to_dict():
    """Test config to dictionary conversion."""
    config = Config(learning_rate=0.001, batch_size=32)
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict['learning_rate'] == 0.001


def test_trainer_initialization():
    """Test trainer initialization."""
    model = SimpleModel()
    trainer = Trainer(model, optimizer='adam', loss='mse')
    
    assert trainer.model == model
    assert trainer.optimizer == 'adam'
    assert trainer.loss == 'mse'


def test_trainer_train():
    """Test trainer training."""
    model = SimpleModel()
    trainer = Trainer(model, optimizer='adam', loss='mse')
    
    x_train = np.random.randn(100, 10)
    y_train = np.random.randn(100, 10)
    
    history = trainer.train(x_train, y_train, epochs=3, verbose=0)
    
    assert model.is_trained
    assert 'loss' in history
