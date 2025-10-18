"""Training utilities for Balorg AI Framework."""

from typing import Optional, Dict, Any, Callable
import numpy as np


class Trainer:
    """
    Trainer class for managing the training process of AI models.
    
    This class provides utilities for training models with features like
    checkpointing, early stopping, and learning rate scheduling.
    """
    
    def __init__(self, model, optimizer: str = 'adam', loss: str = 'mse'):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer to use
            loss: Loss function to use
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.history = {}
        self.callbacks = []
        
    def add_callback(self, callback: Callable):
        """
        Add a callback function to be called during training.
        
        Args:
            callback: Callback function
        """
        self.callbacks.append(callback)
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 10, batch_size: int = 32,
              validation_data: Optional[tuple] = None,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            x_train: Training input data
            y_train: Training target data
            epochs: Number of epochs to train
            batch_size: Batch size for training
            validation_data: Optional validation data tuple (x_val, y_val)
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if not self.model.is_compiled:
            self.model.compile(optimizer=self.optimizer, loss=self.loss)
        
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose
        )
        
        # Call callbacks
        for callback in self.callbacks:
            callback(self.history)
        
        return self.history
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray,
                 verbose: int = 1) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            x_test: Test input data
            y_test: Test target data
            verbose: Verbosity level
            
        Returns:
            Evaluation results dictionary
        """
        return self.model.evaluate(x_test, y_test, verbose=verbose)
    
    def get_history(self) -> Dict[str, Any]:
        """
        Get the training history.
        
        Returns:
            Training history dictionary
        """
        return self.history


class EarlyStopping:
    """Early stopping callback to stop training when a metric stops improving."""
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 5, 
                 min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.wait = 0
        
    def __call__(self, history: Dict[str, Any]):
        """Check if training should stop."""
        if self.monitor not in history:
            return False
        
        current_value = history[self.monitor][-1]
        
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
        return self.wait >= self.patience


class LearningRateScheduler:
    """Learning rate scheduler callback."""
    
    def __init__(self, schedule: Callable[[int], float]):
        """
        Initialize learning rate scheduler.
        
        Args:
            schedule: Function that takes epoch number and returns learning rate
        """
        self.schedule = schedule
        self.epoch = 0
        
    def __call__(self, history: Dict[str, Any]):
        """Update learning rate based on schedule."""
        self.epoch += 1
        new_lr = self.schedule(self.epoch)
        # In a full implementation, this would update the optimizer's learning rate
        return new_lr
