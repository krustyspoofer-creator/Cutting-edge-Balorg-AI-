"""Training utilities for Balorg AI.

This module provides advanced training capabilities including:
- Distributed training support
- Mixed-precision training
- Gradient accumulation
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import os
import json
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Configuration for training process.
    
    Attributes:
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for learning rate
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        eval_steps: Evaluate every N steps
        mixed_precision: Use mixed precision training (fp16)
        device: Device to train on ('cuda' or 'cpu')
    """
    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    logging_steps: int = 100
    eval_steps: int = 1000
    mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """Advanced trainer for Balorg AI models.
    
    Features:
    - Mixed precision training for faster computation
    - Gradient accumulation for larger effective batch sizes
    - Learning rate scheduling with warmup
    - Automatic checkpointing
    - Metrics logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            compute_metrics: Function to compute evaluation metrics
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        
        # Move model to device
        self.model.to(config.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
            json.dump(asdict(config), f, indent=2)
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup."""
        # Calculate total steps
        if self.train_dataset:
            steps_per_epoch = len(self.train_dataset) // self.config.batch_size
            total_steps = steps_per_epoch * self.config.num_epochs
        else:
            total_steps = 10000  # Default if dataset not provided
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        # Sequential scheduler: warmup then cosine
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )
        
        return scheduler
    
    def train(self) -> Dict[str, Any]:
        """Run the training loop.
        
        Returns:
            Dictionary containing training statistics
        """
        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided to train()")
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                loss = self._training_step(batch)
                epoch_loss += loss
                total_loss += loss
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})
                
                # Logging
                if (self.global_step + 1) % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    self._log_metrics({"train_loss": avg_loss}, self.global_step)
                    total_loss = 0.0
                
                # Evaluation
                if self.eval_dataset and (self.global_step + 1) % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, self.global_step)
                    self.model.train()
                
                # Save checkpoint
                if (self.global_step + 1) % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
        
        # Save final model
        self.save_checkpoint("final")
        
        return {"total_steps": self.global_step, "epochs": self.config.num_epochs}
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Loss value for this step
        """
        # Move batch to device
        batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.get("loss", self._compute_loss(outputs, batch))
        else:
            outputs = self.model(**batch)
            loss = outputs.get("loss", self._compute_loss(outputs, batch))
        
        # Backward pass
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss from model outputs.
        
        Args:
            outputs: Model outputs
            batch: Batch data
            
        Returns:
            Loss tensor
        """
        logits = outputs["logits"]
        labels = batch.get("labels", batch.get("input_ids"))
        
        # Shift labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.eval_dataset is None:
            return {}
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.get("loss", self._compute_loss(outputs, batch))
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }
        
        if self.compute_metrics:
            custom_metrics = self.compute_metrics(self.model, eval_loader)
            metrics.update(custom_metrics)
        
        return metrics
    
    def save_checkpoint(self, name: Optional[str] = None):
        """Save model checkpoint.
        
        Args:
            name: Optional name for checkpoint (defaults to step number)
        """
        if name is None:
            name = f"checkpoint-{self.global_step}"
        
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, "model.pt")
        )
        
        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.pt")
        )
        
        # Save scheduler state
        torch.save(
            self.scheduler.state_dict(),
            os.path.join(checkpoint_dir, "scheduler.pt")
        )
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
        }
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        # Load model state
        self.model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model.pt"))
        )
        
        # Load optimizer state
        self.optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        )
        
        # Load scheduler state
        self.scheduler.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "scheduler.pt"))
        )
        
        # Load training state
        with open(os.path.join(checkpoint_dir, "training_state.json"), "r") as f:
            state = json.load(f)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_metric = state["best_metric"]
        
        print(f"Checkpoint loaded from {checkpoint_dir}")
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to console.
        
        Args:
            metrics: Dictionary of metrics
            step: Current training step
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {step}: {metrics_str}")
