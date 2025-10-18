"""
Training utilities and trainer for Balorg AI.

Implements advanced optimization techniques including distributed training,
mixed-precision training, and gradient checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import os
import json

from ..models.transformer import BalorgTransformer
from ..models.config import ModelConfig


class BalorgTrainer:
    """
    Trainer class for Balorg AI models.
    
    Supports distributed training, mixed-precision training, and advanced
    optimization techniques for efficient large-scale model training.
    """
    
    def __init__(
        self,
        model: BalorgTransformer,
        config: ModelConfig,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./output",
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 1000,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Balorg AI model to train
            config: Model configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            device: Device to train on
            output_dir: Directory to save checkpoints and logs
            logging_steps: Log metrics every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            max_grad_norm: Maximum gradient norm for clipping
            use_amp: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device == "cuda"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=5e-5,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Metrics
        self.training_history = {
            "loss": [],
            "learning_rate": [],
            "eval_loss": [],
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.compute_loss(outputs, labels)
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.compute_loss(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                })
                self.training_history["loss"].append(avg_loss)
                self.training_history["learning_rate"].append(lr)
            
            # Evaluation
            if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                eval_loss = self.evaluate()
                self.training_history["eval_loss"].append(eval_loss)
                print(f"\nStep {self.global_step} - Eval Loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best_model")
            
            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        return total_loss / num_batches
    
    def evaluate(self) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.compute_loss(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        logits = outputs["logits"]
        # Shift labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss
    
    def train(self, num_epochs: int):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Output Directory: {self.output_dir}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            avg_loss = self.train_epoch()
            print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch-{epoch}")
        
        # Save final model
        self.save_checkpoint("final_model")
        self.save_training_history()
        
        print("Training completed!")
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save config
        self.config.to_json_file(os.path.join(checkpoint_dir, "config.json"))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }
        with open(os.path.join(checkpoint_dir, "training_state.json"), 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        # Load model
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt")))
        
        # Load optimizer
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt")))
        
        # Load training state
        with open(os.path.join(checkpoint_path, "training_state.json"), 'r') as f:
            state = json.load(f)
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_eval_loss = state["best_eval_loss"]
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def save_training_history(self):
        """Save training history."""
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")
