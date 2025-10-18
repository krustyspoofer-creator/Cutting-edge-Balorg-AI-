"""
Optimization utilities for Balorg AI.

Provides advanced optimization techniques for efficient training.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Iterator
from torch.optim import Optimizer


class AdamWScheduleFree(Optimizer):
    """
    AdamW optimizer with schedule-free learning rate.
    
    Implements AdamW with decoupled weight decay and automatic learning rate adjustment.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Decoupled weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-group['lr'])
        
        return loss


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """
    Create a learning rate scheduler with linear warmup and decay.
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch number
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a learning rate scheduler with cosine annealing and warmup.
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch number
        
    Returns:
        Learning rate scheduler
    """
    import math
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing for memory efficiency.
    
    Args:
        model: Model to enable gradient checkpointing for
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    else:
        # Manual gradient checkpointing setup
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_params: Optional[List[str]] = None,
) -> List[dict]:
    """
    Get parameter groups for optimization with weight decay.
    
    Args:
        model: Model to get parameter groups for
        weight_decay: Weight decay value
        no_decay_params: List of parameter name patterns to exclude from weight decay
        
    Returns:
        List of parameter groups
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_params) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_params) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    return optimizer_grouped_parameters
