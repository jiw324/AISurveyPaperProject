"""Trainer for prompt injection detection models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import time

from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for transformer-based classifiers."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 5,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 2,
        output_dir: str = "results/models",
        mixed_precision: bool = True,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Test dataloader (optional)
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_epochs: Number of epochs
            warmup_ratio: Warmup ratio for scheduler
            early_stopping_patience: Patience for early stopping
            output_dir: Directory to save models
            mixed_precision: Whether to use mixed precision training
            max_grad_norm: Max gradient norm for clipping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed precision training
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Evaluator
        self.evaluator = Evaluator()
        
        # Training state
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
            else:
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: Dataloader to evaluate on
            split_name: Name of the split (for logging)
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(dataloader, desc=f"Evaluating ({split_name})")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
            else:
                outputs = self.model(input_ids, attention_mask, labels)
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Collect predictions and labels
            all_predictions.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
        
        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics(all_predictions, all_labels)
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def train(self) -> Dict[str, any]:
        """
        Train the model.
        
        Returns:
            Training history and best metrics
        """
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Training batches per epoch: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*80}")
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            val_metrics = self.evaluate(self.val_loader, split_name="val")
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_metrics'].append(val_metrics)
            
            # Log validation metrics
            self.evaluator.log_metrics(val_metrics, prefix="Validation")
            
            # Check for improvement
            current_f1 = val_metrics['f1']
            
            if current_f1 > self.best_val_f1:
                logger.info(f"✓ Validation F1 improved: {self.best_val_f1:.4f} → {current_f1:.4f}")
                self.best_val_f1 = current_f1
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint("best_model.pt", epoch, val_metrics)
                logger.info(f"Saved best model to {self.output_dir / 'best_model.pt'}")
            else:
                self.patience_counter += 1
                logger.info(f"✗ No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break
            
            # Save checkpoint every epoch
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch, val_metrics)
        
        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {training_time/60:.2f} minutes")
        
        # Evaluate on test set if available
        test_metrics = None
        if self.test_loader is not None:
            logger.info("\nEvaluating on test set...")
            # Load best model
            self.load_checkpoint(self.output_dir / "best_model.pt")
            test_metrics = self.evaluate(self.test_loader, split_name="test")
            self.evaluator.log_metrics(test_metrics, prefix="Test")
        
        return {
            'history': self.history,
            'best_val_f1': self.best_val_f1,
            'test_metrics': test_metrics,
            'training_time': training_time
        }
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_f1': self.best_val_f1
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {checkpoint_path}")

