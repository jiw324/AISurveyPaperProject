"""Evaluation metrics for prompt injection detection."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for computing detection metrics."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize evaluator.
        
        Args:
            threshold: Classification threshold for binary predictions
        """
        self.threshold = threshold
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        return_detailed: bool = False
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions (logits or probabilities) [N, 2]
            labels: Ground truth labels [N]
            return_detailed: Whether to return detailed per-class metrics
        
        Returns:
            Dictionary of metrics
        """
        # Convert logits to probabilities if needed
        if predictions.shape[1] == 2:
            probs = predictions[:, 1]  # Probability of injection class
        else:
            probs = predictions
        
        # Binary predictions
        pred_labels = (probs >= self.threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(labels, pred_labels),
            'precision': precision_score(labels, pred_labels, zero_division=0),
            'recall': recall_score(labels, pred_labels, zero_division=0),
            'f1': f1_score(labels, pred_labels, zero_division=0),
        }
        
        # AUC-ROC (requires probabilities)
        try:
            metrics['auc_roc'] = roc_auc_score(labels, probs)
        except ValueError as e:
            logger.warning(f"Could not compute AUC-ROC: {e}")
            metrics['auc_roc'] = 0.0
        
        # False Positive Rate at 95% Recall
        metrics['fpr_at_95_recall'] = self.compute_fpr_at_recall(
            labels, probs, target_recall=0.95
        )
        
        if return_detailed:
            # Confusion matrix
            cm = confusion_matrix(labels, pred_labels)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Per-class metrics
            report = classification_report(labels, pred_labels, output_dict=True)
            metrics['classification_report'] = report
        
        return metrics
    
    def compute_fpr_at_recall(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        target_recall: float = 0.95
    ) -> float:
        """
        Compute False Positive Rate at a target recall level.
        This is critical for usability - we want high recall but low FPR.
        
        Args:
            labels: Ground truth labels
            probs: Predicted probabilities for positive class
            target_recall: Target recall level (default: 0.95)
        
        Returns:
            False positive rate at target recall
        """
        # Sort by probability (descending)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_labels = labels[sorted_indices]
        
        # Find threshold that gives target recall
        n_positives = np.sum(labels == 1)
        n_negatives = np.sum(labels == 0)
        
        if n_positives == 0:
            return 0.0
        
        target_true_positives = int(n_positives * target_recall)
        
        # Find how many false positives we get
        cumsum_positives = np.cumsum(sorted_labels == 1)
        
        # Find first index where we reach target recall
        idx = np.where(cumsum_positives >= target_true_positives)[0]
        
        if len(idx) == 0:
            return 1.0  # Can't reach target recall
        
        idx = idx[0]
        
        # Count false positives up to this point
        false_positives = np.sum(sorted_labels[:idx+1] == 0)
        
        fpr = false_positives / n_negatives if n_negatives > 0 else 0.0
        
        return fpr
    
    def compute_per_attack_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        attack_types: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics separately for each attack type.
        
        Args:
            predictions: Model predictions [N, 2]
            labels: Ground truth labels [N]
            attack_types: Attack type for each example [N]
        
        Returns:
            Dictionary mapping attack type to metrics
        """
        probs = predictions[:, 1] if predictions.shape[1] == 2 else predictions
        pred_labels = (probs >= self.threshold).astype(int)
        
        unique_types = set(attack_types)
        per_type_metrics = {}
        
        for attack_type in unique_types:
            # Get indices for this attack type
            indices = [i for i, t in enumerate(attack_types) if t == attack_type]
            
            if len(indices) == 0:
                continue
            
            type_labels = labels[indices]
            type_pred_labels = pred_labels[indices]
            type_probs = probs[indices]
            
            # Compute metrics for this type
            metrics = {
                'count': len(indices),
                'accuracy': accuracy_score(type_labels, type_pred_labels),
                'precision': precision_score(type_labels, type_pred_labels, zero_division=0),
                'recall': recall_score(type_labels, type_pred_labels, zero_division=0),
                'f1': f1_score(type_labels, type_pred_labels, zero_division=0),
            }
            
            try:
                metrics['auc_roc'] = roc_auc_score(type_labels, type_probs)
            except ValueError:
                metrics['auc_roc'] = 0.0
            
            per_type_metrics[attack_type] = metrics
        
        return per_type_metrics
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Log metrics in a readable format.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for logging (e.g., "train", "val", "test")
        """
        prefix_str = f"{prefix} " if prefix else ""
        
        logger.info(f"{prefix_str}Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        logger.info(f"  FPR@95%:   {metrics['fpr_at_95_recall']:.4f}")

