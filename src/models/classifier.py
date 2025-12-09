"""Transformer-based classifiers for prompt injection detection."""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BertModel,
    RobertaModel,
    DistilBertModel
)
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PromptInjectionClassifier(nn.Module):
    """Binary classifier for prompt injection detection."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Initialize classifier.
        
        Args:
            model_name: Name of pre-trained model from HuggingFace
            num_labels: Number of output classes (2 for binary)
            dropout: Dropout probability
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained model
        logger.info(f"Loading pre-trained model: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder if specified (for ablation studies)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen")
        
        # Classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        logger.info(f"Model initialized: {model_name} with {num_labels} labels")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size]
            return_dict: Whether to return dictionary
        
        Returns:
            Dictionary with loss, logits, hidden_states, attentions
        """
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
        
        # Get [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "pooled_output": pooled_output
        }
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            Predicted class probabilities [batch_size, num_labels]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, return_dict=True)
            probs = torch.softmax(outputs["logits"], dim=-1)
        return probs


class RandomInitClassifier(nn.Module):
    """
    Classifier with randomly initialized encoder (for ablation study).
    Same architecture as pre-trained, but no pre-trained weights.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load config but initialize randomly
        logger.info(f"Initializing random model with {model_name} architecture")
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Create model with random weights
        if "bert" in model_name.lower() and "roberta" not in model_name.lower():
            self.encoder = BertModel(self.config)
        elif "roberta" in model_name.lower():
            self.encoder = RobertaModel(self.config)
        elif "distilbert" in model_name.lower():
            self.encoder = DistilBertModel(self.config)
        else:
            # Fallback to AutoModel with random init
            self.config._name_or_path = None  # Prevent loading pretrained
            self.encoder = AutoModel.from_config(self.config)
        
        # Classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        logger.info("Random initialization complete")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass (same as PromptInjectionClassifier)."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "pooled_output": pooled_output
        }


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = 2,
    random_init: bool = False
):
    """
    Load model and tokenizer.
    
    Args:
        model_name: Model name or path
        num_labels: Number of output labels
        random_init: Use random initialization (for ablation)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if random_init:
        model = RandomInitClassifier(model_name, num_labels)
    else:
        model = PromptInjectionClassifier(model_name, num_labels)
    
    return model, tokenizer

