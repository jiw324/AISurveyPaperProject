"""Dataset loading and preprocessing for prompt injection detection."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import jsonlines

logger = logging.getLogger(__name__)


class PromptInjectionDataset(Dataset):
    """Dataset for prompt injection detection task."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        label_map: Optional[Dict[str, int]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSONL file with data
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            label_map: Mapping from label names to integers
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or {"legitimate": 0, "injection": 1}
        
        self.examples = self._load_data()
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file."""
        examples = []
        
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            logger.warning("Run 'python scripts/download_data.py' to generate datasets")
            return examples
        
        with jsonlines.open(self.data_path) as reader:
            for obj in reader:
                examples.append(obj)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            example["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get label
        label = self.label_map.get(example["label"], example["label"])
        if isinstance(label, str):
            label = int(label)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": example["text"],
            "attack_type": example.get("attack_type", "none"),
            "metadata": example.get("metadata", {})
        }


class SanitizationDataset(Dataset):
    """Dataset for prompt sanitization (seq2seq) task."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 512
    ):
        """
        Initialize sanitization dataset.
        
        Args:
            data_path: Path to JSONL file
            tokenizer: HuggingFace tokenizer
            max_source_length: Max length for input (malicious prompt)
            max_target_length: Max length for output (sanitized prompt)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        self.examples = self._load_data()
        logger.info(f"Loaded {len(self.examples)} sanitization pairs from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file."""
        examples = []
        
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return examples
        
        with jsonlines.open(self.data_path) as reader:
            for obj in reader:
                if "source" in obj and "target" in obj:
                    examples.append(obj)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Tokenize source (malicious prompt)
        source_encoding = self.tokenizer(
            example["source"],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target (sanitized prompt)
        target_encoding = self.tokenizer(
            example["target"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
            "source_text": example["source"],
            "target_text": example["target"]
        }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for train, validation, and test sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

