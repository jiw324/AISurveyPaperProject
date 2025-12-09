"""
ONE-LINE REPRODUCTION: python run_experiment.py

Transformer-Based Detection of Prompt Injection Attacks
Complete experiment - trains model and generates results for paper.

Author: Your Name
Date: December 2025
"""

import torch
import json
import logging
import argparse
import time
from pathlib import Path
from torch.utils.data import Subset

from src.data.dataset import PromptInjectionDataset, create_dataloaders
from src.models.classifier import load_model_and_tokenizer
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description="Transformer-Based Prompt Injection Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (recommended)
  python run_experiment.py
  
  # Full training
  python run_experiment.py --fraction 1.0 --epochs 5
  
  # Different model
  python run_experiment.py --model roberta-base
        """
    )
    
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                       help='Model name (default: distilbert-base-uncased)')
    parser.add_argument('--fraction', type=float, default=0.25,
                       help='Fraction of data to use (default: 0.25)')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results/)')
    
    args = parser.parse_args()
    
    # Check data exists
    if not Path("data/train.jsonl").exists():
        logger.error("Data not found! Please run: python scripts/download_data.py")
        return 1
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data fraction: {args.fraction*100:.0f}%")
    logger.info(f"Epochs: {args.epochs}")
    
    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, num_labels=2)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {num_params:,}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_full = PromptInjectionDataset("data/train.jsonl", tokenizer, 512)
    val_full = PromptInjectionDataset("data/val.jsonl", tokenizer, 512)
    test_full = PromptInjectionDataset("data/test.jsonl", tokenizer, 512)
    
    # Create subsets
    if args.fraction < 1.0:
        train_size = int(len(train_full) * args.fraction)
        val_size = int(len(val_full) * args.fraction)
        train_idx = list(range(0, len(train_full), int(1/args.fraction)))[:train_size]
        val_idx = list(range(0, len(val_full), int(1/args.fraction)))[:val_size]
        train_dataset = Subset(train_full, train_idx)
        val_dataset = Subset(val_full, val_idx)
    else:
        train_dataset = train_full
        val_dataset = val_full
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_full)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_full,
        batch_size=args.batch_size, num_workers=0
    )
    
    # Train
    logger.info("Starting training...")
    output_dir = Path(args.output_dir)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=2e-5,
        num_epochs=args.epochs,
        output_dir=str(output_dir),
        mixed_precision=torch.cuda.is_available()
    )
    
    start = time.time()
    results = trainer.train()
    training_time = time.time() - start
    
    # Save results
    output = {
        'model_name': args.model,
        'num_parameters': num_params,
        'data_fraction': args.fraction,
        'epochs': args.epochs,
        'training_time_minutes': training_time / 60,
        'best_val_f1': float(results['best_val_f1']),
        'test_metrics': {
            k: float(v) for k, v in results['test_metrics'].items()
        } if results['test_metrics'] else None
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETED!")
    logger.info("="*80)
    if results['test_metrics']:
        tm = results['test_metrics']
        logger.info(f"Test F1:        {tm['f1']:.4f}")
        logger.info(f"Test Precision: {tm['precision']:.4f}")
        logger.info(f"Test Recall:    {tm['recall']:.4f}")
        logger.info(f"FPR@95% Recall: {tm['fpr_at_95_recall']:.4f}")
    logger.info(f"Training time:  {training_time/60:.1f} minutes")
    logger.info(f"Results saved:  {output_dir / 'results.json'}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())

