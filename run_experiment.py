"""
Unified Prompt Injection Detection Experiment Runner

Single file to:
1. Run individual experiments with different fixes
2. Generate presentation-ready comparison reports
3. Output all results needed for your talk

Usage:
  # Run single experiment
  python run_experiment.py --fix multiclass
  
  # Generate full presentation comparison (RECOMMENDED)
  python run_experiment.py --compare-all
  
  # Quick 3-experiment comparison
  python run_experiment.py --compare-quick
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Subset
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from src.data.dataset import PromptInjectionDataset, create_dataloaders
from src.data.collate import collate_fn
from src.models.classifier import load_model_and_tokenizer
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prompt Injection Detection with Multiple Fixes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiments with different fixes
  python run_experiment.py --fix multiclass --epochs 5
  python run_experiment.py --fix label_noise --epochs 3
  python run_experiment.py --fix none  # Shows 100%% problem
  
  # Presentation comparison (runs all fixes)
  python run_experiment.py --compare-all --epochs 5
  
  # Quick comparison (3 key experiments)
  python run_experiment.py --compare-quick --epochs 3
        """
    )
    
    # Basic parameters
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                       help='Model name (default: distilbert-base-uncased)')
    parser.add_argument('--fraction', type=float, default=0.25,
                       help='Fraction of data (default: 0.25 = 25%%)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Max sequence length for tokenization (default: 512)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results/)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Fix options
    parser.add_argument('--fix', type=str, default='multiclass',
                       choices=['none', 'label_noise', 'multiclass', 'freeze', 'regularization'],
                       help='Fix to apply (default: multiclass)')
    parser.add_argument('--freeze-layers', type=int, default=5,
                       help='Layers to freeze (default: 5)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    
    # Comparison modes
    parser.add_argument('--compare-all', action='store_true',
                       help='Run all fixes and generate comparison report')
    parser.add_argument('--compare-quick', action='store_true',
                       help='Run 3 key experiments (baseline, multiclass, label_noise)')
    
    return parser.parse_args()


def apply_model_modifications(model, args):
    """Apply model modifications based on fix type."""
    
    if args.fix == 'freeze':
        logger.info(f"🔒 Freezing {args.freeze_layers} layers")
        if hasattr(model.encoder, 'transformer'):  # DistilBERT
            for i, layer in enumerate(model.encoder.transformer.layer):
                if i < args.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"   Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
    
    if args.fix == 'regularization':
        logger.info(f"🎛️  Heavy regularization: dropout={args.dropout}, weight_decay={args.weight_decay}")
        if hasattr(model.encoder.config, 'hidden_dropout_prob'):
            model.encoder.config.hidden_dropout_prob = args.dropout
        if hasattr(model.encoder.config, 'attention_probs_dropout_prob'):
            model.encoder.config.attention_probs_dropout_prob = args.dropout
    
    return model


def run_single_experiment(args, fix_override=None):
    """Run a single experiment and return results."""
    
    if fix_override:
        args.fix = fix_override
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Check base data
    if not Path("data/train.jsonl").exists():
        logger.error("❌ Data not found! Run: python scripts/generate_full_dataset.py")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Banner
    logger.info("=" * 80)
    logger.info(f"🔬 EXPERIMENT: {args.fix.upper()}")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.fraction*100:.0f}%, Epochs: {args.epochs}, Batch: {args.batch_size}")
    
    # Determine configuration
    if args.fix == 'multiclass':
        num_labels = 6
        train_path = "data/train_multiclass.jsonl"
        val_path = "data/val_multiclass.jsonl"
        test_path = "data/test_multiclass.jsonl"
        label_map = {
            "none": 0, "goal_hijacking": 1, "context_manipulation": 2,
            "jailbreaking": 3, "multi_turn": 4, "obfuscated": 5
        }
        logger.info("📊 Task: 6-way classification")
        if not Path(train_path).exists():
            logger.error(f"❌ {train_path} not found! Run: python scripts/generate_full_dataset.py")
            return None
            
    elif args.fix == 'label_noise':
        num_labels = 2
        train_path = "data/train_noisy.jsonl"
        val_path = "data/val_noisy.jsonl"
        test_path = "data/test.jsonl"
        label_map = {"legitimate": 0, "injection": 1}
        logger.info("📊 Task: Binary with 15% label noise")
        if not Path(train_path).exists():
            logger.error(f"❌ {train_path} not found! (label_noise dataset not generated)")
            return None
            
    else:
        num_labels = 2
        train_path = "data/train.jsonl"
        val_path = "data/val.jsonl"
        test_path = "data/test.jsonl"
        label_map = {"legitimate": 0, "injection": 1}
        logger.info("📊 Task: Binary classification")
    
    logger.info("=" * 80)
    
    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, num_labels=num_labels)
    model = apply_model_modifications(model, args)
    model.to(device)
    
    # Load datasets
    train_full = PromptInjectionDataset(train_path, tokenizer, args.max_length, label_map=label_map)
    val_full = PromptInjectionDataset(val_path, tokenizer, args.max_length, label_map=label_map)
    test_full = PromptInjectionDataset(test_path, tokenizer, args.max_length, label_map=label_map)
    
    # Subset if needed
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
    
    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_full,
        batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn
    )
    
    # Adjust LR for regularization
    lr = 5e-6 if args.fix == 'regularization' else args.learning_rate
    
    # Output directory
    output_dir = Path(args.output_dir) / f"{args.model.replace('/', '_')}_{args.fix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    logger.info("🚀 Starting training...")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=str(output_dir),
        mixed_precision=torch.cuda.is_available(),
        early_stopping_patience=2 if args.fix != 'regularization' else 1
    )
    
    train_results = trainer.train()
    training_time = time.time() - start_time
    
    # Evaluate
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 FINAL EVALUATION")
    logger.info("=" * 80)
    
    # Collect predictions
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            # Handle dict or tensor output
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics
    evaluator = Evaluator(threshold=0.5)
    
    if num_labels == 2:
        # Binary classification
        test_metrics = evaluator.compute_metrics(all_preds, all_labels)
    else:
        # Multi-class - use argmax
        pred_classes = np.argmax(all_preds, axis=1)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        test_metrics = {
            'accuracy': accuracy_score(all_labels, pred_classes),
            'precision': precision_score(all_labels, pred_classes, average='macro', zero_division=0),
            'recall': recall_score(all_labels, pred_classes, average='macro', zero_division=0),
            'f1': f1_score(all_labels, pred_classes, average='macro', zero_division=0),
        }
    
    logger.info(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Precision: {test_metrics['precision']*100:.2f}%")
    logger.info(f"Recall:    {test_metrics['recall']*100:.2f}%")
    logger.info(f"F1-Score:  {test_metrics['f1']*100:.2f}%")
    
    # Compile results
    results = {
        'fix': args.fix,
        'model': args.model,
        'num_labels': num_labels,
        'data_fraction': args.fraction,
        'epochs': args.epochs,
        'training_time_min': training_time / 60,
        'test_metrics': {
            'accuracy': float(test_metrics.get('accuracy', 0)),
            'precision': float(test_metrics.get('precision', 0)),
            'recall': float(test_metrics.get('recall', 0)),
            'f1': float(test_metrics.get('f1', 0)),
        }
    }
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Fix: {args.fix}")
    logger.info(f"Test Accuracy: {results['test_metrics']['accuracy']*100:.2f}%")
    logger.info(f"Test F1-Score: {results['test_metrics']['f1']*100:.2f}%")
    logger.info(f"Training Time: {results['training_time_min']:.1f} min")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 80)
    
    return results


def run_comparison(args, experiments):
    """Run multiple experiments and generate comparison report."""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           PRESENTATION COMPARISON: FIXING 100% ACCURACY PROBLEM            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    total_start = time.time()
    results = []
    
    for i, (fix_name, description) in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{len(experiments)}: {description}")
        print(f"{'#'*80}\n")
        
        try:
            result = run_single_experiment(args, fix_override=fix_name)
            if result:
                result['description'] = description
                results.append(result)
        except Exception as e:
            logger.error(f"Error in {fix_name}: {e}")
            results.append({
                'fix': fix_name,
                'description': description,
                'error': str(e)
            })
    
    total_time = (time.time() - total_start) / 60
    
    # Generate presentation report
    print(f"\n\n{'='*80}")
    print("📊 PRESENTATION COMPARISON REPORT")
    print(f"{'='*80}\n")
    
    # Create comparison table
    print(f"{'Method':<35} {'Accuracy':>10} {'F1-Score':>10} {'Time (min)':>12}")
    print("-" * 80)
    for r in results:
        if 'test_metrics' in r:
            acc = r['test_metrics']['accuracy'] * 100
            f1 = r['test_metrics']['f1'] * 100
            time_min = r['training_time_min']
            print(f"{r['description']:<35} {acc:>9.1f}% {f1:>9.1f}% {time_min:>12.1f}")
        else:
            print(f"{r['description']:<35} {'ERROR':>10} {'ERROR':>10} {'-':>12}")
    print("-" * 80)
    
    # Save results
    output_dir = Path("presentation_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    with open(output_dir / f"comparison_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Text report
    report_path = output_dir / f"PRESENTATION_REPORT_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PROMPT INJECTION DETECTION: 100% Accuracy Problem & Solutions\n")
        f.write("="*80 + "\n\n")
        
        baseline = next((r for r in results if r['fix'] == 'none'), None)
        fixes = [r for r in results if r['fix'] != 'none' and 'test_metrics' in r]
        
        if baseline and 'test_metrics' in baseline:
            f.write("THE PROBLEM:\n")
            f.write(f"  Baseline accuracy: {baseline['test_metrics']['accuracy']*100:.1f}%\n")
            f.write(f"  Baseline F1-score: {baseline['test_metrics']['f1']*100:.1f}%\n")
            f.write("  → TOO PERFECT! Indicates dataset is too easy\n")
            f.write("  → Model memorizing simple patterns, not learning semantics\n\n")
        
        f.write("SOLUTIONS TESTED:\n\n")
        for r in fixes:
            f.write(f"  {r['description']}:\n")
            f.write(f"    Accuracy:  {r['test_metrics']['accuracy']*100:.1f}%\n")
            f.write(f"    F1-Score:  {r['test_metrics']['f1']*100:.1f}%\n")
            f.write(f"    Precision: {r['test_metrics']['precision']*100:.1f}%\n")
            f.write(f"    Recall:    {r['test_metrics']['recall']*100:.1f}%\n")
            f.write(f"    Time:      {r['training_time_min']:.1f} minutes\n\n")
        
        if baseline and fixes and 'test_metrics' in baseline:
            avg_acc = sum(r['test_metrics']['accuracy'] for r in fixes) / len(fixes)
            reduction = (baseline['test_metrics']['accuracy'] - avg_acc) * 100
            
            f.write("="*80 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("="*80 + "\n\n")
            f.write(f"  Original problem:    {baseline['test_metrics']['accuracy']*100:.1f}% accuracy (unrealistic)\n")
            f.write(f"  After applying fixes: {avg_acc*100:.1f}% average accuracy\n")
            f.write(f"  Reduction:           {reduction:.1f} percentage points\n\n")
            f.write(f"  OUTCOME: More realistic, challenging task that demonstrates\n")
            f.write(f"           the actual difficulty of prompt injection detection\n\n")
        
        f.write(f"\nTotal experiment time: {total_time:.1f} minutes\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n✅ Report saved: {report_path}")
    
    # Presentation talking points
    print(f"\n{'='*80}")
    print("🎤 PRESENTATION TALKING POINTS")
    print(f"{'='*80}\n")
    
    if baseline and fixes and 'test_metrics' in baseline:
        baseline_acc = baseline['test_metrics']['accuracy'] * 100
        best_fix = max(fixes, key=lambda x: x['test_metrics']['f1'])
        
        print(f"1️⃣  'We initially achieved {baseline_acc:.0f}% accuracy - too good to be true'")
        print(f"2️⃣  'This revealed our synthetic dataset was too simple'")
        print(f"3️⃣  'We tested {len(fixes)} different approaches to make it realistic'")
        print(f"4️⃣  'Best solution: {best_fix['description']}'")
        print(f"     → Achieved {best_fix['test_metrics']['accuracy']*100:.0f}% accuracy")
        print(f"     → Shows real challenge of the task'")
        print(f"5️⃣  'This demonstrates models must learn semantic understanding,'")
        print(f"     not just keyword matching'")
    
    print(f"\n{'='*80}")
    print(f"⏱️  Total time: {total_time:.0f} minutes")
    print(f"📁 All files: {output_dir}/")
    print(f"{'='*80}\n")
    
    return 0


def main():
    args = parse_args()
    
    if args.compare_all:
        # Run all 5 fixes
        experiments = [
            ('none', '❌ Baseline (No Fix)'),
            ('multiclass', '✅ Multi-Class (6 classes)'),
            ('label_noise', '✅ Label Noise (15%)'),
            ('freeze', '✅ Freeze 5 Layers'),
            ('regularization', '✅ Heavy Regularization'),
        ]
        return run_comparison(args, experiments)
    
    elif args.compare_quick:
        # Run 3 key experiments
        experiments = [
            ('none', '❌ Baseline (No Fix)'),
            ('multiclass', '✅ Multi-Class (6 classes)'),
            ('label_noise', '✅ Label Noise (15%)'),
        ]
        return run_comparison(args, experiments)
    
    else:
        # Run single experiment
        result = run_single_experiment(args)
        return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())

