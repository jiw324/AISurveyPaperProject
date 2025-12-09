"""
Main entry point for Transformer-Based Prompt Injection Detection experiments.

One-line reproduction: python main.py --run-all --output-dir results/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup experiment environment and check requirements."""
    logger.info("Setting up environment...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available. Training will be slow on CPU.")
    
    # Create output directories
    Path("results/models").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/visualizations").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment setup complete.")


def run_experiment_1(args):
    """
    Experiment 1: Detection Performance Comparison
    Compare different transformer architectures for prompt injection detection.
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 1: Detection Performance Comparison")
    logger.info("=" * 80)
    
    from src.experiments.detection_comparison import DetectionExperiment
    
    experiment = DetectionExperiment(
        models=args.models,
        output_dir=args.output_dir,
        device=args.device
    )
    
    results = experiment.run()
    logger.info(f"Experiment 1 completed. Results saved to {args.output_dir}")
    return results


def run_experiment_2(args):
    """
    Experiment 2: Multi-Type Attack Detection
    Analyze performance across different attack categories.
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 2: Multi-Type Attack Detection")
    logger.info("=" * 80)
    
    from src.experiments.attack_analysis import AttackAnalysisExperiment
    
    experiment = AttackAnalysisExperiment(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    results = experiment.run()
    logger.info(f"Experiment 2 completed. Results saved to {args.output_dir}")
    return results


def run_experiment_3(args):
    """
    Experiment 3: Interpretability & Feature Analysis
    Understand what features transformers use to detect injections.
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 3: Interpretability & Feature Analysis")
    logger.info("=" * 80)
    
    from src.experiments.interpretability import InterpretabilityExperiment
    
    experiment = InterpretabilityExperiment(
        model_name=args.model,
        output_dir=args.output_dir,
        visualize=args.visualize,
        device=args.device
    )
    
    results = experiment.run()
    logger.info(f"Experiment 3 completed. Results saved to {args.output_dir}")
    return results


def run_experiment_4(args):
    """
    Experiment 4: Defense Mechanism - Sanitization
    Train seq2seq model to neutralize injections while preserving intent.
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 4: Defense Mechanism - Sanitization")
    logger.info("=" * 80)
    
    from src.experiments.sanitization import SanitizationExperiment
    
    experiment = SanitizationExperiment(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    results = experiment.run()
    logger.info(f"Experiment 4 completed. Results saved to {args.output_dir}")
    return results


def run_ablation_study(args):
    """
    Experiment 5: Ablation Studies
    Run various ablation experiments to understand model components.
    """
    logger.info("=" * 80)
    logger.info(f"ABLATION STUDY: {args.ablation}")
    logger.info("=" * 80)
    
    from src.experiments.ablations import AblationRunner
    
    runner = AblationRunner(
        ablation_type=args.ablation,
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    if args.ablation == "pretrain":
        results = runner.run_pretrain_ablation()
    elif args.ablation == "context":
        results = runner.run_context_ablation(sizes=args.sizes)
    elif args.ablation == "data_diversity":
        results = runner.run_data_diversity_ablation()
    elif args.ablation == "attention":
        results = runner.run_attention_ablation(architectures=args.compare)
    else:
        raise ValueError(f"Unknown ablation type: {args.ablation}")
    
    logger.info(f"Ablation study completed. Results saved to {args.output_dir}")
    return results


def run_experiment_6(args):
    """
    Experiment 6: Adversarial Robustness
    Test model robustness against adversarial evasion attempts.
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 6: Adversarial Robustness")
    logger.info("=" * 80)
    
    from src.experiments.adversarial_robustness import AdversarialExperiment
    
    experiment = AdversarialExperiment(
        model_name=args.model,
        attack_types=args.attacks,
        output_dir=args.output_dir,
        device=args.device
    )
    
    results = experiment.run()
    logger.info(f"Experiment 6 completed. Results saved to {args.output_dir}")
    return results


def run_all_experiments(args):
    """Run all experiments in sequence."""
    logger.info("=" * 80)
    logger.info("RUNNING ALL EXPERIMENTS")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    all_results = {}
    
    # Experiment 1: Detection Comparison
    args.models = ['bert', 'roberta', 'distilbert', 'deberta']
    all_results['experiment_1'] = run_experiment_1(args)
    
    # Use best model from Experiment 1 for subsequent experiments
    best_model = all_results['experiment_1']['best_model']
    args.model = best_model
    logger.info(f"Using best model '{best_model}' for remaining experiments")
    
    # Experiment 2: Attack Analysis
    all_results['experiment_2'] = run_experiment_2(args)
    
    # Experiment 3: Interpretability
    args.visualize = True
    all_results['experiment_3'] = run_experiment_3(args)
    
    # Experiment 4: Sanitization
    args.model = 't5-small'  # Use seq2seq model
    all_results['experiment_4'] = run_experiment_4(args)
    
    # Experiment 5: Ablations
    args.model = best_model
    ablations = ['pretrain', 'context', 'data_diversity', 'attention']
    all_results['ablations'] = {}
    
    for ablation in ablations:
        args.ablation = ablation
        if ablation == 'context':
            args.sizes = [64, 128, 256, 512]
        elif ablation == 'attention':
            args.compare = ['lstm', 'local', 'full']
        all_results['ablations'][ablation] = run_ablation_study(args)
    
    # Experiment 6: Adversarial Robustness
    args.attacks = ['paraphrase', 'substitute', 'gradient']
    all_results['experiment_6'] = run_experiment_6(args)
    
    # Generate final report
    from src.utils.reporting import generate_final_report
    generate_final_report(all_results, args.output_dir)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    logger.info("=" * 80)
    logger.info(f"ALL EXPERIMENTS COMPLETED in {duration:.2f} hours")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Transformer-Based Prompt Injection Detection Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (one-line reproduction)
  python main.py --run-all --output-dir results/
  
  # Run specific experiment
  python main.py --experiment detection_comparison --models bert roberta
  
  # Run ablation study
  python main.py --ablation pretrain --model bert
  
  # Run with custom config
  python main.py --config custom_config.yaml --run-all
        """
    )
    
    # General arguments
    parser.add_argument('--run-all', action='store_true',
                        help='Run all experiments in sequence')
    parser.add_argument('--experiment', type=str, choices=[
        'detection_comparison', 'attack_analysis', 'interpretability',
        'sanitization', 'adversarial'
    ], help='Specific experiment to run')
    parser.add_argument('--ablation', type=str, choices=[
        'pretrain', 'context', 'data_diversity', 'attention'
    ], help='Ablation study to run')
    parser.add_argument('--output-dir', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Experiment-specific arguments
    parser.add_argument('--models', type=str, nargs='+',
                        default=['bert', 'roberta', 'distilbert'],
                        help='Models to compare (Exp 1)')
    parser.add_argument('--model', type=str, default='roberta',
                        help='Model to use for single-model experiments')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations (Exp 3)')
    parser.add_argument('--attacks', type=str, nargs='+',
                        default=['paraphrase', 'substitute'],
                        help='Attack types for adversarial testing (Exp 6)')
    parser.add_argument('--sizes', type=int, nargs='+',
                        default=[64, 128, 256, 512],
                        help='Context window sizes (Ablation: context)')
    parser.add_argument('--compare', type=str, nargs='+',
                        default=['lstm', 'local', 'full'],
                        help='Architectures to compare (Ablation: attention)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.warning(f"Config file {args.config} not found. Using defaults.")
        config = {}
    
    try:
        # Run experiments
        if args.run_all:
            results = run_all_experiments(args)
        elif args.experiment == 'detection_comparison':
            results = run_experiment_1(args)
        elif args.experiment == 'attack_analysis':
            results = run_experiment_2(args)
        elif args.experiment == 'interpretability':
            results = run_experiment_3(args)
        elif args.experiment == 'sanitization':
            results = run_experiment_4(args)
        elif args.experiment == 'adversarial':
            results = run_experiment_6(args)
        elif args.ablation:
            results = run_ablation_study(args)
        else:
            parser.print_help()
            sys.exit(1)
        
        logger.info("Execution completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

