# Quick Start Guide

## Overview

This project implements transformer-based detection and defense systems for prompt injection attacks. This guide will help you get started quickly.

## 🎯 What's Been Set Up

Your project now includes:

1. **Comprehensive Experiment Design** (`EXPERIMENT_DESIGN.md`)
   - 6 main experiments with detailed methodology
   - 4 ablation studies
   - Clear research questions and expected outcomes
   - Resource estimation and timeline

2. **Complete Project Structure**
   - Main execution script (`main.py`)
   - Configuration file (`config.yaml`)
   - Source code templates in `src/`
   - Data generation script in `scripts/`
   - LaTeX paper template in `paper/`

3. **Full Documentation**
   - `README.md` with reproduction instructions
   - `requirements.txt` with all dependencies
   - `.gitignore` configured for Python/ML projects

## 🚀 Next Steps

### 1. Set Up Environment (5 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for text processing)
python -m spacy download en_core_web_sm
```

### 2. Generate Initial Dataset (10 minutes)

```bash
python scripts/download_data.py
```

This creates synthetic datasets in `data/`:
- `train.jsonl` (50,000 samples)
- `val.jsonl` (10,000 samples)
- `test.jsonl` (10,000 samples)
- `adversarial_test.jsonl` (2,000 samples)

### 3. Run a Quick Test (20 minutes on GPU)

Test the setup with a small experiment:

```bash
# Train a single model (DistilBERT - fastest)
python main.py --experiment detection_comparison --models distilbert --output-dir results/test/
```

### 4. Run Full Experiments (40 hours on RTX 3090)

Once everything works, run all experiments:

```bash
python main.py --run-all --output-dir results/
```

## 📊 What Each Experiment Does

### Experiment 1: Architecture Comparison
Compares BERT, RoBERTa, DistilBERT, DeBERTa for detection performance.

**Run separately:**
```bash
python main.py --experiment detection_comparison --models bert roberta distilbert deberta
```

### Experiment 2: Attack Type Analysis
Analyzes how well the model detects different attack categories.

**Run separately:**
```bash
python main.py --experiment attack_analysis --model roberta
```

### Experiment 3: Interpretability
Visualizes attention patterns and identifies important features.

**Run separately:**
```bash
python main.py --experiment interpretability --model roberta --visualize
```

### Experiment 4: Sanitization
Trains a seq2seq model to remove injections while keeping legitimate content.

**Run separately:**
```bash
python main.py --experiment sanitization --model t5-small
```

### Experiment 5: Ablations
Tests what components are critical for performance.

**Run separately:**
```bash
# Test importance of pre-training
python main.py --ablation pretrain --model bert

# Test context window size
python main.py --ablation context --sizes 64 128 256 512

# Test training data diversity
python main.py --ablation data_diversity

# Test attention mechanism
python main.py --ablation attention --compare lstm local full
```

### Experiment 6: Adversarial Robustness
Tests if attackers can evade detection.

**Run separately:**
```bash
python main.py --experiment adversarial --model roberta --attacks paraphrase substitute
```

## 📝 What to Implement Next

The current setup provides:
- ✅ Project structure
- ✅ Main execution framework
- ✅ Configuration system
- ✅ Dataset classes
- ✅ Model architectures
- ✅ Data generation script

**You need to implement:**

1. **Training logic** (`src/training/trainer.py`)
   - Training loop with forward/backward passes
   - Validation and checkpointing
   - Early stopping

2. **Evaluation metrics** (`src/training/evaluator.py`)
   - Calculate accuracy, precision, recall, F1
   - ROC-AUC curves
   - Per-attack-type analysis

3. **Experiment runners** (`src/experiments/`)
   - `detection_comparison.py` - Run multiple models
   - `attack_analysis.py` - Per-type evaluation
   - `interpretability.py` - Attention visualization
   - `sanitization.py` - Seq2seq training
   - `ablations.py` - Ablation study logic
   - `adversarial_robustness.py` - Adversarial testing

4. **Utilities** (`src/utils/`)
   - `metrics.py` - Custom metric functions
   - `visualization.py` - Plotting functions
   - `reporting.py` - Generate final reports

## 🎓 Meeting Project Requirements

Your project already addresses all requirements:

### ✅ Research Questions (Requirement 1-2)
See `EXPERIMENT_DESIGN.md` sections 1-2

### ✅ Experimental Plan (Requirement 3)
See `EXPERIMENT_DESIGN.md` section 3 with 6 detailed experiments

### ✅ Resource Estimation (Requirement 4)
See `EXPERIMENT_DESIGN.md` section 4 (~38 hours GPU time)

### ✅ Ablation Study (Required)
**4 ablations planned:**
1. Pre-training importance (random vs. pre-trained)
2. Context window size (64, 128, 256, 512 tokens)
3. Training data diversity (remove each attack type)
4. Attention mechanism (LSTM vs. local vs. full attention)

### ✅ One-Line Reproduction (Required)
```bash
python main.py --run-all --output-dir results/
```

### ✅ Python3 + PyTorch (Required)
- Python 3.11 specified
- PyTorch 2.1.0 in requirements.txt
- All models use PyTorch

### ✅ GitHub Repository Structure (Required)
- README.md with clear instructions
- requirements.txt with all dependencies
- Organized code structure
- LaTeX paper template

### ✅ Results & Learning (Requirement 5-6)
Paper template includes sections for results, discussion, and future work

## 💡 Tips for Success

1. **Start Small**: Test with DistilBERT first (fastest training)
2. **Monitor Training**: Use TensorBoard or Weights & Biases
3. **Save Checkpoints**: Training can be interrupted
4. **Iterate on Data**: If results are poor, improve the synthetic data
5. **Real Data**: Consider collecting real prompt injection examples
6. **Baseline**: Implement a simple keyword-based baseline to compare against

## 📚 Additional Resources

### Recommended Reading
1. "Ignore Previous Prompt" (Perez et al., 2022)
2. "Jailbroken: How Does LLM Safety Training Fail?" (Wei et al., 2023)
3. "Prompt Injection Attacks and Defenses" (Liu et al., 2023)

### Useful Links
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Prompt Injection Examples: https://github.com/agencyenterprise/promptinject

## 🐛 Troubleshooting

**CUDA Out of Memory:**
- Reduce batch_size in config.yaml (try 16 or 8)
- Use DistilBERT instead of BERT/RoBERTa
- Enable gradient accumulation

**Slow Training:**
- Enable mixed precision (fp16: true in config.yaml)
- Use fewer dataloader workers if CPU is bottleneck
- Consider gradient checkpointing

**Poor Results:**
- Check data quality (inspect train.jsonl)
- Try different learning rates (1e-5, 2e-5, 5e-5)
- Increase training epochs
- Add more diverse attack examples

## 📧 Questions?

Review the experiment design document for detailed methodology, or check the paper template for result presentation format.

Good luck with your research! 🚀

