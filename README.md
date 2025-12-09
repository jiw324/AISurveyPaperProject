# Transformer-Based Detection and Defense Against Prompt Injection Attacks

A research project investigating the use of transformer models to detect and defend against prompt injection attacks on Large Language Models.

## 🎯 Project Overview

This project implements and evaluates multiple transformer architectures for detecting prompt injection attacks in real-time. We also explore defensive mechanisms including prompt sanitization and adversarial robustness techniques.

**Research Questions:**
- Can transformers effectively detect prompt injections with high accuracy and low false positives?
- How do different architectures compare for this security task?
- What linguistic features do models learn to identify attacks?
- Can we sanitize malicious prompts while preserving legitimate user intent?

## 📋 Requirements

- Python 3.11+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 20GB disk space for datasets and models

## 🚀 Quick Start - Reproduce All Experiments

**One-line reproduction:**
```bash
python main.py --run-all --output-dir results/
```

This will:
1. Download and prepare datasets
2. Train all models (BERT, RoBERTa, DistilBERT, DeBERTa)
3. Run all ablation studies
4. Generate evaluation metrics and visualizations
5. Save results to `results/` directory

**Estimated runtime:** ~40 hours on RTX 3090 (or ~20 hours on A100)

## 📦 Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/prompt-injection-detection.git
cd prompt-injection-detection
```

### Step 2: Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download datasets (optional - auto-downloads if not present)
```bash
python scripts/download_data.py
```

## 🔬 Running Individual Experiments

### Experiment 1: Architecture Comparison
```bash
python main.py --experiment detection_comparison --models bert roberta distilbert deberta
```

### Experiment 2: Attack Type Analysis
```bash
python main.py --experiment attack_analysis --model roberta
```

### Experiment 3: Interpretability Analysis
```bash
python main.py --experiment interpretability --model roberta --visualize
```

### Experiment 4: Sanitization Model
```bash
python main.py --experiment sanitization --model t5-small
```

### Experiment 5: Ablation Studies
```bash
# Ablation: Pre-training importance
python main.py --ablation pretrain --model bert

# Ablation: Context window size
python main.py --ablation context --sizes 64 128 256 512

# Ablation: Training data diversity
python main.py --ablation data_diversity

# Ablation: Attention mechanism
python main.py --ablation attention --compare lstm local full
```

### Experiment 6: Adversarial Robustness
```bash
python main.py --experiment adversarial --model roberta --attacks paraphrase substitute gradient
```

## 📊 Project Structure

```
prompt-injection-detection/
├── main.py                          # Main entry point for all experiments
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── EXPERIMENT_DESIGN.md            # Detailed experimental design
│
├── data/                            # Datasets (auto-downloaded)
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── adversarial_test.jsonl
│
├── src/
│   ├── data/
│   │   ├── dataset.py              # Dataset loading and preprocessing
│   │   ├── generators.py           # Synthetic attack generation
│   │   └── augmentation.py         # Data augmentation
│   │
│   ├── models/
│   │   ├── classifier.py           # Transformer classifiers
│   │   ├── sanitizer.py            # Seq2seq sanitization model
│   │   └── custom_transformer.py   # Small custom transformer
│   │
│   ├── training/
│   │   ├── trainer.py              # Training loop
│   │   ├── evaluator.py            # Evaluation metrics
│   │   └── ablations.py            # Ablation study implementations
│   │
│   ├── interpretability/
│   │   ├── attention_viz.py        # Attention visualization
│   │   ├── saliency.py             # Gradient-based saliency
│   │   └── probing.py              # Probing classifiers
│   │
│   ├── attacks/
│   │   ├── adversarial.py          # Adversarial attack generation
│   │   └── evasion_tests.py        # Evasion testing
│   │
│   └── utils/
│       ├── metrics.py              # Custom metrics
│       ├── visualization.py        # Plotting utilities
│       └── config.py               # Configuration management
│
├── scripts/
│   ├── download_data.py            # Dataset download script
│   ├── generate_attacks.py         # Attack generation
│   └── evaluate_all.py             # Batch evaluation
│
├── tests/                          # Unit tests
│   ├── test_models.py
│   ├── test_data.py
│   └── test_attacks.py
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── exploratory_analysis.ipynb
│   ├── results_visualization.ipynb
│   └── case_studies.ipynb
│
├── results/                        # Experiment outputs (gitignored)
│   ├── models/                     # Trained model checkpoints
│   ├── metrics/                    # Evaluation results
│   ├── visualizations/             # Plots and figures
│   └── logs/                       # Training logs
│
└── paper/                          # LaTeX paper
    ├── main.tex
    ├── sections/
    ├── figures/
    └── references.bib
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architectures and hyperparameters
- Training settings (batch size, learning rate, epochs)
- Dataset paths and split ratios
- Evaluation metrics and thresholds

Example:
```yaml
training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 5
  early_stopping_patience: 2

models:
  bert:
    pretrained: "bert-base-uncased"
    max_length: 512
  roberta:
    pretrained: "roberta-base"
    max_length: 512
```

## 📈 Results

After running experiments, view results:

```bash
# Generate summary report
python scripts/generate_report.py --results-dir results/

# Launch visualization dashboard
python scripts/dashboard.py --port 8080
```

Results will include:
- Model comparison table (accuracy, precision, recall, F1, latency)
- Per-attack-type performance breakdown
- Attention visualizations
- Ablation study results
- Adversarial robustness scores

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run integration tests:
```bash
pytest tests/ --integration
```

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{prompt-injection-detection-2025,
  author = {Your Name},
  title = {Transformer-Based Detection and Defense Against Prompt Injection Attacks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/prompt-injection-detection}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a research project. For questions or collaboration:
- Open an issue on GitHub
- Contact: your.email@university.edu

## 🙏 Acknowledgments

- Datasets: ShareGPT, Alpaca, LMSYS
- Pre-trained models: HuggingFace Transformers
- Inspiration: Recent work on LLM security (see EXPERIMENT_DESIGN.md for references)

## 📚 Additional Resources

- **Experiment Design:** See `EXPERIMENT_DESIGN.md` for detailed methodology
- **Paper Draft:** See `paper/main.tex` for LaTeX source
- **Notebooks:** See `notebooks/` for exploratory analysis and visualization

---

**Status:** 🚧 Under Development

**Last Updated:** December 2025

