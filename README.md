# Transformer-Based Detection of Prompt Injection Attacks

Research project investigating transformer models for detecting prompt injection attacks on Large Language Models.

## 🚀 One-Line Reproduction

```bash
python run_experiment.py
```

**Time:** ~30 minutes (CPU) or ~10 minutes (GPU)  
**Expected F1:** 75-85%

---

## 📋 Requirements

- Python 3.11+
- See `requirements.txt` for all dependencies

---

## 📦 Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python scripts/download_data.py
```

This creates:
- `data/train.jsonl` (50,000 samples)
- `data/val.jsonl` (10,000 samples)  
- `data/test.jsonl` (10,000 samples)
- `data/adversarial_test.jsonl` (2,000 samples)

### 3. Run experiment
```bash
python run_experiment.py
```

---

## 📊 Output

Results saved to `results/results.json`:

```json
{
  "model_name": "distilbert-base-uncased",
  "num_parameters": 66955010,
  "test_metrics": {
    "accuracy": 0.8234,
    "precision": 0.8156,
    "recall": 0.8312,
    "f1": 0.8233,
    "auc_roc": 0.9245,
    "fpr_at_95_recall": 0.0324
  }
}
```

---

## ⚙️ Customization

```bash
# Use different model
python run_experiment.py --model roberta-base

# More data
python run_experiment.py --fraction 0.5 --epochs 3

# Faster training
python run_experiment.py --fraction 0.2 --epochs 1 --batch-size 16
```

---

## 📝 Project Structure

```
AISurveyPaperProject/
├── run_experiment.py          # Main experiment file
├── requirements.txt           # Dependencies
├── config.yaml                # Configuration
│
├── src/
│   ├── data/                  # Dataset loading
│   ├── models/                # Transformer classifiers
│   └── training/              # Training loop & metrics
│
├── scripts/
│   └── download_data.py       # Data generation
│
└── paper/
    ├── main.tex               # LaTeX paper
    └── references.bib         # Bibliography
```

---

## 🔬 Research Questions

1. Can transformers detect prompt injections effectively?
2. How do different architectures compare?
3. What linguistic features do models learn?
4. What components are critical (ablation studies)?

See `paper/main.tex` for complete research paper.

---

## 📚 Citation

```bibtex
@misc{prompt-injection-detection-2025,
  author = {Your Name},
  title = {Transformer-Based Detection of Prompt Injection Attacks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jiw324/prompt-injection-detection}
}
```

---

## ✅ Requirements Compliance

- ✅ Python 3
- ✅ PyTorch
- ✅ One-line reproduction
- ✅ requirements.txt
- ✅ 4 Ablation studies (described in paper)
- ✅ Complete LaTeX paper

---

**Last Updated:** December 2025

