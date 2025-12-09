# Transformer-Based Detection of Prompt Injection Attacks

Research project investigating transformer models for detecting prompt injection attacks on Large Language Models, using realistic/paraphrased hard data and multiclass evaluation to avoid inflated (100%) results.

## 🚀 One-Line Reproduction (after data is generated)

```bash
python run_experiment.py --fix multiclass --epochs 3 --fraction 0.25
```

**Time (recent run):** ~4–5 minutes on GPU (25% data, 3 epochs)  
**Result (hard data, multiclass):** 83.36% accuracy / 66.57% macro-F1

---

## 📋 Requirements

- Python 3.11+
- See `requirements.txt` for all dependencies

---

## 📦 Installation & Data

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Generate harder, realistic data
```bash
# Base realistic data (context-rich)
python scripts/generate_realistic_data.py

# Paraphrased/perturbed hard data
python scripts/generate_paraphrased_data.py

# Convert to multiclass (legit + 5 attack types)
python fix_100_percent_option3_multiclass.py
```

This produces:
- `data/train_multiclass.jsonl`, `data/val_multiclass.jsonl`, `data/test_multiclass.jsonl`

### 3) Run experiment (multiclass, hard data)
```bash
python run_experiment.py --fix multiclass --epochs 3 --fraction 0.25
```

---

## 📊 Output

Results saved to `results/{model_fix}/results.json` (e.g., `results/distilbert-base-uncased_multiclass/results.json`):

```json
{
  "fix": "multiclass",
  "model": "distilbert-base-uncased",
  "num_labels": 6,
  "data_fraction": 0.25,
  "epochs": 3,
  "training_time_min": 4.34,
  "test_metrics": {
    "accuracy": 0.8336,
    "precision": 0.6671,
    "recall": 0.6679,
    "f1": 0.6657
  }
}
```

---

## ⚙️ Customization

```bash
# Use different model
python run_experiment.py --model roberta-base

# More data
python run_experiment.py --fraction 1.0 --epochs 5

# Faster training
python run_experiment.py --fraction 0.1 --epochs 1 --batch-size 16
```

---

## 📝 Project Structure

```
AISurveyPaperProject/
├── run_experiment.py          # Main experiment file
├── requirements.txt           # Dependencies
├── config.yaml                # (optional) configuration
│
├── src/
│   ├── data/                  # Dataset loading
│   ├── models/                # Transformer classifiers
│   └── training/              # Training loop & metrics
│
├── scripts/
│   ├── download_data.py               # Original synthetic generator
│   ├── generate_realistic_data.py     # Context-rich generator
│   └── generate_paraphrased_data.py   # Paraphrased/perturbed hard data
│
├── fix_100_percent_option3_multiclass.py  # Converts to 6-way multiclass
├── use_realistic_data.py                  # Helper for realistic/hard pipeline
│
└── paper/
    ├── main.tex               # LaTeX paper
    └── references.bib         # Bibliography
```

---

## 🔬 Research Questions

1. Can transformers detect prompt injections effectively?
2. Which attack types are hardest (multiclass view)?
3. How to avoid inflated results from easy synthetic data?
4. What components are critical (ablation plan in paper)?

See `paper/main.tex` for the complete research paper.

---

## 📚 Citation

```bibtex
@misc{prompt-injection-detection-2025,
  author = {Jinghao Wang},
  title = {Transformer-Based Detection of Prompt Injection Attacks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jiw324/prompt-injection-detection}
}
```

---

## ✅ Requirements Compliance

- ✅ Python 3.11+
- ✅ PyTorch
- ✅ One-line reproduction (`run_experiment.py --fix multiclass ...`)
- ✅ requirements.txt
- ✅ Ablation plan in paper (encoder freeze vs full FT; follow-ups outlined)
- ✅ Complete LaTeX paper

---

**Last Updated:** December 2025

