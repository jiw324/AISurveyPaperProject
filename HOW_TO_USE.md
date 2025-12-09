# 🚀 How to Use: ONE Command for Everything

## ✅ **Single Main File: `run_experiment.py`**

Everything you need is in **ONE file**. No more confusion!

---

## 🎯 **For Your Presentation**

### **Quick Comparison (RECOMMENDED)** ⭐️

Runs 3 key experiments (~1-2 hours):
```bash
python run_experiment.py --compare-quick --epochs 5
```

**Generates:**
- ✅ Comparison table showing baseline vs. fixes
- ✅ Text report with talking points
- ✅ JSON data for further analysis
- ✅ All saved to `presentation_results/`

**What it runs:**
1. Baseline (no fix) - shows 100% problem
2. Multi-class classification - best solution
3. Label noise - alternative approach

---

### **Full Comparison (If you have time)**

Runs all 5 fixes (~3-4 hours):
```bash
python run_experiment.py --compare-all --epochs 5
```

**Additional fixes tested:**
4. Freeze layers
5. Heavy regularization

---

## 📊 **Run Individual Experiments**

Test each fix separately:

```bash
# Multi-class (BEST for paper)
python run_experiment.py --fix multiclass --epochs 5

# Label noise
python run_experiment.py --fix label_noise --epochs 5

# Baseline (shows the problem)
python run_experiment.py --fix none --epochs 5

# Freeze layers
python run_experiment.py --fix freeze --epochs 5

# Heavy regularization
python run_experiment.py --fix regularization --epochs 5
```

---

## 📁 **Output Files for Presentation**

After running `--compare-quick` or `--compare-all`, you'll get:

```
presentation_results/
├── comparison_TIMESTAMP.json           # Raw data
├── PRESENTATION_REPORT_TIMESTAMP.txt   # Ready-to-use report
└── Individual experiment results in results/
```

---

## 🎤 **Presentation Report Contents**

The `PRESENTATION_REPORT_*.txt` file contains:

1. **THE PROBLEM**
   - Shows baseline 100% accuracy
   - Explains why it's unrealistic

2. **SOLUTIONS TESTED**
   - Each fix with its accuracy/F1
   - Training time for each

3. **KEY FINDINGS**
   - Average improvement
   - Best solution
   - Takeaways

4. **Talking points ready to copy!**

---

## ⚡ **Quick Start (5 minutes)**

```bash
# 1. Make sure data is generated
python scripts/download_data.py

# 2. Generate comparison datasets
python fix_100_percent_option1_label_noise.py
python fix_100_percent_option3_multiclass.py

# 3. Run comparison for presentation
python run_experiment.py --compare-quick --epochs 3

# 4. Check results
cat presentation_results/PRESENTATION_REPORT_*.txt
```

---

## 🎯 **Example Output**

```
================================================================================
📊 PRESENTATION COMPARISON REPORT
================================================================================

Method                              Accuracy   F1-Score   Time (min)
--------------------------------------------------------------------------------
❌ Baseline (No Fix)                    99.2%      99.1%         35.2
✅ Multi-Class (6 classes)              72.5%      71.8%         38.7
✅ Label Noise (15%)                    82.3%      81.5%         36.1
--------------------------------------------------------------------------------

KEY FINDINGS:
  Original problem:     99.2% accuracy (unrealistic)
  After applying fixes: 77.4% average accuracy
  Reduction:            21.8 percentage points
  
  OUTCOME: More realistic, challenging task
```

---

## 🔧 **Command Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--compare-quick` | - | Run 3 key experiments |
| `--compare-all` | - | Run all 5 experiments |
| `--fix` | multiclass | Which fix to use |
| `--epochs` | 5 | Number of epochs |
| `--fraction` | 0.25 | Data fraction (25%) |
| `--batch-size` | 32 | Batch size |

---

## ✅ **That's It!**

**One file. One command. All your presentation results.**

```bash
python run_experiment.py --compare-quick --epochs 5
```

Then use the generated report in `presentation_results/` for your talk! 🎉

