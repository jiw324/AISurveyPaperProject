"""
Option 4: Add heavy regularization and dropout
Makes the model less likely to overfit/memorize
"""

TRAINING_MODIFICATIONS = """
🔧 Option 4: Heavy Regularization
============================================================
Add these settings to prevent overfitting/memorization
============================================================

In run_experiment.py or config.yaml, modify training parameters:

1. INCREASE DROPOUT (add randomness during training):
   model.config.hidden_dropout_prob = 0.3  # Up from 0.1
   model.config.attention_probs_dropout_prob = 0.3  # Up from 0.1

2. INCREASE WEIGHT DECAY (prevent large weights):
   weight_decay = 0.1  # Up from 0.01 (10x stronger)

3. REDUCE LEARNING RATE (slower, more careful learning):
   learning_rate = 5e-6  # Down from 2e-5 (4x slower)

4. ADD LABEL SMOOTHING (soften targets):
   loss_fn = CrossEntropyLoss(label_smoothing=0.1)

5. USE SMALLER BATCH SIZE (more noise):
   batch_size = 8  # Down from 32

6. ADD EARLY STOPPING (stop before memorizing):
   early_stopping_patience = 1  # Stop after 1 epoch of no improvement

============================================================
IMPLEMENTATION:
============================================================
"""

def create_regularized_config():
    """Create a config file with heavy regularization."""
    import yaml
    
    config = {
        'training': {
            'learning_rate': 5e-6,  # Much lower
            'weight_decay': 0.1,  # Much higher
            'batch_size': 8,  # Much smaller
            'gradient_accumulation_steps': 4,  # To maintain effective batch size
            'early_stopping_patience': 1,  # Stop early
            'dropout': 0.3,  # High dropout
            'label_smoothing': 0.1,  # Smooth labels
            'max_grad_norm': 0.5,  # Clip gradients more aggressively
        }
    }
    
    with open('config_regularized.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("✅ Created config_regularized.yaml")
    print("\n📊 Expected: 70-85% accuracy (heavy regularization prevents overfitting)")
    print("\nTo use: python run_experiment.py --config config_regularized.yaml")

if __name__ == "__main__":
    print(TRAINING_MODIFICATIONS)
    try:
        create_regularized_config()
    except ImportError:
        print("⚠️  PyYAML not installed. Manual modification needed.")
        print("    Install with: pip install pyyaml")

