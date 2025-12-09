"""
Option 2: Reduce model capacity by freezing most layers
Use only the classification head with frozen encoder
"""
import torch
from src.models.classifier import PromptInjectionClassifier
from transformers import AutoTokenizer

def create_frozen_model(model_name='distilbert-base-uncased', freeze_layers=5):
    """Create a model with most layers frozen."""
    print(f"🔧 Option 2: Freezing {freeze_layers} out of 6 layers")
    print("="*60)
    print("This reduces model capacity, making memorization harder")
    print("="*60 + "\n")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PromptInjectionClassifier(model_name, num_labels=2, freeze_encoder=False)
    
    # Freeze specific transformer layers
    if hasattr(model.encoder, 'transformer'):  # DistilBERT
        num_layers = len(model.encoder.transformer.layer)
        print(f"Found {num_layers} transformer layers")
        
        for i, layer in enumerate(model.encoder.transformer.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"  ✓ Froze layer {i}")
            else:
                print(f"  ○ Training layer {i}")
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    return model, tokenizer

def modify_run_experiment():
    """Instructions to modify run_experiment.py"""
    print("\n" + "="*60)
    print("📝 TO USE THIS:")
    print("="*60)
    print("In run_experiment.py, replace the model loading:")
    print("""
OLD:
    model, tokenizer = load_model_and_tokenizer(args.model_name)

NEW:
    from fix_100_percent_option2_freeze_model import create_frozen_model
    model, tokenizer = create_frozen_model(args.model_name, freeze_layers=5)
""")
    print("\n📊 Expected: 70-80% accuracy with reduced capacity")

if __name__ == "__main__":
    model, tokenizer = create_frozen_model()
    modify_run_experiment()

