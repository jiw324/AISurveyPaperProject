"""
Option 1: Add 15% label noise to make the task harder
This flips labels randomly to create ambiguity
"""
import jsonlines
import random

def add_label_noise(input_file, output_file, noise_rate=0.15):
    """Flip labels for noise_rate fraction of examples."""
    data = list(jsonlines.open(input_file))
    
    num_to_flip = int(len(data) * noise_rate)
    indices_to_flip = random.sample(range(len(data)), num_to_flip)
    
    for idx in indices_to_flip:
        # Flip the label
        if data[idx]['label'] == 'injection':
            data[idx]['label'] = 'legitimate'
            data[idx]['original_label'] = 'injection'  # Track for analysis
        else:
            data[idx]['label'] = 'injection'
            data[idx]['original_label'] = 'legitimate'
    
    with jsonlines.open(output_file, 'w') as writer:
        for item in data:
            writer.write(item)
    
    print(f"✅ Added {noise_rate*100:.0f}% label noise to {input_file}")
    print(f"   Flipped {num_to_flip}/{len(data)} labels")
    print(f"   Saved to {output_file}")

if __name__ == "__main__":
    random.seed(42)
    
    print("🔧 Option 1: Adding Label Noise")
    print("="*60)
    print("This makes the task harder by intentionally mislabeling")
    print("15% of examples. The model must learn despite noisy labels.")
    print("="*60 + "\n")
    
    add_label_noise('data/train.jsonl', 'data/train_noisy.jsonl', noise_rate=0.15)
    add_label_noise('data/val.jsonl', 'data/val_noisy.jsonl', noise_rate=0.10)
    
    print("\n✅ Now run: python run_experiment.py")
    print("   But first modify dataset.py to use *_noisy.jsonl files")
    print("\n📊 Expected: 75-85% accuracy (realistic with label noise)")

