"""
Quick script to generate and test with the new realistic data generator.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               ALTERNATIVE DATA GENERATION APPROACH                           ║
║                                                                              ║
║  Using: Context-rich, natural language examples                             ║
║  Goal: More realistic, harder to classify                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("🔄 Step 1: Generating NEW realistic dataset...")
    print("=" * 80)
    result = subprocess.run([sys.executable, "scripts/generate_realistic_data.py"], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("❌ Error:", result.stderr)
        return 1
    
    print("\n✅ Data generated!")
    print("\n" + "=" * 80)
    print("🧪 Step 2: Testing with new data...")
    print("=" * 80)
    print("\nWe need to update run_experiment.py to use *_realistic.jsonl files")
    print("\nOptions:")
    print("  A) Quick test (1 epoch, 10% data) - 5 minutes")
    print("  B) Full test (5 epochs, 25% data) - 30-40 minutes")
    print("  C) Manual (I'll update and you run later)")
    
    choice = input("\nChoice (A/B/C): ").strip().upper()
    
    if choice == 'C':
        print("\n📝 Manual steps:")
        print("1. Update dataset paths in run_experiment.py:")
        print("   Change: 'data/train.jsonl' → 'data/train_realistic.jsonl'")
        print("   Change: 'data/val.jsonl' → 'data/val_realistic.jsonl'")
        print("   Change: 'data/test.jsonl' → 'data/test_realistic.jsonl'")
        print("\n2. Run: python run_experiment.py --fix none --epochs 5")
        return 0
    
    # Update paths temporarily
    print("\n🔧 Updating dataset paths...")
    
    # Read run_experiment.py
    with open('run_experiment.py', 'r') as f:
        content = f.read()
    
    # Backup
    with open('run_experiment.py.backup', 'w') as f:
        f.write(content)
    
    # Update paths
    content = content.replace('"data/train.jsonl"', '"data/train_realistic.jsonl"')
    content = content.replace('"data/val.jsonl"', '"data/val_realistic.jsonl"')
    content = content.replace('"data/test.jsonl"', '"data/test_realistic.jsonl"')
    
    with open('run_experiment.py', 'w') as f:
        f.write(content)
    
    print("✓ Paths updated (backup saved as run_experiment.py.backup)")
    
    # Run experiment
    if choice == 'A':
        print("\n🚀 Running quick test (1 epoch, 10% data)...")
        cmd = [sys.executable, "run_experiment.py", "--fix", "none", 
               "--epochs", "1", "--fraction", "0.1"]
    else:  # B
        print("\n🚀 Running full test (5 epochs, 25% data)...")
        cmd = [sys.executable, "run_experiment.py", "--fix", "none", 
               "--epochs", "5", "--fraction", "0.25"]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis may take a while. Results will show if data is more realistic...")
    print("=" * 80 + "\n")
    
    result = subprocess.run(cmd)
    
    # Restore original
    print("\n🔄 Restoring original run_experiment.py...")
    with open('run_experiment.py.backup', 'r') as f:
        content = f.read()
    with open('run_experiment.py', 'w') as f:
        f.write(content)
    Path('run_experiment.py.backup').unlink()
    
    print("✓ Restored")
    
    print("\n" + "=" * 80)
    print("📊 COMPARISON")
    print("=" * 80)
    print("\nCheck if accuracy is:")
    print("  • Still ~100%? → Need even more realistic data or different approach")
    print("  • 70-85%? → Good! More realistic task")
    print("  • 50-65%? → Very challenging, might be too hard")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

