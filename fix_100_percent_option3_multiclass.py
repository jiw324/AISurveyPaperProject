"""
Option 3: Change to multi-class classification (MUCH HARDER)
Instead of binary (injection/legitimate), classify attack TYPE
"""
import jsonlines

def convert_to_multiclass(input_file, output_file):
    """Convert binary labels to multi-class (6 classes)."""
    data = list(jsonlines.open(input_file))
    
    # Label mapping:
    # 0: legitimate
    # 1: goal_hijacking
    # 2: context_manipulation
    # 3: jailbreaking
    # 4: multi_turn
    # 5: obfuscated
    
    label_map = {
        'none': 0,
        'goal_hijacking': 1,
        'context_manipulation': 2,
        'jailbreaking': 3,
        'multi_turn': 4,
        'obfuscated': 5
    }
    
    for item in data:
        attack_type = item['attack_type']
        item['label'] = label_map[attack_type]
        item['label_name'] = attack_type
    
    with jsonlines.open(output_file, 'w') as writer:
        for item in data:
            writer.write(item)
    
    print(f"✅ Converted {input_file} to 6-class classification")
    print(f"   Classes: legitimate, goal_hijacking, context_manipulation,")
    print(f"            jailbreaking, multi_turn, obfuscated")

def modify_for_multiclass():
    """Instructions to modify code for multi-class."""
    print("\n" + "="*60)
    print("📝 TO USE MULTI-CLASS CLASSIFICATION:")
    print("="*60)
    print("""
1. In run_experiment.py, change:
   OLD: model = PromptInjectionClassifier(model_name, num_labels=2)
   NEW: model = PromptInjectionClassifier(model_name, num_labels=6)

2. In src/data/dataset.py, update label_map:
   OLD: label_map = {"legitimate": 0, "injection": 1}
   NEW: label_map = {
       "none": 0, "goal_hijacking": 1, "context_manipulation": 2,
       "jailbreaking": 3, "multi_turn": 4, "obfuscated": 5
   }

3. In src/training/evaluator.py, change metrics for multi-class:
   - Use 'macro' averaging for F1, precision, recall
   - Add confusion matrix
   - Remove AUC-ROC (or adapt for multi-class)
""")
    print("\n📊 Expected: 60-75% accuracy (much harder task!)")
    print("    Model must distinguish between 6 classes, not just 2")

if __name__ == "__main__":
    print("🔧 Option 3: Multi-Class Classification")
    print("="*60)
    print("This is MUCH harder - model must classify the TYPE")
    print("of injection, not just detect if it's an injection")
    print("="*60 + "\n")
    
    convert_to_multiclass('data/train.jsonl', 'data/train_multiclass.jsonl')
    convert_to_multiclass('data/val.jsonl', 'data/val_multiclass.jsonl')
    convert_to_multiclass('data/test.jsonl', 'data/test_multiclass.jsonl')
    
    modify_for_multiclass()

