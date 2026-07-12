import json
from pathlib import Path
from src.output_optimizer import detect_task_type

DATA_PATH = Path("data/labeled_dataset.json")

def main():
    records = json.loads(DATA_PATH.read_text())
    
    old_tokens = 0
    new_tokens = 0
    ner_count = 0
    
    for r in records:
        # Simulate Old Pipeline (using the ground-truth label as proxy for router)
        label = r["label"]
        if label == "hard":
            old_tokens += r["expensive_tokens"]
        else:
            old_tokens += r["cheap_tokens"]
            
        # Simulate New Pipeline (Local NER intercept)
        task_type = detect_task_type(r["prompt"])
        if task_type == "entity_extraction":
            # 0 tokens!
            new_tokens += 0
            ner_count += 1
        else:
            if label == "hard":
                new_tokens += r["expensive_tokens"]
            else:
                new_tokens += r["cheap_tokens"]
                
    savings = old_tokens - new_tokens
    print("=== Token Simulation Results ===")
    print(f"Old Architecture Tokens: {old_tokens:,}")
    print(f"New Architecture Tokens: {new_tokens:,}")
    print(f"\nTotal Tokens Saved:      {savings:,} (-{(savings/old_tokens)*100:.1f}%)")
    print(f"NER Queries Intercepted: {ner_count} out of {len(records)}")
    
if __name__ == "__main__":
    main()
