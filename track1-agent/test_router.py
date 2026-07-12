import argparse
import json
from pathlib import Path

from output_optimizer import detect_task_type

# Map dataset prefixes/categories to the internal categories used by the router
CATEGORY_MAP = {
    "factual": "knowledge_qa",
    "math": "math_solving",
    "sentiment": "sentiment_analysis",
    "summary": "summarization",
    "ner": "entity_extraction",
    "debug": "bug_fixing",
    "logic": "logical_puzzles",
    "codegen": "code_authoring"
}

def main():
    parser = argparse.ArgumentParser(description="Test router accuracy on a dataset.")
    parser.add_argument("--dataset", type=str, default="data/track1_balanced_40_tasks.json", help="Path to the dataset JSON")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset {dataset_path} not found.")
        return
        
    tasks = json.loads(dataset_path.read_text())
    
    total = 0
    correct = 0
    
    # Track performance per category
    category_stats = {}
    
    for task in tasks:
        prompt = task["prompt"]
        
        # Determine the true category of the task
        true_category = task.get("category")
        if not true_category:
            task_id = task.get("task_id", task.get("id", ""))
            if "_" in task_id:
                true_category = task_id.split("_")[0]
            else:
                true_category = "unknown"
                
        expected_router_category = CATEGORY_MAP.get(true_category, true_category)
        
        # Run the router
        detected = detect_task_type(prompt)
        
        if expected_router_category not in category_stats:
            category_stats[expected_router_category] = {"total": 0, "correct": 0}
            
        category_stats[expected_router_category]["total"] += 1
        total += 1
        
        if detected == expected_router_category:
            correct += 1
            category_stats[expected_router_category]["correct"] += 1
        else:
            print(f"❌ Mismatch in {task.get('task_id', 'unknown')}:")
            print(f"   Expected: {expected_router_category}")
            print(f"   Detected: {detected}")
            print(f"   Prompt snippet: {prompt[:100]}...\n")

    print(f"\n--- Router Accuracy ---")
    print(f"Dataset: {args.dataset}")
    print(f"Overall Accuracy: {correct}/{total} ({(correct/total*100) if total else 0:.1f}%)")
    
    print("\nBreakdown by expected category:")
    for cat, stats in sorted(category_stats.items()):
        cat_correct = stats["correct"]
        cat_total = stats["total"]
        acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
        print(f"  • {cat:<20} | {cat_correct}/{cat_total} ({acc:.1f}%)")

if __name__ == "__main__":
    main()
