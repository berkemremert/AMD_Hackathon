import json
from src.local_solvers import solve_ner

# Load your hackathon dataset
data = json.load(open('data/labeled_dataset.json'))

# Filter out only the NER tasks
ner_tasks = [t for t in data if t['category'] == 'named_entity_recognition']

print(f"Found {len(ner_tasks)} NER tasks in your dataset.")
print("Testing the 0-token local solver on the first 3 tasks...\n")

for i, task in enumerate(ner_tasks[:3]):
    print(f"--- Task {i+1} ---")
    print(f"Prompt: {task['prompt']}")
    print("\nLocal Solver Output (0 Tokens!):")
    print(solve_ner(task['prompt']))
    print("\n" + "="*50 + "\n")
