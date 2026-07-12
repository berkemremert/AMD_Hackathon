import json
from eval_agent import run_local_task, records
import sys

# find the sentiment records
sent_records = [r for r in records if r.get('category') == 'sentiment_classification']

for task in sent_records:
    prompt = task['prompt']
    from local_solvers import solve_sentiment
    output = solve_sentiment(prompt)
    if output:
        from eval_agent import verify_with_glm
        verdict, reason, tokens = verify_with_glm(task, output)
        print(f"[{verdict}] {reason}")

