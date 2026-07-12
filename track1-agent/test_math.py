import json
from eval_agent import run_local_task, records
import sys

# find the math records
math_records = [r for r in records if r.get('category') == 'math_reasoning']

for task in math_records:
    prompt = task['prompt']
    from local_solvers import solve_math_exact
    output = solve_math_exact(prompt)
    if output:
        from eval_agent import verify_with_glm
        verdict, reason, tokens = verify_with_glm(task, output)
        print(f"[{verdict}] {reason}")

