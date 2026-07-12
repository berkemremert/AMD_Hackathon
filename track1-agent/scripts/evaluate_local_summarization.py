import json
import sys
import os
import time

# Ensure we can import from track1-agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.local_summarization.config import get_mode
from src.local_summarization.service import summarize

def evaluate():
    print(f"Running Local Summarization Evaluation (Mode: {get_mode()})")
    
    try:
        data = json.load(open('data/labeled_dataset.json'))
    except FileNotFoundError:
        print("Dataset not found. Please ensure data/labeled_dataset.json exists.")
        return

    tasks = [t for t in data if t['category'] == 'text_summarization' and len(t['prompt'].split()) > 100][:5]
    
    success_count = 0
    total_latency = 0
    total_original_words = 0
    total_compressed_words = 0
    results = []

    print(f"Evaluating {len(tasks)} tasks...")

    for i, task in enumerate(tasks, 1):
        task_id = task.get('id', task.get('task_id', f'unknown_{i}'))
        print(f"\nTask {i}/{len(tasks)}: {task_id}")
        start_time = time.time()
        
        result = summarize(task['prompt'])
        
        latency = (time.time() - start_time) * 1000
        total_latency += latency
        
        total_original_words += result.original_word_count
        total_compressed_words += result.compressed_word_count

        # Use the old evaluation system (Fireworks GLM-5p2 Judge)
        import data.label_dataset as data_labels
        from fireworks_client import chat
        judge_model = os.environ.get("MODEL_JUDGE", "accounts/fireworks/models/glm-5p2")
        
        judge_prompt = data_labels.JUDGE_PROMPT.format(
            prompt=task['prompt'],
            ground_truth=task.get('ground_truth', 'N/A'),
            cheap_answer=result.answer,
            expensive_answer="N/A"
        )
        verdict = chat(judge_model, judge_prompt, max_tokens=100)
        
        # Parse verdict
        is_correct = False
        import re
        if "```json" in verdict["text"]:
            try:
                json_block = verdict["text"].split("```json")[1].split("```")[0].strip()
                v_data = json.loads(json_block)
                if isinstance(v_data, list):
                    is_correct = v_data[0].get("correctness") == "CORRECT"
                else:
                    is_correct = v_data.get("correctness") == "CORRECT"
            except:
                pass
        else:
            if "CORRECT" in verdict["text"].upper() and "INCORRECT" not in verdict["text"].upper():
                is_correct = True
            elif "A: -> CORRECT" in verdict["text"] or "A: -> <CORRECT>" in verdict["text"]:
                is_correct = True
                
        status = "✅ PASS" if is_correct else "❌ FAIL"
        if is_correct:
            success_count += 1
            
        print(f"[{status}] Latency: {latency:.0f}ms | Tokens: {result.original_word_count}->{result.compressed_word_count} | Attempts: {len(result.attempts)}")
        if not is_correct:
            print(f"Errors: {result.validation.errors}")
            
        results.append({
            "task_id": task_id,
            "status": "PASS" if is_correct else "FAIL",
            "latency_ms": latency,
            "original_words": result.original_word_count,
            "compressed_words": result.compressed_word_count,
            "compression_applied": result.compression_applied,
            "attempts": len(result.attempts),
            "errors": result.validation.errors,
            "final_answer": result.answer
        })

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Accuracy: {success_count}/{len(tasks)} ({(success_count/len(tasks))*100:.1f}%)")
    print(f"Average Latency: {total_latency/len(tasks):.0f} ms")
    print(f"Average Source Words: {total_original_words/len(tasks):.1f}")
    if total_original_words > 0:
        print(f"Compression Ratio: {(total_compressed_words/total_original_words)*100:.1f}% of original size")
    
    with open("evaluation_local_summarization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to evaluation_local_summarization.json")

if __name__ == "__main__":
    evaluate()
