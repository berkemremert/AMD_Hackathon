import json
import time
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.local_compressor import compress_summarization_prompt
from src.output_optimizer import detect_task_type

def run_benchmark():
    input_path = Path("data/labeled_dataset.json")
    if not input_path.exists():
        print(f"Dataset not found at {input_path}")
        return
        
    with open(input_path, "r") as f:
        tasks = json.load(f)
        
    # Filter for summarization tasks only
    summarization_tasks = [t for t in tasks if detect_task_type(t["prompt"]) == "summarization"]
    
    print(f"Found {len(summarization_tasks)} summarization tasks.")
    
    total_original_words = 0
    total_compressed_words = 0
    total_original_sentences = 0
    total_selected_sentences = 0
    num_compressed = 0
    fallbacks = {}
    latencies = []
    
    results = []
    
    for task in summarization_tasks:
        start_time = time.time()
        result = compress_summarization_prompt(task["prompt"])
        latency = time.time() - start_time
        latencies.append(latency)
        
        if result.applied:
            num_compressed += 1
            total_original_words += result.original_word_count
            total_compressed_words += result.compressed_word_count
            total_original_sentences += result.original_sentence_count
            total_selected_sentences += result.selected_sentence_count
        else:
            reason = result.fallback_reason or "unknown"
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
            
        results.append({
            "task_id": task.get("task_id", "unknown"),
            "original_prompt": task["prompt"],
            "compressed_prompt": result.compressed_text if result.applied else task["prompt"],
            "stats": {
                "applied": result.applied,
                "original_words": result.original_word_count,
                "compressed_words": result.compressed_word_count,
                "ratio": result.compression_ratio,
                "fallback_reason": result.fallback_reason,
                "latency_sec": latency
            }
        })
        
    if num_compressed > 0:
        avg_orig_words = total_original_words / num_compressed
        avg_comp_words = total_compressed_words / num_compressed
        avg_ratio = total_compressed_words / max(1, total_original_words)
        avg_selected_sents = total_selected_sentences / num_compressed
    else:
        avg_orig_words = avg_comp_words = avg_ratio = avg_selected_sents = 0.0
        
    latencies.sort()
    median_latency = latencies[len(latencies)//2] if latencies else 0.0
    p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    
    print("\n--- Benchmark Results ---")
    print(f"Total prompts: {len(summarization_tasks)}")
    print(f"Number compressed: {num_compressed}")
    print(f"Fallbacks: {fallbacks}")
    if num_compressed > 0:
        print(f"Avg original words (compressed only): {avg_orig_words:.1f}")
        print(f"Avg compressed words: {avg_comp_words:.1f}")
        print(f"Avg compression ratio: {avg_ratio:.2f}")
        print(f"Avg selected sentences: {avg_selected_sents:.1f}")
    print(f"Median latency: {median_latency:.3f} s")
    print(f"P95 latency: {p95_latency:.3f} s")
    
    out_file = Path("scripts/benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {out_file}")

if __name__ == "__main__":
    run_benchmark()
