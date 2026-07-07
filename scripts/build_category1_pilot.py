import json
import string
import random
from collections import defaultdict

SAMPLE_SIZE = 50
DATASET_PATH = "qa/wikipedia-dev.json"
OUTPUT_JSON_PATH = "data/category1_pilot_v1.json"
OUTPUT_README_PATH = "data/category1_pilot_v1_README.md"

# Blacklist of offensive/inappropriate words (basic heuristics)
BLACKLIST = {"rape", "murder", "kill", "nazi", "hitler", "suicide", "terrorist", "assassination", "sexual"}

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def determine_topic(question_text):
    text = question_text.lower()
    if any(w in text for w in ['who ', 'actor', 'actress', 'president', 'author', 'singer', 'person', 'whose']):
        return 'people'
    if any(w in text for w in ['where ', 'city', 'country', 'state', 'river', 'mountain', 'capital']):
        return 'places'
    if any(w in text for w in ['when ', 'war', 'battle', 'king ', 'queen ', 'century', 'history']):
        return 'history'
    if any(w in text for w in ['science', 'planet', 'biology', 'chemistry', 'physics', 'element', 'animal', 'body', 'disease']):
        return 'science'
    if any(w in text for w in ['sport', 'team', 'player', 'football', 'baseball', 'olympics', 'golf', 'tennis']):
        return 'sports'
    if any(w in text for w in ['movie', 'film', 'music', 'album', 'band', 'song', 'character', 'tv show', 'television']):
        return 'entertainment'
    if any(w in text for w in ['book', 'novel', 'poem', 'writer']):
        return 'literature'
    return 'other'

def determine_difficulty(aliases, answer_value):
    num_aliases = len(aliases)
    ans_len = len(answer_value)
    
    if num_aliases > 10 or ans_len < 10:
        return 'easy'
    elif num_aliases <= 3 and ans_len > 15:
        return 'hard'
    else:
        return 'medium'

def main():
    random.seed(42) # For reproducibility
    
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
        
    candidates = []
    seen_questions = set()
    
    for item in data['Data']:
        question = item['Question']
        norm_q = normalize_text(question)
        
        # Check blacklist
        if any(bad_word in norm_q for bad_word in BLACKLIST):
            continue
            
        # Check length
        words = norm_q.split()
        if len(words) < 5 or len(words) > 40:
            continue
            
        # Deduplicate
        if norm_q in seen_questions:
            continue
            
        # Check evidence
        if not item.get('EntityPages'):
            continue
            
        # Ensure single primary answer (using MatchedWikiEntityName)
        answer_data = item.get('Answer', {})
        if not answer_data.get('MatchedWikiEntityName'):
            continue
            
        seen_questions.add(norm_q)
        
        topic = determine_topic(question)
        difficulty = determine_difficulty(answer_data.get('Aliases', []), answer_data.get('Value', ''))
        
        candidate = {
            "prompt": question,
            "reference_answer": answer_data.get('Value', ''),
            "answer_aliases": answer_data.get('Aliases', []),
            "difficulty": difficulty,
            "source": "triviaqa",
            "source_id": item['QuestionId'],
            "topic": topic
        }
        candidates.append(candidate)
        
    # Stratified Topic Sampling
    topic_buckets = defaultdict(list)
    for c in candidates:
        topic_buckets[c['topic']].append(c)
        
    final_dataset = []
    topics_chosen = set()
    
    # Shuffle each bucket
    for t in topic_buckets:
        random.shuffle(topic_buckets[t])
        
    # Pick a balanced number from each bucket
    while len(final_dataset) < SAMPLE_SIZE:
        added_in_round = 0
        for topic in list(topic_buckets.keys()):
            if len(final_dataset) >= SAMPLE_SIZE:
                break
            
            # Check constraints: max 15 per bucket
            count_in_final = sum(1 for item in final_dataset if item['topic'] == topic)
            if count_in_final >= 15:
                continue
                
            if topic_buckets[topic]:
                item = topic_buckets[topic].pop(0)
                # Assign task id
                item['task_id'] = f"cat1_pilot_{len(final_dataset)+1:04d}"
                item['category'] = "factual_knowledge"
                
                final_dataset.append(item)
                topics_chosen.add(topic)
                added_in_round += 1
                
        if added_in_round == 0:
            print("Warning: ran out of candidates before reaching sample size!")
            break

    # Gather metrics before removing 'topic'
    topic_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    for item in final_dataset:
        topic_counts[item['topic']] += 1
        difficulty_counts[item['difficulty']] += 1
        # Remove topic from output
        del item['topic']
        
    # Order dictionary to match requested format
    ordered_final = []
    for item in final_dataset:
        ordered_item = {
            "task_id": item["task_id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "reference_answer": item["reference_answer"],
            "answer_aliases": item["answer_aliases"],
            "difficulty": item["difficulty"],
            "source": item["source"],
            "source_id": item["source_id"]
        }
        ordered_final.append(ordered_item)

    # Save JSON
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(ordered_final, f, indent=2)
        
    # Generate README
    readme_content = f"""# Category 1 Pilot Dataset v1\n\nThis pilot dataset was generated from the local TriviaQA dataset to serve as a factual knowledge test.\n\n## Source File\nDataset Variant: **TriviaQA RC (Reading Comprehension) Split**\nFile Used: `qa/wikipedia-dev.json`\n\n## Filter Thresholds\n- **Question Length**: 5 to 40 words.\n- **Evidence**: Must have Wikipedia-sourced evidence (`EntityPages` present).\n- **Primary Answer**: Must have an unambiguous primary answer (`MatchedWikiEntityName` must be present).\n- **Deduplication**: Exact string match after lowercasing and stripping punctuation.\n- **Safety**: Basic blacklist heuristic matching to exclude violence, crime, and offensive words.\n\n## Difficulty Heuristics\n*Note: These heuristics are provisional and intended only for this pilot phase.*\n- **Easy**: Answer has > 10 aliases OR the primary answer length is < 10 characters.\n- **Hard**: Answer has <= 3 aliases AND the primary answer length is > 15 characters.\n- **Medium**: Everything else.\n\n## Dataset Distribution\n\n### Topic Buckets\n- Covering at least 5 distinct buckets. Maximum ~15 entries per bucket.\n"""
    for t, c in topic_counts.items():
        readme_content += f"- **{t}**: {c}\n"

    readme_content += "\n### Difficulty Labels\n"
    for d, c in difficulty_counts.items():
        readme_content += f"- **{d}**: {c}\n"
        
    readme_content += """\n## Known Limitations\n- The topic classification is based on simple keyword heuristics, which can be noisy or miscategorize nuanced questions.\n- The difficulty classification relies on answer string length and alias count, which is a structural proxy rather than semantic difficulty.\n- **Coverage Gap**: TriviaQA predominantly consists of entity-recall questions (Who, What, Where) and lacks deep explanatory factual prompts (How things work, Why things happen). Additional datasets or manual authoring will be needed to cover this gap before Phase 2 scaling.\n"""

    with open(OUTPUT_README_PATH, 'w') as f:
        f.write(readme_content)
        
    print(f"Generated {len(ordered_final)} items.")
    print("Topics:", dict(topic_counts))
    print("Difficulties:", dict(difficulty_counts))

if __name__ == "__main__":
    main()
