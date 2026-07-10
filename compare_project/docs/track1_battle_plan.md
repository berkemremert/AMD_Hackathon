# Track 1 Battle Plan

## North Star

Win by passing the accuracy gate, then spending fewer recorded Fireworks tokens than the field.

## Current Track Shape

Track 1 is now a general-purpose AI agent across eight categories:

- Factual knowledge
- Mathematical reasoning
- Sentiment classification
- Text summarization
- Named entity recognition
- Code debugging
- Logical / deductive reasoning
- Code generation

The container reads `/input/tasks.json` and writes `/output/results.json`. The harness injects `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`.

## Winning Loop

1. Run deterministic shortcuts only for extremely safe tasks.
2. Route everything else to the cheapest reliable allowed model.
3. Keep prompts category-specific and short.
4. Cap output tokens by category.
5. Tag failures by type.
6. Compare token spend against accuracy.
7. Tune model policy and max-token caps.
8. Repeat in small batches.

## Model Policy

- Code generation/debugging: prefer `kimi-k2p7-code`.
- General, factual, math, logic: prefer Gemma where quality holds.
- Sentiment, summarization, NER: prefer smaller Gemma options first.
- Always choose from runtime `ALLOWED_MODELS`; never hardcode a model as required.

## Error Taxonomy

- Format failures: malformed JSON, too verbose, wrong label, missing field.
- Knowledge failures: factual answer is wrong or vague.
- Reasoning failures: math, code, multi-step logic.
- Ambiguity failures: prompt underspecified or multiple valid answers.
- Overcall failures: Fireworks used when deterministic shortcut would have passed.
- Undercall failures: deterministic shortcut used when Fireworks was needed.

## Submission Story

Bastion is not just a wrapper around an API. It is a cost-aware agent with task classification, model routing, deterministic zero-token shortcuts, Fireworks proxy compliance, and decision logging. Gemma usage is real whenever the allowed-model policy selects Gemma for a task category.
