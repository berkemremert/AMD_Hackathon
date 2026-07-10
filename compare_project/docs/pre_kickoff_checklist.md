# Hackathon Checklist

## Access

- lablab.ai enrollment approved.
- AMD AI Developer Program account active.
- AMD Developer Community Discord joined.
- lablab.ai Discord joined.
- Fireworks account created.
- Fireworks API key created and stored.
- `FIREWORKS_API_KEY` tested in a fresh terminal.
- AMD Developer Cloud and Fireworks credit requests submitted.

## Repo

- README includes official setup and usage instructions.
- Docker build succeeds.
- Local sample run succeeds.
- Fireworks smoke test succeeds.
- Decision logs are written to `logs/decisions.jsonl`.
- Official output is written to `/output/results.json`.

## Launch Rules

- Track 1 input path: `/input/tasks.json`.
- Track 1 output path: `/output/results.json`.
- Fireworks calls must use `FIREWORKS_BASE_URL`.
- Model choices must come from `ALLOWED_MODELS`.
- Accuracy gate comes first; passing entries are ranked by recorded token count.
- Docker image must be public and include a linux/amd64 manifest.
- Image compressed size must be under 10GB.
- Submission rate limit: 10 per hour per team.

## First Build Loop

1. Run local JSON contract tests.
2. Run a local Docker smoke test with mounted `/input` and `/output`.
3. Check Fireworks calls use only `ALLOWED_MODELS`.
4. Check decision logs for token counts and model choices.
5. Tune model policy and token caps from failures.
6. Push a public linux/amd64 image only when ready to submit.
