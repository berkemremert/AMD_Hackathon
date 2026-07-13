# Track 1: Hybrid Token-Efficient Routing Agent

This repository contains our submission for the **AMD Developer Hackathon: ACT II (Track 1)**. 

Our agent is a highly optimized hybrid router designed to absolutely minimize Fireworks API token usage. It achieves this through zero-token local deterministic solvers for math/logic, aggressive token starvation (disabling reasoning tokens & stripping tags), and a strict cross-tier fallback mechanism.

## Prerequisites

- Docker installed
- A valid Fireworks AI API Key

## Setup & Usage Instructions

This project is fully containerized. To run the agent and see the routing logic in action, follow these steps:

### 1. Build the Docker Image

Run the following command in the root directory to build the Docker image:

```bash
docker build -t track1-agent .
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add your Fireworks API key:

```env
FIREWORKS_API_KEY=your_api_key_here
```

*(Optional: You can also specify the allowed models by adding `ALLOWED_MODELS` to your `.env` file.)*

### 3. Run the Container

Run the container using the `.env` file you just created:

```bash
mkdir -p output
docker run --env-file .env \
  --mount type=bind,src="$PWD/input",dst=/input,readonly \
  --mount type=bind,src="$PWD/output",dst=/output \
  track1-agent
```

The container executes `agent.py`, reads `input/tasks.json`, and writes `output/results.json`.

The original hackathon requirements are preserved in [`docs/submission-guide.txt`](docs/submission-guide.txt).
