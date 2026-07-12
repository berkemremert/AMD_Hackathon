"""Container entrypoint for the Track 1 submission."""
import sys

from src.track1_agent.pipeline import run


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"agent failed: {exc}", file=sys.stderr)
        sys.exit(1)
