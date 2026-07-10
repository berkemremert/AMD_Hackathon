"""OpenAI-compatible mock of the Fireworks proxy for offline contract tests.
Returns category-plausible canned answers and realistic usage numbers."""
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def single(system: str) -> str:
    if "arithmetic expression" in system:
        return "0.85*240+18"
    if "Sentiment" in system:
        return "Positive. The reviewer is clearly satisfied."
    if "entities" in system:
        return "Tim Cook — person\nApple — organization"
    if "code" in system.lower():
        return "```python\ndef add(a, b):\n    return a + b\n```"
    if "Answer: <" in system:
        return "Reasoning here.\nAnswer: 42"
    return "A concise generic answer."


# Numbered items in the user message => batched call: answer each line.
BATCH_ITEM = re.compile(r"^\s*(\d+)\.\s", re.M)


def canned(system: str, user: str) -> str:
    if "numbered items" in system:
        nums = BATCH_ITEM.findall(user)
        if nums:
            per = "Positive" if "Sentiment" in system else "A concise answer"
            return "\n".join(f"{n}: {per}" for n in nums)
    return single(system)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        sys_msg = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
        user = next((m["content"] for m in body["messages"] if m["role"] == "user"), "")
        if not sys_msg:
            # MERGE_SYSTEM mode folds the instruction into the user message;
            # a real endpoint reads it regardless of role, so match on it too.
            sys_msg = user
        content = canned(sys_msg, user)
        prompt_toks = max(1, (len(sys_msg) + len(user)) // 4)
        completion_toks = max(1, len(content) // 4)
        resp = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": prompt_toks + completion_toks,
            },
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    # MOCK_BIND=0.0.0.0 exposes the mock to Docker containers via
    # host.docker.internal for in-container contract tests.
    ThreadingHTTPServer((os.environ.get("MOCK_BIND", "127.0.0.1"), 18080), Handler).serve_forever()
