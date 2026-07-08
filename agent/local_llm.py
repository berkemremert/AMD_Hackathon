"""Ollama-backed local LLM client.

Wraps the Ollama HTTP API (``/api/generate``) to provide a simple
``generate(prompt) -> str`` interface used by the debugging pipeline.
"""

import requests


class LocalLLM:
    """Synchronous client for the Ollama local inference server.

    Args:
        model: Ollama model tag (e.g. ``"qwen2.5-coder:3b"``).
        base_url: Ollama server URL (default ``http://localhost:11434``).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 0.9,
        num_ctx: int = 4096,
        num_predict: int = 900,
    ) -> str:
        """Send a prompt to Ollama and return the generated text.

        Args:
            prompt: The full prompt string.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Nucleus sampling threshold.
            num_ctx: Context window size in tokens.
            num_predict: Maximum tokens to generate.

        Returns:
            The model's response text, stripped of leading/trailing whitespace.
        """
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
