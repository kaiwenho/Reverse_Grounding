"""
Ollama adapter for gpt-oss.

Implements the `LLMClient` protocol (a `complete(system, user) -> str` method)
against a locally running Ollama server. Requires `pip install ollama` and
`ollama pull gpt-oss:120b` (or whichever model you serve).
"""

from __future__ import annotations

from typing import Optional

import ollama


class OllamaGPTOSSClient:
    def __init__(
        self,
        model: str = "gpt-oss:120b",
        temperature: float = 0.0,
        host: Optional[str] = None,
        num_ctx: int = 32768,
    ):
        """
        :param model: Ollama model tag.
        :param temperature: 0.0 for reproducible planning.
        :param host: Ollama server URL. None = default (http://localhost:11434).
        :param num_ctx: Context window size. gpt-oss:120b supports large contexts;
                        the full-schema prompt is ~17K tokens, standard is ~6K.
                        Bump if you add many exemplars.
        """
        self.model = model
        self.temperature = temperature
        self._client = ollama.Client(host=host) if host else ollama.Client()
        self.num_ctx = num_ctx

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
            format="json",
        )
        return response["message"]["content"]
