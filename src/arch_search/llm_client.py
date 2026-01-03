from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Dict, Any


class ChatCompletionClient:
    """Minimal HTTP client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM-driven search.")
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 600) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"LLM request failed: HTTP {e.code}, {detail}") from e
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

        parsed = json.loads(body)
        try:
            return parsed["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response: {parsed}") from e
