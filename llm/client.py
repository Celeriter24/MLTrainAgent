"""
LLM Client — abstracts Ollama, LM Studio, llama.cpp, vLLM
All expose an OpenAI-compatible /v1/chat/completions endpoint.
"""

import json
import logging
import time
from typing import Generator
import requests

logger = logging.getLogger(__name__)

# HTTP-Statuscodes die einen Retry rechtfertigen (Server-seitige Fehler)
_RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMClient:
    def __init__(self, config: dict):
        self.cfg = config["llm"]
        self.backend = self.cfg["backend"]
        self.model = self.cfg["model"]
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.temperature = self.cfg.get("temperature", 0.2)
        self.max_tokens = self.cfg.get("max_tokens", 8192)
        self.timeout = self.cfg.get("timeout", 120)
        self.retries = self.cfg.get("retries", 3)

    def _post_with_retry(self, url: str, **kwargs) -> requests.Response:
        """
        POST mit automatischem Retry bei Netzwerkfehlern und 5xx-Antworten.
        Exponentielles Backoff: 1s → 2s → 4s → ...
        4xx-Fehler werden sofort weitergereicht (kein Retry).
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(url, **kwargs)
                if resp.status_code in _RETRY_STATUS:
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code}", response=resp
                    )
                return resp
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.HTTPError) as exc:
                last_exc = exc
                if attempt < self.retries:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s, ...
                    logger.warning(
                        f"LLM-Anfrage fehlgeschlagen (Versuch {attempt}/{self.retries}): "
                        f"{exc} — retry in {wait}s"
                    )
                    time.sleep(wait)
        raise last_exc

    def _endpoint(self) -> str:
        if self.backend == "ollama":
            return f"{self.base_url}/api/chat"
        # LM Studio, vLLM, llama.cpp all use OpenAI-compatible path
        return f"{self.base_url}/v1/chat/completions"

    def chat(self, messages: list[dict], stream: bool = False) -> str:
        """Send messages and return full response text."""
        if self.backend == "ollama":
            return self._ollama_chat(messages, stream)
        return self._openai_chat(messages, stream)

    def stream_chat(self, messages: list[dict]) -> Generator[str, None, None]:
        """Stream tokens from the LLM."""
        if self.backend == "ollama":
            yield from self._ollama_stream(messages)
        else:
            yield from self._openai_stream(messages)

    # ── Ollama ──────────────────────────────────────────────────────────────

    def _ollama_chat(self, messages: list[dict], stream: bool) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        resp = self._post_with_retry(
            self._endpoint(), json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _ollama_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        with requests.post(
            self._endpoint(), json=payload, stream=True, timeout=self.timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

    # ── OpenAI-compatible (LM Studio, vLLM, llama.cpp) ──────────────────────

    def _openai_chat(self, messages: list[dict], stream: bool) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        resp = self._post_with_retry(
            self._endpoint(), json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _openai_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        with requests.post(
            self._endpoint(), json=payload, stream=True, timeout=self.timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    raw = line[6:]
                    if raw.strip() == b"[DONE]":
                        break
                    data = json.loads(raw)
                    delta = data["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token

    def health_check(self) -> bool:
        try:
            if self.backend == "ollama":
                resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            else:
                resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False
