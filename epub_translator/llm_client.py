"""Unified LLM API client for OpenAI-compatible providers (Kimi) and Anthropic.

Supports a translation model and a cheaper analysis model, shared retry/backoff,
and thread-safe token accounting split between translation and analysis usage.
"""

import json
import os
import random
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple


def extract_json_block(text: str) -> Any:
    """Extract a JSON object/array from an LLM response, tolerating fences and prose."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
    raise ValueError("No JSON object/array found in model response")


def _require_client_libraries():
    """Import the provider SDKs lazily so structural imports never fail."""
    try:
        from anthropic import Anthropic  # noqa: F401
        from openai import OpenAI  # noqa: F401
    except ImportError:
        print("Error: Required dependencies not installed.")
        print("Please run: pip install anthropic openai beautifulsoup4 lxml tqdm")
        raise SystemExit(1)
    return Anthropic, OpenAI


class LLMClient:
    """Provider-agnostic chat client with retry/backoff and token accounting."""

    def __init__(
        self,
        provider: str,
        config: Dict,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        analysis_model: Optional[str] = None,
    ):
        self.provider = provider
        provider_cfg = config.get(provider, {}) or {}

        Anthropic, OpenAI = _require_client_libraries()

        if provider == "kimi":
            self.api_key = api_key or provider_cfg.get("api_key") or os.environ.get("KIMI_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "Kimi API key required. Set it in config.json, KIMI_API_KEY env var, or --api-key."
                )
            base_url = provider_cfg.get("base_url", "https://api.moonshot.cn/v1")
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        elif provider == "anthropic":
            self.api_key = api_key or provider_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key required. Set it in config.json, ANTHROPIC_API_KEY env var, or --api-key."
                )
            self.client = Anthropic(api_key=self.api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'kimi' or 'anthropic'.")

        self.model = model or provider_cfg.get("model", "moonshot-v1-128k")
        self.analysis_model = analysis_model or provider_cfg.get("analysis_model") or self.model
        self.temperature = provider_cfg.get("temperature", 0.3)
        self.max_tokens = provider_cfg.get("max_tokens", 16384)
        self.max_retries = config.get("max_retries", 5)
        self.retry_base_delay = config.get("retry_base_delay", 2)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.analysis_input_tokens = 0
        self.analysis_output_tokens = 0
        self._lock = threading.Lock()

    def chat(
        self,
        system: str,
        user: str,
        analysis: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a system+user request and return the model text response."""
        model = self.analysis_model if analysis else self.model
        temp = self.temperature if temperature is None else temperature
        max_tok = self.max_tokens if max_tokens is None else max_tokens

        last_error: Exception = Exception("Unknown error")
        for attempt in range(self.max_retries):
            try:
                text, in_tokens, out_tokens = self._call(model, system, user, temp, max_tok)
                with self._lock:
                    if analysis:
                        self.analysis_input_tokens += in_tokens
                        self.analysis_output_tokens += out_tokens
                    else:
                        self.total_input_tokens += in_tokens
                        self.total_output_tokens += out_tokens
                return text
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = min(self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1), 60)
                    print(
                        f"  ⚠ {self.provider} API error (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {type(e).__name__}: {e}"
                    )
                    time.sleep(delay)

        raise RuntimeError(f"{self.provider} API failed after {self.max_retries} attempts: {last_error}")

    def _call(self, model: str, system: str, user: str, temperature: float, max_tokens: int) -> Tuple[str, int, int]:
        """Single raw API call. Returns (text, input_tokens, output_tokens)."""
        if self.provider == "anthropic":
            message = self.client.messages.create(
                model=model,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            ).strip()
            return text, message.usage.input_tokens, message.usage.output_tokens

        # OpenAI-compatible (Kimi and similar)
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        return text, response.usage.prompt_tokens, response.usage.completion_tokens
