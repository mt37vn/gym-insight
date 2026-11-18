import os
import json
import time
from typing import Optional
from openai import OpenAI

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class DeepseekClient:
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        self._client: Optional[OpenAI] = None
        self.conversations: dict[str, list[dict]] = {}
        self.total_tokens = 0
        self.total_cached_tokens = 0
        self.total_cost = 0.0

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        use_cache: bool = True,
        conversation_id: str = None,
    ) -> dict:
        if not self.is_configured():
            return {"error": "API key not configured", "result": None}

        if conversation_id:
            conv_key = conversation_id
        else:
            conv_key = "_default"

        if conv_key not in self.conversations:
            self.conversations[conv_key] = []

        full_messages = list(messages)
        if use_cache and len(full_messages) > 1:
            full_messages[0]["content"] = full_messages[0]["content"]

        kwargs = {
            "model": model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if use_cache and model == "deepseek-chat":
            kwargs["extra_body"] = {"ephemeral_prompt_caching": 1}

        try:
            start = time.time()
            response = self.client.chat.completions.create(**kwargs)
            elapsed = time.time() - start

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            cached_tokens = 0

            if usage and hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

            self.total_tokens += prompt_tokens + completion_tokens
            self.total_cached_tokens += cached_tokens

            cost_per_1m_input = 0.5 if model != "deepseek-reasoner" else 2.0
            cost_per_1m_output = 2.0 if model != "deepseek-reasoner" else 8.0
            request_cost = (prompt_tokens / 1_000_000 * cost_per_1m_input +
                           completion_tokens / 1_000_000 * cost_per_1m_output)
            self.total_cost += request_cost

            result = {
                "content": response.choices[0].message.content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "cost": request_cost,
                "elapsed": elapsed,
                "model": model,
            }

            if conversation_id:
                self.conversations[conv_key].append({
                    "role": "assistant",
                    "content": result["content"],
                })

            return {"result": result, "error": None}

        except Exception as e:
            return {"result": None, "error": str(e)}

    def get_cached_context(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost": self.total_cost,
            "cache_ratio": self.total_cached_tokens / max(self.total_tokens, 1),
        }

    def reset_stats(self):
        self.total_tokens = 0
        self.total_cached_tokens = 0
        self.total_cost = 0.0
