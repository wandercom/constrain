"""OpenAI-compatible backend (OpenAI, Azure, vLLM, Ollama, etc.)."""

from __future__ import annotations

import os

from . import (
    BackendAuthError,
    BackendConnectionError,
    BackendRateLimitError,
    BackendTimeoutError,
)

DEFAULT_MODEL = "gpt-4o"


class OpenAIBackend:
    """Backend using the OpenAI Python SDK.

    Works with any OpenAI-compatible API by setting base_url
    (e.g., Ollama, vLLM, Azure, Together).
    """

    def __init__(
        self,
        model: str | None = None,
        client=None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenAI backend.\n"
                "Install with: pip install constrain[openai]"
            )
        self._openai = _openai
        self.model = model or DEFAULT_MODEL
        if client:
            self.client = client
        else:
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url or os.environ.get("OPENAI_BASE_URL"):
                kwargs["base_url"] = base_url or os.environ["OPENAI_BASE_URL"]
            self.client = _openai.OpenAI(**kwargs)
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], max_tokens: int | None = None) -> str:
        # Map Anthropic-style (system, messages) to OpenAI format
        oai_messages = [{"role": "system", "content": system}]
        for m in messages:
            oai_messages.append({"role": m["role"], "content": m["content"]})

        try:
            request = dict(
                model=self.model,
                messages=oai_messages,
                max_tokens=max_tokens or self.max_tokens,
            )
            # OpenAI-compatible reasoning models (including Ollama cloud
            # models) may spend the entire completion budget on a private
            # reasoning field and return an empty assistant message.  Keep the
            # provider-specific control opt-in so ordinary OpenAI-compatible
            # servers receive the exact request shape they did before.
            reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip()
            if reasoning_effort:
                request["reasoning_effort"] = reasoning_effort
            resp = self.client.chat.completions.create(**request)
            choice = resp.choices[0]
            if not choice.message.content:
                raise RuntimeError("API returned an empty response")
            return choice.message.content
        except self._openai.RateLimitError as e:
            raise BackendRateLimitError(str(e)) from e
        except self._openai.APITimeoutError as e:
            raise BackendTimeoutError(str(e)) from e
        except self._openai.APIConnectionError as e:
            raise BackendConnectionError(str(e)) from e
        except self._openai.AuthenticationError as e:
            raise BackendAuthError(str(e)) from e
