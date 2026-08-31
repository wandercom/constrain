"""Anthropic (Claude) backend."""

from __future__ import annotations

import os

from . import (
    BackendAuthError,
    BackendConnectionError,
    BackendRateLimitError,
    BackendTimeoutError,
)

DEFAULT_MODEL = "claude-opus-5"

_API_KEY_ENV_VARS = (
    "WANDER_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
    "JMC_ANTHROPIC_API_KEY",
)


def _anthropic_api_key() -> str | None:
    """Resolve the Anthropic billing key, preferring the Wander account."""
    for name in _API_KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


class AnthropicBackend:
    """Backend using the Anthropic Python SDK."""

    def __init__(self, model: str | None = None, client=None, max_tokens: int = 4096) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic backend.\n"
                "Install with: pip install constrain[anthropic]"
            )
        self._anthropic = _anthropic
        self.model = model or DEFAULT_MODEL
        if client is not None:
            self.client = client
        else:
            api_key = _anthropic_api_key()
            self.client = (
                _anthropic.Anthropic(api_key=api_key)
                if api_key
                else _anthropic.Anthropic()
            )
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], max_tokens: int | None = None) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=system,
                messages=messages,
            )
            text = "".join(
                block.text
                for block in resp.content
                if getattr(block, "type", "text") == "text"
                and isinstance(getattr(block, "text", None), str)
            )
            if not text:
                raise RuntimeError("API returned no text content")
            return text
        except self._anthropic.RateLimitError as e:
            raise BackendRateLimitError(str(e)) from e
        except self._anthropic.APITimeoutError as e:
            raise BackendTimeoutError(str(e)) from e
        except self._anthropic.APIConnectionError as e:
            raise BackendConnectionError(str(e)) from e
        except self._anthropic.AuthenticationError as e:
            raise BackendAuthError(str(e)) from e
