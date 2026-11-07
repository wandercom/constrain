"""Backend abstraction for LLM providers."""

from __future__ import annotations

import os
from typing import Protocol


class Completion(str):
    """Text-compatible response retaining provider termination evidence."""

    def __new__(cls, text: str, stop_reason: str | None = None, output_tokens: int | None = None):
        obj = super().__new__(cls, text)
        obj.stop_reason = stop_reason
        obj.output_tokens = output_tokens
        return obj

    @property
    def truncated(self) -> bool:
        return self.stop_reason in {"max_tokens", "length"}


class BackendError(Exception):
    """Base class for backend errors."""


class BackendRateLimitError(BackendError):
    """Rate limit exceeded — retry after delay."""


class BackendTimeoutError(BackendError):
    """Request timed out."""


class BackendConnectionError(BackendError):
    """Could not connect to the backend."""


class BackendAuthError(BackendError):
    """Authentication failed — do not retry."""


class Backend(Protocol):
    """Protocol for LLM backends. Implementations must provide complete()."""

    def complete(self, system: str, messages: list[dict], max_tokens: int | None = None) -> str:
        """Send a completion request. Returns the assistant's text content."""
        ...


def create_backend(
    backend: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> Backend:
    """Factory: create a backend from env vars or explicit args.

    Reads CONSTRAIN_BACKEND (default: "anthropic"), CONSTRAIN_MODEL, and
    CONSTRAIN_MAX_TOKENS.
    """
    backend = backend or os.environ.get("CONSTRAIN_BACKEND", "anthropic")
    model = model or os.environ.get("CONSTRAIN_MODEL")
    max_tokens = max_tokens if max_tokens is not None else _max_tokens_from_env()
    max_tokens = max_tokens or 4096

    if backend == "anthropic":
        from .anthropic import AnthropicBackend

        return AnthropicBackend(model=model, max_tokens=max_tokens, **kwargs)
    elif backend == "openai":
        from .openai import OpenAIBackend

        return OpenAIBackend(model=model, max_tokens=max_tokens, **kwargs)
    elif backend == "codex":
        from .codex import CodexBackend

        return CodexBackend(model=model, max_tokens=max_tokens, **kwargs)
    elif backend == "claude":
        from .claude import ClaudeBackend

        return ClaudeBackend(model=model, max_tokens=max_tokens, **kwargs)
    elif backend == "opencode":
        from .opencode import OpenCodeBackend

        return OpenCodeBackend(model=model, max_tokens=max_tokens, **kwargs)
    elif backend == "cursor":
        from .cursor import CursorBackend

        return CursorBackend(model=model, max_tokens=max_tokens, **kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported: anthropic, openai, "
            "codex, claude, opencode, cursor. "
            "Set CONSTRAIN_BACKEND to one of these."
        )


def _max_tokens_from_env() -> int | None:
    raw = os.environ.get("CONSTRAIN_MAX_TOKENS")
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid CONSTRAIN_MAX_TOKENS: expected a positive integer, got {raw!r}."
        ) from exc
    if value < 1:
        raise ValueError(
            f"Invalid CONSTRAIN_MAX_TOKENS: expected a positive integer, got {raw!r}."
        )
    return value
