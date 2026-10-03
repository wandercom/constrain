"""Which environment variables hold the Anthropic API key, in preference order.

Default is the vendor-standard ``ANTHROPIC_API_KEY``. Operators who keep
several Anthropic keys (e.g. one per billing account) configure the ordered
list of NAMES (never values), either with ``CONSTRAIN_ANTHROPIC_API_KEY_ENV``
(comma-separated) or in ``$XDG_CONFIG_HOME/constrain/config.toml``
(default ``~/.config/constrain/config.toml``)::

    anthropic_api_key_env = ["MY_ORG_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"]

The env var wins over the file. See ``config.example.toml``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULT_ANTHROPIC_API_KEY_ENV_VARS: tuple[str, ...] = ("ANTHROPIC_API_KEY",)
ANTHROPIC_API_KEY_ENV_OVERRIDE = "CONSTRAIN_ANTHROPIC_API_KEY_ENV"
CONFIG_KEY = "anthropic_api_key_env"


class KeyConfigError(RuntimeError):
    """The user's key-order config exists but cannot be used."""


def config_path() -> Path:
    """Location of the optional user config file (XDG, outside any repo)."""
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".config"
    return root / "constrain" / "config.toml"


def anthropic_api_key_env_vars() -> tuple[str, ...]:
    """Ordered env-var names to try for the Anthropic key.

    A malformed config raises instead of silently falling back: billing the
    wrong account quietly is worse than failing loudly.
    """
    raw = os.environ.get(ANTHROPIC_API_KEY_ENV_OVERRIDE, "").strip()
    if raw:
        names = tuple(n.strip() for n in raw.split(",") if n.strip())
        if names:
            return names
    path = config_path()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return DEFAULT_ANTHROPIC_API_KEY_ENV_VARS
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise KeyConfigError(f"Could not read Constrain config {path}: {exc}") from exc
    names = data.get(CONFIG_KEY)
    if names is None:
        return DEFAULT_ANTHROPIC_API_KEY_ENV_VARS
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",")]
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise KeyConfigError(
            f"{path}: '{CONFIG_KEY}' must be a list of environment variable names"
        )
    cleaned = tuple(n.strip() for n in names if n.strip())
    return cleaned or DEFAULT_ANTHROPIC_API_KEY_ENV_VARS


def anthropic_api_key() -> str | None:
    """Resolve the Anthropic key from the configured env-var names, in order."""
    for name in anthropic_api_key_env_vars():
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None
