"""Shared test isolation."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_key_config(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> None:
    """Keep the developer's real ~/.config/constrain and key-order env out of tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    monkeypatch.delenv("CONSTRAIN_ANTHROPIC_API_KEY_ENV", raising=False)
