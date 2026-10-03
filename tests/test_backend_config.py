from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from constrain.backends import create_backend
from constrain.backends.anthropic import AnthropicBackend, DEFAULT_MODEL
from constrain.backends.local_agent import LocalAgentBackend
from constrain.backends.openai import OpenAIBackend


def test_create_backend_uses_env_max_tokens_for_local_agent(monkeypatch):
    monkeypatch.setenv("CONSTRAIN_CODEX_COMMAND", "python3")
    monkeypatch.setenv("CONSTRAIN_MAX_TOKENS", "32000")

    backend = create_backend("codex")

    assert backend.max_tokens == 32000


def test_create_backend_rejects_invalid_env_max_tokens(monkeypatch):
    monkeypatch.setenv("CONSTRAIN_MAX_TOKENS", "nope")

    with pytest.raises(ValueError, match="CONSTRAIN_MAX_TOKENS"):
        create_backend("codex")


def test_local_agent_backend_runs_configured_command():
    backend = LocalAgentBackend(
        name="testagent",
        command="python3",
        args_template=["-c", "print('agent output')"],
        max_tokens=16000,
    )

    assert backend.complete("system", [{"role": "user", "content": "hello"}]) == "agent output"


def test_anthropic_backend_defaults_to_standard_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")
    monkeypatch.setenv("ORG_ANTHROPIC_API_KEY", "org-key")

    with patch("anthropic.Anthropic") as client_cls:
        backend = AnthropicBackend()

    client_cls.assert_called_once_with(api_key="generic-key")
    assert backend.model == DEFAULT_MODEL == "claude-opus-5"


def test_anthropic_backend_honors_key_order_env(monkeypatch):
    monkeypatch.setenv("CONSTRAIN_ANTHROPIC_API_KEY_ENV", "ORG_ANTHROPIC_API_KEY, ANTHROPIC_API_KEY")
    monkeypatch.setenv("ORG_ANTHROPIC_API_KEY", "org-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")

    with patch("anthropic.Anthropic") as client_cls:
        AnthropicBackend()

    client_cls.assert_called_once_with(api_key="org-key")


def test_anthropic_backend_honors_key_order_config_file(monkeypatch, tmp_path):
    cfg = tmp_path / "constrain" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('anthropic_api_key_env = ["ORG_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"]\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ORG_ANTHROPIC_API_KEY", "org-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")

    with patch("anthropic.Anthropic") as client_cls:
        AnthropicBackend()

    client_cls.assert_called_once_with(api_key="org-key")


def test_anthropic_backend_does_not_fall_back_to_excluded_standard_key(monkeypatch):
    from constrain.backends import BackendAuthError

    monkeypatch.setenv("CONSTRAIN_ANTHROPIC_API_KEY_ENV", "ORG_ANTHROPIC_API_KEY")
    monkeypatch.delenv("ORG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "excluded-key")

    with patch("anthropic.Anthropic") as client_cls:
        with pytest.raises(BackendAuthError, match="ORG_ANTHROPIC_API_KEY"):
            AnthropicBackend()

    client_cls.assert_not_called()


@pytest.mark.parametrize("override", [",", " , "])
def test_empty_key_order_override_is_rejected(monkeypatch, override):
    from constrain.keyconfig import KeyConfigError, anthropic_api_key

    monkeypatch.setenv("CONSTRAIN_ANTHROPIC_API_KEY_ENV", override)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")

    with pytest.raises(KeyConfigError, match="names no environment variables"):
        anthropic_api_key()


def test_empty_key_order_config_is_rejected(monkeypatch, tmp_path):
    from constrain.keyconfig import KeyConfigError, anthropic_api_key

    cfg = tmp_path / "constrain" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("anthropic_api_key_env = []\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")

    with pytest.raises(KeyConfigError, match="names no environment variables"):
        anthropic_api_key()


def test_malformed_key_config_fails_loudly(monkeypatch, tmp_path):
    from constrain.keyconfig import KeyConfigError, anthropic_api_key

    cfg = tmp_path / "constrain" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("anthropic_api_key_env = 3\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    with pytest.raises(KeyConfigError, match="anthropic_api_key_env"):
        anthropic_api_key()


def test_anthropic_backend_skips_non_text_blocks():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="private reasoning"),
            SimpleNamespace(type="text", text="answer"),
        ]
    )
    client = Mock()
    client.messages.create.return_value = response

    backend = AnthropicBackend(client=client)

    assert backend.complete("system", [{"role": "user", "content": "hello"}]) == "answer"


def test_openai_backend_forwards_opt_in_reasoning_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "none")
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )

    backend = OpenAIBackend(client=client, model="qwen3.5:cloud")

    assert backend.complete("system", [{"role": "user", "content": "hello"}]) == "answer"
    client.chat.completions.create.assert_called_once_with(
        model="qwen3.5:cloud",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
        max_tokens=4096,
        reasoning_effort="none",
    )
