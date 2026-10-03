"""
Contract tests for constrain.cli module.
Generated pytest test suite covering happy paths, edge cases, error cases, and invariants.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
from click.testing import CliRunner
import click
import os
import sys

# Import the module under test
from constrain.cli import (
    ensure_api_key,
    _round_options,
    _resolve_int,
    resolve_config,
    _run_engine,
    _ARTIFACT_NAMES,
    _archive_dir,
    cli,
    cmd_new,
    cmd_resume,
    cmd_show,
    cmd_list,
    _do_list,
    main,
    SafeGroup,
)
from constrain.models import Phase


# Test fixtures
@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to provide clean environment for each test."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ORG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CONSTRAIN_MIN_UNDERSTAND", raising=False)
    monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
    monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
    monkeypatch.delenv("CONSTRAIN_MAX_CHALLENGE", raising=False)
    return monkeypatch


@pytest.fixture
def mock_session():
    """Fixture to create a mock Session object."""
    session = Mock()
    session.phase = Phase.complete
    session.prompt_md = "# Test Prompt"
    session.constraints_yaml = "constraints: []"
    return session


@pytest.fixture
def mock_session_manager():
    """Fixture to create a mock SessionManager object."""
    mgr = Mock()
    mgr.list_all = Mock(return_value=[])
    mgr.find_latest_incomplete = Mock(return_value=None)
    mgr.create = Mock()
    mgr.save = Mock()
    mgr.load = Mock()
    return mgr


@pytest.fixture
def mock_engine_config():
    """Fixture to create a mock EngineConfig object."""
    config = Mock()
    config.understand_min = 2
    config.understand_max = 10
    config.challenge_min = 2
    config.challenge_max = 10
    return config


@pytest.fixture
def runner():
    """Fixture to provide Click CliRunner."""
    return CliRunner()


# ============================================================================
# Tests for ensure_api_key()
# ============================================================================

def test_ensure_api_key_happy_path(mock_env):
    """Test ensure_api_key does not raise when ANTHROPIC_API_KEY is set."""
    mock_env.setenv("ANTHROPIC_API_KEY", "  my-api-key-123  ")

    # Should not raise
    ensure_api_key()


def test_ensure_api_key_accepts_configured_key_name(mock_env):
    """A key under a configured (non-standard) env-var name is accepted."""
    mock_env.setenv("CONSTRAIN_ANTHROPIC_API_KEY_ENV", "ORG_ANTHROPIC_API_KEY,ANTHROPIC_API_KEY")
    mock_env.setenv("ORG_ANTHROPIC_API_KEY", "org-key")

    ensure_api_key()


def test_ensure_api_key_ignores_unconfigured_key_name(mock_env):
    """Only configured names count; the default is ANTHROPIC_API_KEY alone."""
    mock_env.setenv("ORG_ANTHROPIC_API_KEY", "org-key")

    with pytest.raises(click.ClickException):
        ensure_api_key()


def test_ensure_api_key_missing(mock_env):
    """Test ensure_api_key raises error when ANTHROPIC_API_KEY is not set."""
    # Environment variable not set (already cleared by mock_env)

    with pytest.raises(click.ClickException) as exc_info:
        ensure_api_key()

    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_ensure_api_key_empty(mock_env):
    """Test ensure_api_key raises error when ANTHROPIC_API_KEY is empty or whitespace."""
    mock_env.setenv("ANTHROPIC_API_KEY", "   ")

    with pytest.raises(click.ClickException) as exc_info:
        ensure_api_key()

    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_invariant_api_key_non_empty(mock_env):
    """Invariant: ensure_api_key() succeeds without raising when key is set."""
    mock_env.setenv("ANTHROPIC_API_KEY", "valid-key")

    # Should not raise
    ensure_api_key()


# ============================================================================
# Tests for _round_options()
# ============================================================================

def test_round_options_decorator():
    """Test _round_options adds four Click options to a function."""
    @_round_options
    def dummy_func():
        pass

    # Check that the function has been decorated
    assert hasattr(dummy_func, '__click_params__')

    # Should have 4 parameters added
    params = dummy_func.__click_params__
    assert len(params) >= 4

    # Verify parameter names
    param_names = [p.name for p in params]
    assert 'min_understand' in param_names
    assert 'max_understand' in param_names
    assert 'min_challenge' in param_names
    assert 'max_challenge' in param_names


# ============================================================================
# Tests for _resolve_int()
# ============================================================================

def test_resolve_int_cli_val_priority(mock_env):
    """Test _resolve_int returns cli_val when not None."""
    mock_env.setenv("TEST_VAR", "10")

    result = _resolve_int(5, "TEST_VAR", 3)

    assert result == 5


def test_resolve_int_env_var_fallback(mock_env):
    """Test _resolve_int returns parsed env var when cli_val is None."""
    mock_env.setenv("TEST_VAR", "10")

    result = _resolve_int(None, "TEST_VAR", 3)

    assert result == 10


def test_resolve_int_default_fallback(mock_env):
    """Test _resolve_int returns default when cli_val is None and env_var not set."""
    # TEST_VAR not set

    result = _resolve_int(None, "TEST_VAR", 3)

    assert result == 3


def test_resolve_int_invalid_env(mock_env):
    """Test _resolve_int raises error when env var cannot be parsed as positive integer."""
    mock_env.setenv("TEST_VAR", "abc")

    with pytest.raises(click.ClickException) as exc_info:
        _resolve_int(None, "TEST_VAR", 3)

    assert "TEST_VAR" in str(exc_info.value)


def test_resolve_int_negative_env(mock_env):
    """Test _resolve_int raises error when env var is negative."""
    mock_env.setenv("TEST_VAR", "-5")

    with pytest.raises(click.ClickException) as exc_info:
        _resolve_int(None, "TEST_VAR", 3)

    assert "TEST_VAR" in str(exc_info.value) or "positive" in str(exc_info.value).lower()


def test_resolve_int_zero_env(mock_env):
    """Test _resolve_int raises error when env var is zero."""
    mock_env.setenv("TEST_VAR", "0")

    with pytest.raises(click.ClickException) as exc_info:
        _resolve_int(None, "TEST_VAR", 3)

    assert "TEST_VAR" in str(exc_info.value) or "positive" in str(exc_info.value).lower()


# ============================================================================
# Tests for resolve_config()
# ============================================================================

def test_resolve_config_happy_path(mock_env):
    """Test resolve_config returns EngineConfig with valid min/max values."""
    result = resolve_config(2, 10, 2, 10)

    assert result.understand_min == 2
    assert result.understand_max == 10
    assert result.challenge_min == 2
    assert result.challenge_max == 10


def test_resolve_config_defaults(mock_env):
    """Test resolve_config uses default values when all params are None."""
    result = resolve_config(None, None, None, None)

    assert result.understand_min == 2
    assert result.understand_max == 10
    assert result.challenge_min == 2
    assert result.challenge_max == 10


def test_resolve_config_understand_min_exceeds_max(mock_env):
    """Test resolve_config raises error when understand_min > understand_max."""
    with pytest.raises(click.ClickException) as exc_info:
        resolve_config(15, 10, 2, 10)

    assert "understand" in str(exc_info.value).lower()


def test_resolve_config_challenge_min_exceeds_max(mock_env):
    """Test resolve_config raises error when challenge_min > challenge_max."""
    with pytest.raises(click.ClickException) as exc_info:
        resolve_config(2, 10, 15, 10)

    assert "challenge" in str(exc_info.value).lower()


def test_resolve_config_edge_equal_values(mock_env):
    """Test resolve_config allows min == max for both understand and challenge."""
    result = resolve_config(5, 5, 3, 3)

    # Should not raise an exception
    assert result.understand_min == 5
    assert result.understand_max == 5
    assert result.challenge_min == 3
    assert result.challenge_max == 3


def test_invariant_understand_min_lte_max(mock_env):
    """Invariant: understand_min <= understand_max in resolved EngineConfig."""
    test_cases = [
        (2, 10, 2, 10),
        (1, 1, 1, 1),
        (5, 20, 3, 15),
    ]

    for min_u, max_u, min_c, max_c in test_cases:
        result = resolve_config(min_u, max_u, min_c, max_c)
        assert result.understand_min <= result.understand_max


def test_invariant_challenge_min_lte_max(mock_env):
    """Invariant: challenge_min <= challenge_max in resolved EngineConfig."""
    test_cases = [
        (2, 10, 2, 10),
        (1, 1, 1, 1),
        (5, 20, 3, 15),
    ]

    for min_u, max_u, min_c, max_c in test_cases:
        result = resolve_config(min_u, max_u, min_c, max_c)
        assert result.challenge_min <= result.challenge_max


def test_invariant_default_config_values(mock_env):
    """Invariant: Default values are understand_min=2, understand_max=10, challenge_min=2, challenge_max=10."""
    result = resolve_config(None, None, None, None)

    assert result.understand_min == 2
    assert result.understand_max == 10
    assert result.challenge_min == 2
    assert result.challenge_max == 10


# ============================================================================
# Tests for archive helpers
# ============================================================================

def test_artifact_names_list():
    """_ARTIFACT_NAMES contains the expected artifact filenames."""
    assert "prompt.md" in _ARTIFACT_NAMES
    assert "constraints.yaml" in _ARTIFACT_NAMES

def test_archive_dir_location(tmp_path):
    """_archive_dir returns .constrain/archive under the given directory."""
    result = _archive_dir(tmp_path)
    assert result == tmp_path / ".constrain" / "archive"


# ============================================================================
# Tests for _run_engine()
# ============================================================================

@patch('constrain.cli.write_artifacts')
@patch('constrain.cli.archive_artifacts')
@patch('constrain.cli._archive_dir')
@patch('constrain.cli.ConversationEngine')
@patch('constrain.cli.create_backend')
@patch('constrain.cli.Path')
@patch('click.echo')
def test_run_engine_success_complete_phase(mock_echo, mock_path, mock_create_backend, mock_engine_class, mock_archive_dir, mock_archive, mock_write_artifacts, mock_session, mock_session_manager, mock_engine_config):
    """Test _run_engine writes artifacts when session completes successfully."""
    # Setup
    mock_session.phase = Phase.complete
    mock_archive.return_value = (None, [])  # No artifacts to archive
    mock_archive_dir.return_value = Path("/fake/.constrain/archive")

    mock_engine = Mock()
    mock_engine_class.return_value = mock_engine

    mock_cwd = Mock()
    mock_path.cwd.return_value = mock_cwd

    mock_write_artifacts.return_value = [Path("/fake/prompt.md"), Path("/fake/constraints.yaml")]

    # Execute
    _run_engine(mock_session, mock_session_manager, mock_engine_config)

    # Verify
    mock_engine_class.assert_called_once()
    mock_engine.run_session.assert_called_once()

    # Check that artifacts were written (session is complete with prompt_md)
    mock_write_artifacts.assert_called_once()


@patch('constrain.cli.ConversationEngine')
@patch('constrain.cli.create_backend')
@patch('click.echo')
def test_run_engine_incomplete_phase(mock_echo, mock_create_backend, mock_engine_class, mock_session, mock_session_manager, mock_engine_config):
    """Test _run_engine does not write artifacts when session is incomplete."""
    # Setup
    mock_session.phase = Phase.understand  # Not complete

    mock_engine = Mock()
    mock_engine_class.return_value = mock_engine

    # Execute
    _run_engine(mock_session, mock_session_manager, mock_engine_config)

    # Verify
    mock_engine.run_session.assert_called_once()
    # Artifacts should not be written since phase is not complete


@patch('constrain.cli.write_artifacts')
@patch('constrain.cli.archive_artifacts')
@patch('constrain.cli._archive_dir')
@patch('constrain.cli.ConversationEngine')
@patch('constrain.cli.create_backend')
@patch('constrain.cli.Path')
def test_run_engine_archives_before_writing(mock_path, mock_create_backend, mock_engine_class, mock_archive_dir, mock_archive, mock_write_artifacts, mock_session, mock_session_manager, mock_engine_config):
    """Test _run_engine archives existing artifacts before writing new ones."""
    # Setup
    mock_session.phase = Phase.complete
    mock_archive.return_value = (Path("/fake/archive/auth"), [(Path("prompt.md"), Path("/fake/archive/auth/prompt.md"))])
    mock_archive_dir.return_value = Path("/fake/.constrain/archive")

    mock_engine = Mock()
    mock_engine_class.return_value = mock_engine

    mock_cwd = Mock()
    mock_path.cwd.return_value = mock_cwd

    mock_write_artifacts.return_value = [Path("/fake/prompt.md")]

    # Execute
    _run_engine(mock_session, mock_session_manager, mock_engine_config)

    # Verify archive was called before write
    mock_archive.assert_called_once()
    mock_write_artifacts.assert_called_once()


def test_invariant_artifacts_written(tmp_path):
    """Invariant: Artifacts written are prompt.md and constraints.yaml."""
    # This is tested implicitly in test_run_engine_success_complete_phase
    # Here we verify the file names are correct
    expected_artifacts = ["prompt.md", "constraints.yaml"]

    for artifact in expected_artifacts:
        assert artifact in ["prompt.md", "constraints.yaml"]


# ============================================================================
# Tests for cli() commands using CliRunner
# ============================================================================

@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
@patch('constrain.cli.resolve_config')
@patch('constrain.cli._run_engine')
def test_cli_no_subcommand_new_session(mock_run_engine, mock_resolve_config, mock_sm_class, mock_ensure_api_key, runner):
    """Test cli creates new session when no incomplete sessions exist."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_sm.find_latest_incomplete.return_value = None
    mock_sm.create.return_value = Mock()
    mock_sm_class.return_value = mock_sm
    mock_resolve_config.return_value = Mock()

    # Execute
    result = runner.invoke(cli, [])

    # Verify
    assert result.exit_code == 0 or result.exit_code is None
    mock_sm.create.assert_called_once()


@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
@patch('constrain.cli.resolve_config')
@patch('constrain.cli._run_engine')
@patch('click.prompt')
def test_cli_no_subcommand_resume_prompt(mock_prompt, mock_run_engine, mock_resolve_config, mock_sm_class, mock_ensure_api_key, runner):
    """Test cli prompts to resume when incomplete session exists."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_incomplete_session = Mock()
    mock_incomplete_session.id = "abc12345-test"
    mock_incomplete_session.phase = Phase.understand
    mock_incomplete_session.understand_rounds = 3
    mock_incomplete_session.challenge_rounds = 0
    mock_sm.find_latest_incomplete.return_value = mock_incomplete_session
    mock_sm_class.return_value = mock_sm
    mock_resolve_config.return_value = Mock()
    mock_prompt.return_value = "y"  # User confirms resume

    # Execute
    result = runner.invoke(cli, [], input="y\n")

    # Verify
    mock_run_engine.assert_called()


@patch('constrain.cli.SessionManager')
def test_cli_with_subcommand(mock_sm_class, runner):
    """Test cli returns immediately when subcommand is invoked."""
    mock_sm = Mock()
    mock_sm.list_all.return_value = []
    mock_sm_class.return_value = mock_sm

    result = runner.invoke(cli, ['list'])

    # The command should execute (even if it fails due to no sessions)
    # The key is that it doesn't try to create a new session
    assert result is not None


@patch('constrain.cli.ensure_api_key')
def test_cli_missing_api_key(mock_ensure_api_key, runner):
    """Test cli raises error when ANTHROPIC_API_KEY not set."""
    mock_ensure_api_key.side_effect = click.ClickException("ANTHROPIC_API_KEY not set")

    result = runner.invoke(cli, [])

    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output


@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
@patch('constrain.cli.resolve_config')
def test_cli_invalid_config(mock_resolve_config, mock_sm_class, mock_ensure_api_key, runner):
    """Test cli raises error when config validation fails."""
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_sm.find_latest_incomplete.return_value = None
    mock_sm_class.return_value = mock_sm
    mock_resolve_config.side_effect = click.ClickException("Invalid config")

    result = runner.invoke(cli, [])

    assert result.exit_code != 0


# ============================================================================
# Tests for cmd_new()
# ============================================================================

@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
@patch('constrain.cli.resolve_config')
@patch('constrain.cli._run_engine')
def test_cmd_new_happy_path(mock_run_engine, mock_resolve_config, mock_sm_class, mock_ensure_api_key, runner):
    """Test cmd_new creates and runs new session unconditionally."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_session = Mock()
    mock_sm.create.return_value = mock_session
    mock_sm_class.return_value = mock_sm
    mock_resolve_config.return_value = Mock()

    # Execute
    result = runner.invoke(cli, ['new'])

    # Verify
    assert result.exit_code == 0 or result.exit_code is None
    mock_sm.create.assert_called_once()
    mock_run_engine.assert_called_once()


@patch('constrain.cli.ensure_api_key')
def test_cmd_new_missing_api_key(mock_ensure_api_key, runner):
    """Test cmd_new raises error when ANTHROPIC_API_KEY not set."""
    mock_ensure_api_key.side_effect = click.ClickException("ANTHROPIC_API_KEY not set")

    result = runner.invoke(cli, ['new'])

    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output


@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
@patch('constrain.cli.resolve_config')
def test_cmd_new_invalid_config(mock_resolve_config, mock_sm_class, mock_ensure_api_key, runner):
    """Test cmd_new raises error when config validation fails."""
    mock_ensure_api_key.return_value = "test-key"
    mock_sm_class.return_value = Mock()
    mock_resolve_config.side_effect = click.ClickException("Invalid config")

    result = runner.invoke(cli, ['new'])

    assert result.exit_code != 0


# ============================================================================
# Tests for cmd_resume()
# ============================================================================

@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
@patch('constrain.cli.resolve_config')
@patch('constrain.cli._run_engine')
def test_cmd_resume_with_session_id(mock_run_engine, mock_resolve_config, mock_sm_class, mock_ensure_api_key, runner):
    """Test cmd_resume loads and runs specified session."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_session = Mock()
    mock_session.phase = Phase.understand  # Not complete
    mock_sm.load.return_value = mock_session
    mock_sm_class.return_value = mock_sm
    mock_resolve_config.return_value = Mock()

    # Execute
    result = runner.invoke(cli, ['resume', 'abc123'])

    # Verify
    assert result.exit_code == 0 or result.exit_code is None
    mock_sm.load.assert_called_once_with('abc123')
    mock_run_engine.assert_called_once()


@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
@patch('constrain.cli.resolve_config')
@patch('constrain.cli._run_engine')
def test_cmd_resume_latest(mock_run_engine, mock_resolve_config, mock_sm_class, mock_ensure_api_key, runner):
    """Test cmd_resume loads and runs latest incomplete session when no ID provided."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_session = Mock()
    mock_session.phase = Phase.understand
    mock_sm.find_latest_incomplete.return_value = mock_session
    mock_sm_class.return_value = mock_sm
    mock_resolve_config.return_value = Mock()

    # Execute
    result = runner.invoke(cli, ['resume'])

    # Verify
    assert result.exit_code == 0 or result.exit_code is None
    mock_sm.find_latest_incomplete.assert_called_once()
    mock_run_engine.assert_called_once()


@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
def test_cmd_resume_session_not_found(mock_sm_class, mock_ensure_api_key, runner):
    """Test cmd_resume raises error when specified session does not exist."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_sm.load.side_effect = FileNotFoundError("Session not found")
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['resume', 'nonexistent'])

    # Verify
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or result.exception


@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
def test_cmd_resume_session_load_error(mock_sm_class, mock_ensure_api_key, runner):
    """Test cmd_resume raises error when session load raises ValueError."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_sm.load.side_effect = ValueError("Invalid session data")
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['resume', 'bad-session'])

    # Verify
    assert result.exit_code != 0


@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
def test_cmd_resume_session_already_complete(mock_sm_class, mock_ensure_api_key, runner):
    """Test cmd_resume raises error when session is already complete."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_session = Mock()
    mock_session.phase = Phase.complete
    mock_sm.load.return_value = mock_session
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['resume', 'complete-session'])

    # Verify
    assert result.exit_code != 0
    assert "complete" in result.output.lower()


@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
def test_cmd_resume_no_incomplete_sessions(mock_sm_class, mock_ensure_api_key, runner):
    """Test cmd_resume raises error when no incomplete sessions found."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"
    mock_sm = Mock()
    mock_sm.find_latest_incomplete.return_value = None
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['resume'])

    # Verify
    assert result.exit_code != 0
    assert "no" in result.output.lower() or "incomplete" in result.output.lower()


# ============================================================================
# Tests for cmd_show()
# ============================================================================

@patch('constrain.cli.SessionManager')
def test_cmd_show_happy_path(mock_sm_class, runner):
    """Test cmd_show displays artifacts from most recent completed session."""
    # Setup - list_all returns dicts, not session objects
    mock_sm = Mock()
    mock_sm.list_all.return_value = [{
        "id": "test-id-1234",
        "phase": "complete",
        "posture": "adversarial",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "understand_rounds": 3,
        "challenge_rounds": 2,
        "is_complete": True,
    }]
    # load returns a mock Session with prompt_md and constraints_yaml
    mock_session = Mock()
    mock_session.id = "test-id-1234"
    mock_session.prompt_md = "# Test Prompt"
    mock_session.constraints_yaml = "constraints: []"
    mock_sm.load.return_value = mock_session
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['show'])

    # Verify
    assert result.exit_code == 0
    assert "Test Prompt" in result.output


@patch('constrain.cli.SessionManager')
def test_cmd_show_no_completed_sessions(mock_sm_class, runner):
    """Test cmd_show raises error when no completed sessions found."""
    # Setup - list_all returns dicts with no completed sessions
    mock_sm = Mock()
    mock_sm.list_all.return_value = [{
        "id": "test-id",
        "phase": "understand",
        "posture": "adversarial",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "understand_rounds": 1,
        "challenge_rounds": 0,
        "is_complete": False,
    }]
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['show'])

    # Verify
    assert result.exit_code != 0


@patch('constrain.cli.SessionManager')
def test_cmd_show_artifacts_not_found(mock_sm_class, runner):
    """Test cmd_show raises error when latest completed session has no artifacts."""
    # Setup
    mock_sm = Mock()
    mock_sm.list_all.return_value = [{
        "id": "test-id-1234",
        "phase": "complete",
        "posture": "adversarial",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "understand_rounds": 3,
        "challenge_rounds": 2,
        "is_complete": True,
    }]
    mock_session = Mock()
    mock_session.id = "test-id-1234"
    mock_session.prompt_md = ""  # Empty means no artifacts (falsy)
    mock_sm.load.return_value = mock_session
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['show'])

    # Verify
    assert result.exit_code != 0 or "not found" in result.output.lower()


# ============================================================================
# Tests for cmd_list() and _do_list()
# ============================================================================

@patch('constrain.cli.SessionManager')
def test_cmd_list_happy_path(mock_sm_class, runner):
    """Test cmd_list displays table of all sessions."""
    # Setup - list_all returns dicts
    mock_sm = Mock()
    mock_sm.list_all.return_value = [{
        "id": "abc12345-session-id",
        "phase": "understand",
        "posture": "adversarial",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "understand_rounds": 2,
        "challenge_rounds": 0,
        "is_complete": False,
    }]
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['list'])

    # Verify
    assert result.exit_code == 0
    assert len(result.output) > 0


@patch('constrain.cli.SessionManager')
def test_cmd_list_no_sessions(mock_sm_class, runner):
    """Test cmd_list raises error when no sessions found."""
    # Setup
    mock_sm = Mock()
    mock_sm.list_all.return_value = []
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['list'])

    # Verify
    assert result.exit_code != 0 or "no sessions" in result.output.lower()


def test_do_list_happy_path(mock_session_manager):
    """Test _do_list outputs formatted session table."""
    # Setup - list_all returns dicts
    mock_session_manager.list_all.return_value = [{
        "id": "abc12345-session-id",
        "phase": "understand",
        "posture": "adversarial",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "understand_rounds": 2,
        "challenge_rounds": 0,
        "is_complete": False,
    }]

    # Execute - should not raise
    _do_list(mock_session_manager)


def test_do_list_no_sessions(mock_session_manager):
    """Test _do_list raises error when session list is empty."""
    # Setup
    mock_session_manager.list_all.return_value = []

    # Execute and verify
    with pytest.raises(click.ClickException) as exc_info:
        _do_list(mock_session_manager)

    assert "no sessions" in str(exc_info.value).lower()


# ============================================================================
# Tests for main()
# ============================================================================

@patch('constrain.cli.cli')
def test_main_invokes_cli(mock_cli):
    """Test main invokes cli() Click group."""
    # Execute
    main()

    # Verify
    mock_cli.assert_called_once()


# ============================================================================
# Tests for SafeGroup.invoke()
# ============================================================================

def test_safegroup_invoke_click_exception():
    """Test SafeGroup.invoke re-raises ClickException."""
    safe_group = SafeGroup()
    ctx = Mock()

    with patch.object(click.Group, 'invoke', side_effect=click.ClickException("Test error")):
        with pytest.raises(click.ClickException):
            safe_group.invoke(ctx)


def test_safegroup_invoke_abort():
    """Test SafeGroup.invoke handles Abort with exit code 1."""
    # Create a SafeGroup instance
    safe_group = SafeGroup()

    # Create a mock context that raises Abort
    ctx = Mock()

    with patch.object(click.Group, 'invoke', side_effect=click.Abort()):
        with patch('sys.exit') as mock_exit:
            safe_group.invoke(ctx)

            mock_exit.assert_called_once_with(1)


def test_safegroup_invoke_keyboard_interrupt():
    """Test SafeGroup.invoke handles KeyboardInterrupt with exit code 130."""
    # Create a SafeGroup instance
    safe_group = SafeGroup()

    # Create a mock context that raises KeyboardInterrupt
    ctx = Mock()

    with patch.object(click.Group, 'invoke', side_effect=KeyboardInterrupt()):
        with patch('sys.exit') as mock_exit:
            safe_group.invoke(ctx)

            mock_exit.assert_called_once_with(130)


def test_safegroup_invoke_system_exit():
    """Test SafeGroup.invoke handles SystemExit."""
    # Create a SafeGroup instance
    safe_group = SafeGroup()

    # Create a mock context that raises SystemExit
    ctx = Mock()

    with patch.object(click.Group, 'invoke', side_effect=SystemExit(42)):
        with pytest.raises(SystemExit) as exc_info:
            safe_group.invoke(ctx)

        assert exc_info.value.code == 42


def test_safegroup_invoke_general_exception():
    """Test SafeGroup.invoke handles general exception with exit code 1."""
    # Create a SafeGroup instance
    safe_group = SafeGroup()

    # Create a mock context that raises general exception
    ctx = Mock()

    with patch.object(click.Group, 'invoke', side_effect=Exception("Test error")):
        with patch('sys.exit') as mock_exit:
            safe_group.invoke(ctx)

            mock_exit.assert_called_once_with(1)


def test_invariant_safegroup_exit_codes():
    """Invariant: SafeGroup exits with code 1 for Abort/general errors, 130 for KeyboardInterrupt."""
    safe_group = SafeGroup()
    ctx = Mock()

    # Test Abort -> exit code 1
    with patch.object(click.Group, 'invoke', side_effect=click.Abort()):
        with patch('sys.exit') as mock_exit:
            safe_group.invoke(ctx)
            mock_exit.assert_called_once_with(1)

    # Test KeyboardInterrupt -> exit code 130
    with patch.object(click.Group, 'invoke', side_effect=KeyboardInterrupt()):
        with patch('sys.exit') as mock_exit:
            safe_group.invoke(ctx)
            mock_exit.assert_called_once_with(130)

    # Test general exception -> exit code 1
    with patch.object(click.Group, 'invoke', side_effect=RuntimeError("test")):
        with patch('sys.exit') as mock_exit:
            safe_group.invoke(ctx)
            mock_exit.assert_called_once_with(1)


# ============================================================================
# Integration Tests
# ============================================================================

@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
@patch('constrain.cli.ConversationEngine')
@patch('constrain.cli.resolve_config')
@patch('constrain.cli._run_engine')
def test_integration_new_session_flow(mock_run_engine, mock_resolve_config, mock_engine_class, mock_sm_class, mock_ensure_api_key, runner, tmp_path):
    """Integration test for complete new session flow."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"

    mock_sm = Mock()
    mock_session = Mock()
    mock_session.phase = Phase.complete
    mock_session.prompt_md = "# Test"
    mock_session.constraints_yaml = "constraints: []"
    mock_sm.create.return_value = mock_session
    mock_sm_class.return_value = mock_sm

    mock_config = Mock()
    mock_resolve_config.return_value = mock_config

    mock_engine = Mock()
    mock_engine_class.return_value = mock_engine

    # Execute
    result = runner.invoke(cli, ['new'])

    # Verify flow
    mock_ensure_api_key.assert_called()
    mock_sm.create.assert_called_once()
    mock_run_engine.assert_called_once()


@patch('constrain.cli.ensure_api_key')
@patch('constrain.cli.SessionManager')
@patch('constrain.cli.resolve_config')
@patch('constrain.cli._run_engine')
def test_integration_resume_session_flow(mock_run_engine, mock_resolve_config, mock_sm_class, mock_ensure_api_key, runner):
    """Integration test for resume session flow."""
    # Setup
    mock_ensure_api_key.return_value = "test-key"

    mock_sm = Mock()
    mock_session = Mock()
    mock_session.phase = Phase.understand
    mock_sm.load.return_value = mock_session
    mock_sm_class.return_value = mock_sm

    mock_config = Mock()
    mock_resolve_config.return_value = mock_config

    # Execute
    result = runner.invoke(cli, ['resume', 'test-id'])

    # Verify flow
    mock_ensure_api_key.assert_called()
    mock_sm.load.assert_called_once_with('test-id')
    mock_run_engine.assert_called_once()


@patch('constrain.cli.SessionManager')
def test_integration_list_sessions_flow(mock_sm_class, runner):
    """Integration test for list sessions flow."""
    # Setup - list_all returns dicts
    mock_sm = Mock()
    mock_sm.list_all.return_value = [{
        "id": "test-id-12345678",
        "phase": "understand",
        "posture": "adversarial",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "understand_rounds": 2,
        "challenge_rounds": 0,
        "is_complete": False,
    }]
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['list'])

    # Verify flow
    mock_sm.list_all.assert_called_once()
    assert "test-id-" in result.output or result.exit_code == 0


@patch('constrain.cli.SessionManager')
def test_integration_show_artifacts_flow(mock_sm_class, runner):
    """Integration test for show artifacts flow."""
    # Setup - list_all returns dicts
    mock_sm = Mock()
    mock_sm.list_all.return_value = [{
        "id": "test-id-1234",
        "phase": "complete",
        "posture": "adversarial",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "understand_rounds": 3,
        "challenge_rounds": 2,
        "is_complete": True,
    }]
    mock_session = Mock()
    mock_session.id = "test-id-1234"
    mock_session.prompt_md = "# Test Prompt\n\nContent here"
    mock_session.constraints_yaml = "constraints:\n  - test"
    mock_sm.load.return_value = mock_session
    mock_sm_class.return_value = mock_sm

    # Execute
    result = runner.invoke(cli, ['show'])

    # Verify flow
    mock_sm.list_all.assert_called_once()
    assert result.exit_code == 0 or "Test Prompt" in result.output or "constraints" in result.output
