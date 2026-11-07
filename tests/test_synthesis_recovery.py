"""Incomplete generation must preserve work without publishing a successful result."""
from types import SimpleNamespace as NS

import pytest
from click.testing import CliRunner

from constrain.backends import Completion
from constrain.backends.anthropic import AnthropicBackend
from constrain.backends.openai import OpenAIBackend
from constrain.cli import cli, _project_topic
from constrain.engine import ConversationEngine
from constrain.models import Session, Phase, Posture
from constrain.session import SessionManager

ARTIFACTS = "--- PROMPT ---\n# A project\n--- CONSTRAINTS ---\nconstraints: []\n"


class Backend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, system, messages):
        self.calls.append([dict(message) for message in messages])
        return next(self.responses)


@pytest.fixture
def cli_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CONSTRAIN_KINDEX', 'off')
    monkeypatch.setattr('constrain.cli.ensure_api_key', lambda: None)
    manager = SessionManager(tmp_path)
    session = Session(posture=Posture.adversarial, phase=Phase.synthesize, round=1)
    manager.save(session)
    # Tests exercise provider output, not an optional third-party input translator.
    monkeypatch.setattr(ConversationEngine, '_call_api', lambda self, system, messages: self.backend.complete(system, messages))
    return manager, session


@pytest.mark.parametrize('response', ['--- PROMPT ---\npartial', '--- PROMPT ---\nhello\n--- CONSTRAINTS ---\n'])
def test_failed_generation_is_resumable_and_does_not_replace_artifacts(cli_session, tmp_path, monkeypatch, response):
    manager, session = cli_session
    old = tmp_path / 'prompt.md'
    old.write_text('previous accepted work')
    backend = Backend([response])
    monkeypatch.setattr('constrain.cli.create_backend', lambda **kwargs: backend)
    failed = CliRunner().invoke(cli, ['resume'])
    assert failed.exit_code != 0
    recovered = manager.find_latest_incomplete()
    assert recovered.id == session.id
    assert recovered.phase == Phase.synthesize
    assert recovered.synthesis_attempts[-1]['text'] == response
    assert recovered.synthesis_error
    assert old.read_text() == 'previous accepted work'
    assert not (tmp_path / 'constraints.yaml').exists()

    backend = Backend([ARTIFACTS])
    monkeypatch.setattr('constrain.cli.create_backend', lambda **kwargs: backend)
    resumed = CliRunner().invoke(cli, ['resume'], input='\n')
    assert resumed.exit_code == 0, resumed.output
    assert manager.find_latest_incomplete() is None
    assert (tmp_path / 'constraints.yaml').read_text().strip() == 'constraints: []'
    assert manager.load(session.id).synthesis_error == ''


def test_token_truncation_continues_without_restarting_or_losing_evidence(cli_session, tmp_path, monkeypatch):
    manager, session = cli_session
    cut = ARTIFACTS.index('project') + 3  # even a cut inside a word must preserve text
    backend = Backend([Completion(ARTIFACTS[:cut], 'max_tokens', 100), Completion(ARTIFACTS[cut:], 'end_turn', 20)])
    monkeypatch.setattr('constrain.cli.create_backend', lambda **kwargs: backend)
    result = CliRunner().invoke(cli, ['resume'], input='\n')
    assert result.exit_code == 0, result.output
    saved = manager.load(session.id)
    assert saved.phase == Phase.complete
    assert saved.prompt_md == '# A project'
    assert [x['stop_reason'] for x in saved.synthesis_attempts] == ['max_tokens', 'end_turn']
    assert [x['output_tokens'] for x in saved.synthesis_attempts] == [100, 20]
    assert backend.calls[1][-2]['content'] == ARTIFACTS[:cut]


def test_continuation_is_bounded_even_if_delimiters_are_present(cli_session, tmp_path, monkeypatch):
    manager, session = cli_session
    backend = Backend([Completion(ARTIFACTS, 'max_tokens', 100)] * 3)
    monkeypatch.setattr('constrain.cli.create_backend', lambda **kwargs: backend)
    result = CliRunner().invoke(cli, ['resume'])
    assert result.exit_code != 0
    assert len(backend.calls) == 3
    assert manager.find_latest_incomplete().id == session.id
    assert len(manager.load(session.id).synthesis_attempts) == 3
    assert not (tmp_path / 'prompt.md').exists()


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
def test_provider_termination_metadata_survives_adapter(provider):
    if provider == 'anthropic':
        response = NS(content=[NS(type='text', text='unfinished')], stop_reason='max_tokens', usage=NS(output_tokens=99))
        backend = AnthropicBackend(client=NS(messages=NS(create=lambda **kwargs: response)))
    else:
        response = NS(choices=[NS(message=NS(content='unfinished'), finish_reason='length')], usage=NS(completion_tokens=99))
        backend = OpenAIBackend(client=NS(chat=NS(completions=NS(create=lambda **kwargs: response))))
    result = backend.complete('system', [])
    assert result == 'unfinished'
    assert result.truncated
    assert result.output_tokens == 99


def test_retracted_boundary_no_longer_blocks_synthesis(cli_session, monkeypatch):
    manager, session = cli_session
    session.problem_model.apply_update({'boundary_register': [{
        'id': 'retired', 'kind': 'name', 'relation': 'equal', 'description': 'obsolete contract',
        'sides': [{'party': 'producer', 'role': 'defines', 'value': 'old', 'records': []},
                  {'party': 'consumer', 'role': 'consumes', 'value': 'different', 'records': []}]
    }], 'assumptions': ['old assumption', 'still true']})
    backend = Backend([ARTIFACTS])
    engine = ConversationEngine(session, manager, backend=backend)
    from constrain.boundary import blocking_findings, validate_boundary_register, load_boundary_register_text, render_boundary_register
    register = load_boundary_register_text(render_boundary_register(session.problem_model.boundary_register))
    assert blocking_findings(validate_boundary_register(register, manager.base_path))
    session.problem_model.apply_update({'remove': {'boundary_register': ['retired'], 'assumptions': ['old assumption']}})
    manager.save(session)
    loaded = manager.load(session.id)
    assert loaded.problem_model.assumptions == ['still true']
    assert not loaded.problem_model.boundary_register
    assert engine._boundary_gate()


def test_topic_uses_project_not_run_folder(tmp_path, monkeypatch):
    monkeypatch.delenv('CONSTRAIN_PROJECT', raising=False)
    project = tmp_path / 'kinbase-design'
    run = project / 'constrain'
    run.mkdir(parents=True)
    assert _project_topic(run) == project.name
    (project / '.git').mkdir()
    nested = project / 'work' / 'sessions'
    nested.mkdir(parents=True)
    assert _project_topic(nested) == project.name
    (project / '.kin').mkdir()
    (project / '.kin' / 'config').write_text('name: Kinbase Design\n')
    assert _project_topic(run) == 'Kinbase Design'
    monkeypatch.setenv('CONSTRAIN_PROJECT', 'Explicit project')
    assert _project_topic(run) == 'Explicit project'


@pytest.mark.parametrize('bad_constraints', ['[unterminated', '# nothing here', 'not structured constraints'])
def test_unrepairable_constraints_are_not_published(cli_session, tmp_path, monkeypatch, bad_constraints):
    manager, session = cli_session
    malformed = ARTIFACTS.replace('constraints: []', bad_constraints)
    backend = Backend([malformed, malformed])
    monkeypatch.setattr('constrain.cli.create_backend', lambda **kwargs: backend)
    result = CliRunner().invoke(cli, ['resume'])
    assert result.exit_code != 0
    assert manager.find_latest_incomplete().id == session.id
    assert not (tmp_path / 'constraints.yaml').exists()
    assert len(backend.calls) == 2


def test_failed_revision_retains_feedback_for_resume(cli_session, tmp_path, monkeypatch):
    manager, session = cli_session
    backend = Backend([ARTIFACTS, 'unfinished revision'])
    monkeypatch.setattr('constrain.cli.create_backend', lambda **kwargs: backend)
    result = CliRunner().invoke(cli, ['resume'], input='Use the revised ownership model\n')
    assert result.exit_code != 0
    saved = manager.find_latest_incomplete()
    assert saved.id == session.id
    assert saved.conversation[-1].content == 'Use the revised ownership model'
    assert not (tmp_path / 'prompt.md').exists()
