# Constrain

Interactive constraint elicitation. Interviews users via LLM to extract project boundaries, then synthesizes structured artifacts for the Pact/Arbiter/Baton/Sentinel/Ledger stack.

## Quick Reference

```bash
constrain                          # start new session
constrain --resume <id>            # resume session
constrain --backend openai         # use OpenAI backend
constrain --model <model>          # override model
constrain-mcp                      # start MCP server
python3 -m pytest tests/ -v        # run all tests (309)
```

## Architecture

Three-phase conversation loop:
1. **Understand** — posture: curious, probing. Builds problem model.
2. **Challenge** — posture: adversarial. Tests assumptions, finds edge cases.
3. **Synthesize** — separate conversation. Full history as context. Generates 5 artifacts.

Each response includes structured JSON: `ready_to_proceed: bool`, `problem_model_update: dict`.

## Structure

```
src/constrain/
  cli.py              # Click CLI, entry point
  engine.py           # Conversation loop, Claude API interaction
  posture.py          # Posture definitions, system prompt generation
  synthesizer.py      # Artifact generation and YAML validation
  models.py           # Pydantic data models
  session.py          # Session state, persistence, lifecycle
  mcp_server.py       # MCP server (list/show/search sessions, show artifacts)
  backends/
    __init__.py       # Factory pattern
    anthropic.py      # Anthropic backend (default)
    openai.py         # OpenAI-compatible backend
```

## Output Artifacts

| Artifact | Consumers | Purpose |
|----------|-----------|---------|
| prompt.md | Pact | Induced-understanding briefing |
| constraints.yaml | Pact, Sentinel | Boundary conditions with severity |
| trust_policy.yaml | Arbiter | Trust config, authority map, soak durations |
| component_map.yaml | Pact, Baton | Component topology, edges, dependencies |
| schema_hints.yaml | Ledger | Storage backends, field classifications |

All artifacts cross-validated before writing: component names, authority domains, and edges must match dependencies.

## Conventions

- Python 3.12+, sync-only, Pydantic v2, Click, hatchling
- Type hints on all public functions
- Session persistence via JSON in `.constrain/sessions/`
- Ctrl+C saves session state (resumable)
- Min/max round limits per phase (configurable via CLI and env)
- Backend selection: `CONSTRAIN_BACKEND` env or `--backend` flag
- Model override: `CONSTRAIN_MODEL` env or `--model` flag
- Tests: 309 total (smoke + contract + integration), pytest
- MCP server: `claude mcp add --scope user --transport stdio constrain -- constrain-mcp`

## Kindex

Constrain captures discoveries, decisions, and constraint rationale in [Kindex](~/WanderRepos/repos/kindex). Search before adding. Link related concepts.
