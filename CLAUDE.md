# Constrain

Interactive constraint elicitation. Interviews users via LLM to extract project boundaries, then synthesizes structured artifacts for the Pact/Arbiter/Baton/Sentinel/Ledger stack.

## Quick Reference

```bash
constrain                          # start new session
constrain --resume <id>            # resume session
constrain --backend openai         # use OpenAI backend
constrain --model <model>          # override model
constrain-mcp                      # start MCP server
constrain boundary verify          # verify boundary_register.yaml (CI, no LLM)
python3 -m pytest tests/ -v        # run all tests (530)
```

## Architecture

Three-phase conversation loop:
1. **Understand** — posture: curious, probing. Builds problem model.
2. **Challenge** — posture: adversarial. Tests assumptions, finds edge cases.
3. **Synthesize** — separate conversation. Problem model + verbatim transcript as context. Generates 5 artifacts; code writes the 6th (`boundary_register.yaml`) from the typed register.

Each response includes structured JSON: `ready_to_proceed: bool`, `problem_model_update: dict`.

## Structure

```
src/constrain/
  cli.py              # Click CLI, entry point
  engine.py           # Conversation loop, Claude API interaction
  posture.py          # Posture definitions, system prompt generation
  synthesizer.py      # Artifact generation and YAML validation
  boundary.py         # Boundary register: verification, code-computed records
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
| boundary_register.yaml | Pact | Exact cross-boundary identifiers + records (`boundary-register/1`) |

All artifacts cross-validated before writing: component names, authority domains, and edges must match dependencies.

## Boundary Register

- Never let an identifier crossing a boundary (role, grant, table, metric type, log field, env var, service name, audience, principal...) be assumed. Each side records the exact literal and a record: a file line or a captured observation, pinned by sha256.
- Digests and line numbers are computed by code (`boundary.anchor_record`), never taken from the model. Uncitable values get `records: []` (UNVERIFIED) and an open question.
- `validate_boundary_register` is mechanical (no LLM). Mismatch, near-miss, stale/missing records, placeholders, bad paths, and absence violations block synthesis; UNVERIFIED is written but `constrain validate` exits non-zero.
- Field names of `boundary-register/1` are shared with Pact and the factory: do not rename them.

## Conventions

- Python 3.12+, sync-only, Pydantic v2, Click, hatchling
- Type hints on all public functions
- Session persistence via JSON in `.constrain/sessions/`
- Ctrl+C saves session state (resumable)
- Min/max round limits per phase (configurable via CLI and env)
- Backend selection: `CONSTRAIN_BACKEND` env or `--backend` flag
- Model override: `CONSTRAIN_MODEL` env or `--model` flag
- Tests: 530 total (smoke + contract + integration), pytest
- MCP server: `claude mcp add --scope user --transport stdio constrain -- constrain-mcp`

## Kindex

Constrain captures discoveries, decisions, and constraint rationale in [Kindex](https://github.com/wandercom/kindex). Search before adding. Link related concepts.
