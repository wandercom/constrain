# Standard Operating Procedures

## Language & Runtime
- Python 3.12+
- No async — synchronous conversation loop is appropriate for interactive CLI
- Type hints on all public functions

## Dependencies
- `anthropic` — LLM calls
- `click` — CLI framework
- `pydantic` — Data models and serialization
- `pyyaml` — Constraint output format
- Standard library only beyond these four

## Project Structure
```
src/constrain/
  __init__.py          # Version
  cli.py               # Click commands
  session.py           # Session state, persistence, lifecycle
  engine.py            # Conversation loop, Claude API interaction
  posture.py           # Posture definitions, system prompt generation
  synthesizer.py       # Artifact generation (prompt.md, constraints.yaml)
  models.py            # Pydantic models (Session, Constraint, ProblemModel, etc.)
```

## Code Standards
- Each module has a single responsibility
- Models are Pydantic BaseModel subclasses with JSON serialization
- CLI commands delegate to engine/session — no business logic in cli.py
- System prompts are string constants or templates in posture.py, not external files
- Error messages are human-readable, not stack traces
- All file I/O uses pathlib

## LLM Interaction Pattern
- Use `anthropic.Anthropic()` client (sync)
- Model: `claude-opus-5` for all phases (resolved from `DEFAULT_MODEL` in
  `backends/anthropic.py`; `claude-sonnet-4-20250514` is retired and must not be reintroduced)
- System prompt changes per phase; conversation history is cumulative
- Each LLM response in understand/challenge phases should end with a structured JSON block (```json ... ```) containing:
  - `ready_to_proceed: bool` — whether the phase is complete
  - `problem_model_update: dict` — incremental updates to the problem model
- Parse this JSON from the response; display only the natural language portion to the engineer
- For synthesis, use a separate conversation with the full history as context

## Testing
- pytest
- Unit tests for models, posture prompt generation, session persistence, constraint serialization
- Integration test: mock Anthropic client, verify full session flow produces valid artifacts

## Output
- `prompt.md` written to current working directory
- `constraints.yaml` written to current working directory
- Session state written to `.constrain/sessions/<id>.json`
- Overwrite existing output files with confirmation prompt
