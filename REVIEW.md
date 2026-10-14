# Review guide

## Always check

- Cross-boundary identifiers are never assumed: any new code path that produces a role, grant, table, metric, log field, env var, service name, audience, or principal goes through the boundary register (`src/constrain/boundary.py`), not free text.
- Record digests and line numbers are computed by code (`anchor_record`), never accepted from LLM output.
- Boundary verification stays mechanical and fail-closed: register findings are errors, `constrain validate` / `constrain boundary verify` exit non-zero on them. Do not downgrade an error to a warning.
- `boundary-register/1` field names are shared with Pact and the factory; renaming or adding required fields is a breaking change.
- LLM-proposed problem-model updates are validated as typed models; malformed items are rejected, never appended raw.
- New artifacts are registered everywhere artifacts are enumerated: `Session`, `write_artifacts`, `_ARTIFACT_NAMES`, `show`, `export`, MCP server, archive.
- Behavior changes ship with pytest coverage that fails without the change (follow `tests/test_integrated_artifacts.py` / `tests/test_boundary_register.py` style, `_Backend` stubs, no network).

## Style

- Python 3.12+, sync only, Pydantic v2, Click, pathlib for all file I/O.
- Type hints on public functions; module docstrings state the invariant upheld.
- No business logic in `cli.py` beyond orchestration and output; system prompts live in `posture.py`.
- Human-readable error messages, not stack traces.
- Preserve existing line endings (some files are CRLF).

## Skip

- `.kin/`, `.pact/`, `decomposition/`, `contracts/`, generated tests under `tests/src_constrain_*` (unless the change edits them), and root example artifacts (`prompt.md`, `constraints.yaml`, `trust_policy.yaml`, `component_map.yaml`, `schema_hints.yaml`).
