# Constrain

Find the boundary between specification and intent.

Constrain is a CLI tool that interviews engineers about their problem and produces five artifacts that feed directly into downstream tools: **Pact** (contract-first development), **Arbiter** (trust arbitration), **Baton** (circuit orchestration), **Sentinel** (production monitoring), and **Ledger** (schema scaffolding).

## Install

Install the released package directly from GitHub Releases:

```bash
python3 -m pip install "constrain[anthropic] @ https://github.com/wandercom/constrain/releases/download/v0.5.2/constrain-0.5.2-py3-none-any.whl"
```

For a source checkout:

```bash
git clone https://github.com/wandercom/constrain.git
cd constrain
python3 -m pip install -e ".[all]"
```

> **Important:** the project named `constrain` on PyPI is an unrelated PNNL
> package. Do not use a bare `pip install constrain` for this project.

Requires Python 3.12+ and credentials or a logged-in local agent for your
chosen backend. Replace the `anthropic` extra in the release command with
`openai`, `mcp`, or `all` as needed.

## Usage

```bash
constrain              # Start a new session (or resume incomplete)
constrain new          # Always start fresh
constrain resume       # Resume the most recent incomplete session
constrain show         # Display artifacts from the last completed session
constrain list         # List all sessions
constrain validate     # Validate all artifacts for internal consistency
constrain export -f <format>   # Export skeletons for downstream tools
constrain mcp-server   # Run MCP server (stdio transport)
```

### Export Formats

```bash
constrain export --format pact     # task.md skeleton from prompt.md
constrain export --format baton    # baton.yaml skeleton from component_map.yaml
constrain export --format arbiter  # arbiter.yaml skeleton from trust_policy.yaml
constrain export --format ledger   # schema_hints.yaml skeleton for Ledger
```

### Backend Selection

```bash
# Anthropic (default)
export WANDER_ANTHROPIC_API_KEY=sk-...
constrain

# OpenAI
constrain --backend openai --model gpt-4o

# Local model via OpenAI-compatible API
CONSTRAIN_BACKEND=openai OPENAI_BASE_URL=http://localhost:11434/v1 constrain --model llama3

# Local agent CLIs
constrain --backend codex --model gpt-5.3-codex --max-tokens 32000
constrain --backend claude --model sonnet --max-tokens 16000
constrain --backend opencode
constrain --backend cursor

# Environment variables
CONSTRAIN_BACKEND=anthropic|openai|codex|claude|opencode|cursor
CONSTRAIN_MODEL=<model-name>
CONSTRAIN_MAX_TOKENS=4096
```

Backend credential and command settings:

| Backend | Required/config env vars |
|---------|--------------------------|
| `anthropic` | `WANDER_ANTHROPIC_API_KEY` (preferred), `ANTHROPIC_API_KEY`, or `JMC_ANTHROPIC_API_KEY`; optional `CONSTRAIN_MODEL`, `CONSTRAIN_MAX_TOKENS` |
| `openai` | `OPENAI_API_KEY` or `OPENAI_BASE_URL`, optional `CONSTRAIN_MODEL`, `CONSTRAIN_MAX_TOKENS` |
| `codex` | logged-in `codex` CLI, optional `CONSTRAIN_CODEX_COMMAND`, `CONSTRAIN_CODEX_MODEL`, `CONSTRAIN_CODEX_ARGS` |
| `claude` | logged-in `claude` CLI or its own auth env, optional `CONSTRAIN_CLAUDE_COMMAND`, `CONSTRAIN_CLAUDE_MODEL`, `CONSTRAIN_CLAUDE_ARGS` |
| `opencode` | installed `opencode` CLI, optional `CONSTRAIN_OPENCODE_COMMAND`, `CONSTRAIN_OPENCODE_MODEL`, `CONSTRAIN_OPENCODE_ARGS` |
| `cursor` | installed `cursor-agent` CLI, optional `CONSTRAIN_CURSOR_COMMAND`, `CONSTRAIN_CURSOR_MODEL`, `CONSTRAIN_CURSOR_ARGS` |

`*_ARGS` is shell-split and may use `{prompt}`, `{system}`, `{model}`, and `{max_tokens}` placeholders.

## How It Works

Run `constrain` from your project directory and have a conversation. The tool builds a model of your problem, challenges your description through a randomized adversarial lens, then synthesizes five output files.

### Phase 1: Understand

Collaborative interview that builds a problem model through structured questioning: system context, stakeholders, failure modes, dependencies, assumptions, boundaries. 2-10 rounds until the model is complete.

### Phase 2: Challenge

One of five analytical postures is randomly selected at session start (hidden from you):

| Posture | Lens |
|---------|------|
| **Adversarial** | Worst valid interpretation of every ambiguity |
| **Contrarian** | Challenges the problem itself -- XY problems, wrong abstractions |
| **Critic** | Where two senior engineers would disagree about correctness |
| **Skeptic** | Tests that pass for wrong reasons, gameable metrics |
| **Collaborator** | Same gaps, constructive framing |

The challenge phase also probes for conflict-resolution gaps (data ownership, authority disputes) and storage obligations (databases, sensitive fields, erasure requirements, audit retention, encryption).

### Phase 3: Synthesize

Generates five artifacts from the accumulated conversation, with one revision cycle. All YAML artifacts are validated before writing to disk. Cross-validation checks consistency across artifacts (component names, authority domains, edge/dependency alignment, trust floor ordering, annotation requirements).

## Output Artifacts

**`prompt.md`** -- Induced-understanding prompt. Reads like a briefing to a senior engineer:
- System context, consequence map, failure archaeology
- Dependency landscape, boundary conditions, success shape
- Trust and authority model (natural language summary)
- Component topology (natural language summary)

**`constraints.yaml`** -- Structured constraint set. Each constraint is a black-box boundary condition:

```yaml
constraints:
  - id: C001
    boundary: "what this constrains"
    condition: "the invariant that must hold"
    violation: "what failure looks like"
    severity: must | should | may
    rationale: "why this matters"
    classification_tier: PII | FINANCIAL | AUTH | COMPLIANCE | PUBLIC | null
    affected_components: ["component-name"]
```

**`trust_policy.yaml`** -- Trust and authority model. Consumed by Arbiter:
- Trust configuration (floor, authority override, decay, taint lock tiers)
- Data classification registry with canary eligibility
- Soak durations by tier
- Authority map (which components own which data domains)
- Human gate triggers

**`component_map.yaml`** -- Component topology. Consumed by Pact and Baton:
- Component definitions (name, type, port, protocol, data access, authority, dependencies)
- Edge definitions (from, to, protocol, description)

**`schema_hints.yaml`** -- Storage schema hints. Consumed by Ledger:
- Storage backends (owner component, type, description)
- Field hints (classification, annotations like `gdpr_erasable`, `audit_field`, `encrypted_at_rest`, `immutable`)

## Downstream Integration

| Artifact | Consumed by | Purpose |
|----------|------------|---------|
| `prompt.md` | Pact | System briefing for decomposition agent |
| `constraints.yaml` | Pact, Sentinel | Boundary conditions for contracts and violation detection |
| `trust_policy.yaml` | Arbiter | Classification, soak config, authority, gates |
| `component_map.yaml` | Pact, Baton | Topology for decomposition and circuit orchestration |
| `schema_hints.yaml` | Ledger | Schema scaffolding from interview-derived storage hints |

## Session Persistence

Sessions save to `.constrain/sessions/` in the current directory. Ctrl+C saves and exits (resumable). Ctrl+D ends the current phase early.

## MCP Server

Constrain exposes sessions and artifacts to AI assistants via MCP:

```bash
# Standalone
constrain-mcp

# Via CLI
constrain mcp-server --project-dir /path/to/project

# Register with Claude Code
claude mcp add --scope user --transport stdio constrain -- constrain-mcp
```

**Tools**: `constrain_list_sessions`, `constrain_show_session`, `constrain_show_artifacts`, `constrain_search_sessions`
**Resources**: `constrain://sessions`, `constrain://session/{id}`, `constrain://artifacts/{id}`

## Dependencies

- [Click](https://click.palletsprojects.com/) (CLI)
- [Pydantic](https://docs.pydantic.dev/) (data models)
- [PyYAML](https://pyyaml.org/) (artifact output and validation)
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) (optional, default backend)
- [OpenAI SDK](https://github.com/openai/openai-python) (optional, OpenAI-compatible backend)
- [MCP](https://github.com/modelcontextprotocol/python-sdk) (optional, MCP server)

## License

[MIT](LICENSE)
