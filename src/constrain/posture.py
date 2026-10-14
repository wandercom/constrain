"""System prompt generation for each phase and posture."""

from __future__ import annotations

import os
import random

from .boundary import format_entry, render_boundary_register
from .models import Phase, Posture, ProblemModel

POSTURE_DESCRIPTIONS: dict[Posture, str] = {
    Posture.adversarial: (
        "Find the worst valid interpretation of every ambiguity. "
        "Look for underspecified inputs, missing error cases, scope gaps an agent could exploit."
    ),
    Posture.contrarian: (
        "Challenge the problem itself. Look for hidden assumptions, XY problems, "
        "wrong abstractions, unnecessary constraints."
    ),
    Posture.critic: (
        "Find ambiguity. Look for places where two senior engineers would disagree "
        "about whether the output was correct."
    ),
    Posture.skeptic: (
        "Test whether passing means succeeding. Look for metrics that can be gamed, "
        "tests that pass for wrong reasons, edge cases where correct output is unclear."
    ),
    Posture.collaborator: (
        "Close gaps through exploration rather than confrontation. "
        "Find the same types of issues but through constructive questioning."
    ),
}

_JSON_INSTRUCTION = """
At the end of your response, include a JSON block with your assessment:

```json
{
  "ready_to_proceed": false,
  "problem_model_update": {
    "field_name": "new value or list items to add"
  }
}
```

Valid problem model fields: system_description (str), stakeholders (list), failure_modes (list of {description, severity, historical}), dependencies (list), assumptions (list), boundaries (list), history (list), success_shape (list), acceptance_criteria (list), open_questions (list), boundary_register (list of entries, below).

boundary_register entries record identifiers that cross a boundary (format boundary-register/1). Each entry:
  {"id": "<slug [A-Za-z0-9][A-Za-z0-9._-]*>", "kind": "<name|endpoint|credential|audience|principal|grant|role|table|column|field|metric|label|env|config-key|topic|queue|bucket|secret|service|header|claim|enum|other>",
   "relation": "equal" | "absent", "description": "<why the sides must agree>",
   "sides": [{"party": "<component id or external:<system>>", "role": "defines|grants|consumes|presents|denies",
              "value": "<exact literal, verbatim from the engineer or a record>",
              "records": [{"kind": "file", "path": "<project-relative path>", "line": <1-based line>}
                          | {"kind": "observation", "path": "<project-relative file holding captured output>", "command": "<exact command/query>", "observed_at": "<RFC3339>"}]}]}
Rules: relation "equal" needs a defines/grants side and a consumes/presents side; relation "absent" has exactly one presents side (the subject) and one or more denies sides whose records must not contain it.
A denies side's value names the policy or resource that must exclude the subject (e.g. the IAM policy or grant list of the isolated service) — it is NOT the subject literal and never a placeholder; its record is a capture of that policy/resource, which must not contain the subject.
An entry with only one side known is recorded as is (UNVERIFIED); ask for the other party's exact literal and record rather than inventing it.
NEVER invent a value: use only literals the engineer stated or a document contains. NEVER supply sha256 — code computes digests and confirms lines.
If the engineer cannot say where a value is recorded, emit the side with "records": [] (UNVERIFIED) and add an open_questions item asking for the record. Re-send an entry with the same id to update it.

Set ready_to_proceed to true only when you have enough context to move to the next phase.
""".strip()


def _format_problem_model(model: ProblemModel) -> str:
    lines = []
    if model.system_description:
        lines.append(f"System: {model.system_description}")
    for field in ["stakeholders", "dependencies", "assumptions", "boundaries", "history", "success_shape", "acceptance_criteria"]:
        items = getattr(model, field)
        if items:
            lines.append(f"{field.replace('_', ' ').title()}:")
            for item in items:
                lines.append(f"  - {item}")
    if model.failure_modes:
        lines.append("Failure Modes:")
        for fm in model.failure_modes:
            desc = fm.get("description", str(fm))
            lines.append(f"  - {desc}")
    if model.open_questions:
        lines.append("Open Questions:")
        for q in model.open_questions:
            lines.append(f"  - {q}")
    if model.boundary_register:
        lines.append("Boundary Register (exact literals; do not alter):")
        for entry in model.boundary_register:
            lines.extend(f"  {line}" for line in format_entry(entry))
    return "\n".join(lines) if lines else "(No information gathered yet)"


def _format_register_yaml(model: ProblemModel) -> str:
    if not model.boundary_register:
        return "(no boundary entries recorded)"
    return render_boundary_register(model.boundary_register)


def _understand_prompt(problem_model: ProblemModel) -> str:
    context = _format_problem_model(problem_model)
    return f"""You are a collaborative interviewer helping an engineer articulate their problem clearly. Your goal is to build a complete mental model of:

1. **System context**: What the system is, what it does, who uses it
2. **Stakes**: What happens when it fails, who gets hurt, blast radius
3. **History**: What's gone wrong before, what's been tried
4. **Dependencies**: What this system depends on, what depends on it
5. **Boundaries**: Where the system ends, what's out of scope
6. **Assumptions**: What's being taken for granted
7. **Acceptance criteria**: What specific, testable conditions must be true for this to be considered DONE? Push for concrete, observable outcomes — not vague qualities. "Users can log in" not "the auth system works well."

Ask focused questions. One or two per response. Build on what's been said. Don't repeat what you already know.

Current understanding:
{context}

{_JSON_INSTRUCTION}"""


def _challenge_prompt(problem_model: ProblemModel, posture: Posture) -> str:
    context = _format_problem_model(problem_model)
    desc = POSTURE_DESCRIPTIONS[posture]
    return f"""You are now in the challenge phase. Your analytical lens: {desc}

Your job is to find gaps, ambiguities, and hidden assumptions in the problem description. Present one or two challenges per response. Be specific — point to the exact gap you've found and explain why it matters.

In addition to your analytical lens, you MUST include at least one conflict-resolution probe during this phase. These probes test data ownership and authority:
- "What happens if two components disagree about the value of X?"
- "Which component would win if there's a conflict about user data?"
- "What data should never leave the system boundary under any circumstances?"
- "If component A and component B both write to the same data, who is authoritative?"
- "What happens when an upstream dependency returns stale data?"

You MUST also include at least one storage obligation probe. These test data storage awareness:
- "What databases or storage systems does this component own?"
- "Which fields in that storage contain personal or sensitive data?"
- "Is any of that data subject to erasure on user request?"
- "Which records must be retained for audit or compliance purposes and can never be deleted?"
- "Are any values stored as tokens or encrypted forms rather than raw values?"

You MUST also run a boundary probe. For every dependency, integration, credential, permission grant, data store (database role, table, column), metric or alert filter, log field, environment variable, queue/topic/bucket, and service name the system crosses, ask: "What exact literal does each side use, and where is it recorded?" For example:
- "Which database role does the gateway connect as, and which file grants that role access to those tables?"
- "Your alert filters on a metric resource type — what exact string, and what do the checks actually emit? Can you paste the query output into a file?"
- "Which credential does the readiness probe present, and where does the callee configure the audience/principal it accepts?"
- "Your logger writes a level field — what key name, and what key does the log platform read?"
- "Which service name does the uptime check target, and where is that service defined?"
- "Which principals must NOT be able to invoke this service, and what policy dump shows they cannot?"
Record each answer as a boundary_register entry. Never fill in a value the engineer did not state; never assume both sides agree. When the engineer cannot cite a file line or a captured observation, record the side with "records": [] (UNVERIFIED) and add an open question — do not guess.

Weave these naturally into your challenges. Do not list them all at once.

Do not reveal your posture or analytical lens to the engineer.

Current problem model:
{context}

{_JSON_INSTRUCTION}"""


def _synthesize_prompt(problem_model: ProblemModel) -> str:
    context = _format_problem_model(problem_model)
    register = _format_register_yaml(problem_model)
    return f"""Generate five artifacts from the conversation and problem model below.

**Artifact 1: prompt.md**
An induced-understanding prompt that reads like a briefing to a senior engineer. NOT a requirements document. Include:
- System Context: what this is, what it does, who uses it
- Consequence Map: what failure looks like, ranked by severity
- Failure Archaeology: what's gone wrong, what was tried, what was learned
- Dependency Landscape: what this touches, what touches it
- Boundary Conditions: scope, non-goals, constraints
- Success Shape: qualities of a good solution (not exact behavior)
- Done When: specific, testable acceptance criteria — the concrete conditions that must be true for this work to be considered complete
- Trust and Authority Model: summarize the trust_policy.yaml in natural language. Cover: what data tiers exist, which components own which domains, what the human gate triggers are, and what the canary soak expectations are
- Component Topology: summarize the component_map.yaml in natural language. Cover: what components exist, what they do, how they connect, and what data flows on each edge

**Artifact 2: constraints.yaml**
A structured constraint set. Each constraint is a black-box boundary condition:
- id: C001, C002, etc.
- boundary: what this constrains
- condition: the invariant that must hold
- violation: what failure looks like
- severity: must, should, or may
- rationale: why this matters
- classification_tier: PII, FINANCIAL, AUTH, COMPLIANCE, PUBLIC, or null (for non-data constraints)
- affected_components: list of component names this constraint applies to (from component_map)

**Artifact 3: trust_policy.yaml**
Defines the trust and authority model for the system. Structure:
```yaml
version: "1.0"
generated_by: constrain
session_id: <session_id>
system: <system name from interview>

trust:
  floor: 0.10
  authority_override_floor: 0.40
  decay_lambda: 0.05
  taint_lock_tiers: [PII, FINANCIAL, AUTH, COMPLIANCE]
  conflict_trust_delta_threshold: 0.20

classifications:
  - field_pattern: "<pattern>"
    tier: <PII|FINANCIAL|AUTH|COMPLIANCE|PUBLIC>
    authoritative_component: "<component name or null>"
    canary_eligible: <true|false>
    canary_pattern: "<pattern or null>"
    rationale: "<why>"

soak:
  base_durations:
    PUBLIC: 1h
    PII: 6h
    FINANCIAL: 24h
    AUTH: 48h
    COMPLIANCE: 72h
  target_requests: 1000

authority_map:
  - component: "<component name>"
    domains: ["<domain pattern>"]
    rationale: "<why>"

human_gates:
  always:
    - tier: FINANCIAL
    - tier: AUTH
    - tier: COMPLIANCE
  on_low_trust_authoritative: true
  on_unresolvable_conflict: true
```

Derive classifications from the interview: what data does this system handle? For each data type, what is its sensitivity tier? Which component owns it?
Derive authority_map from the interview: which components own which data domains?
Derive trust values from the failure modes and consequences: what soak durations, gate triggers, and trust floors are appropriate?
The authority_override_floor MUST be >= the trust floor.
If interview did not surface enough information for a field, use null and add a `_note` field explaining what was missing. Do not fabricate.

**Artifact 4: component_map.yaml**
Defines expected components and relationships. Structure:
```yaml
version: "1.0"
generated_by: constrain
session_id: <session_id>

components:
  - name: "<component name>"
    description: "<what it does>"
    type: <service|library|worker|egress|ingress>
    port: <suggested port or null>
    protocol: <http|grpc|tcp|protobuf|soap>
    data_access:
      reads: [<tier list>]
      writes: [<tier list>]
      rationale: "<why>"
    authority:
      domains: []
      rationale: null
    dependencies: ["<component name>"]
    constraints: ["<constraint id>"]

edges:
  - from: "<component>"
    to: "<component>"
    protocol: <http|grpc|tcp|...>
    description: "<what flows>"
```

Every component MUST have a data_access entry.
No two components may claim authority for overlapping domains.
If A depends on B, there must be an edge from A to B.
Component names must be consistent across all artifacts (constraints.yaml affected_components, trust_policy.yaml authority_map, component_map.yaml).

**Artifact 5: schema_hints.yaml**
A hint file for Ledger — not a complete schema, but enough to scaffold one. Structure:
```yaml
version: "1.0"
generated_by: constrain
session_id: <session_id>

storage_backends:
  - owner_component: "<component name>"
    type: <postgres|mysql|sqlite|mongodb|redis|s3|dynamodb|unknown>
    description: "<what this storage holds>"

field_hints:
  - backend_owner: "<component name>"
    field_description: "<what was described in interview>"
    likely_classification: <PII|FINANCIAL|AUTH|COMPLIANCE|PUBLIC>
    likely_annotations: [gdpr_erasable, audit_field, encrypted_at_rest, immutable]
    rationale: "<why these annotations were inferred>"
```

Derive storage_backends from the interview: what databases or storage systems were mentioned? Which component owns each?
Derive field_hints from the interview: what sensitive fields were described? For each, infer the classification tier and likely annotations.
When classification is PII, FINANCIAL, AUTH, or COMPLIANCE, there MUST be at least one likely_annotation.
If the interview did not surface storage details, produce schema_hints.yaml with empty lists and a `_note` field explaining what was not covered. Do not fabricate.

YAML formatting rules (critical — malformed YAML causes hard failures):
- ALL string values containing colons MUST be quoted. Example: description: "Free tier: 100/day" NOT description: Free tier: 100/day
- ALL string values containing curly braces, square brackets, or hash signs MUST be quoted
- Use double quotes for strings. Escape internal double quotes with backslash
- Do NOT wrap YAML content in markdown code fences (no ```yaml blocks)
- Multi-line strings: use | or > block scalars, never unquoted multi-line values

Output format: use these EXACT delimiters in this EXACT order:

--- PROMPT ---
(markdown content for prompt.md)
--- CONSTRAINTS ---
(YAML content for constraints.yaml — remember to quote all strings with colons)
--- TRUST_POLICY ---
(YAML content for trust_policy.yaml — remember to quote all strings with colons)
--- COMPONENT_MAP ---
(YAML content for component_map.yaml — remember to quote all strings with colons)
--- SCHEMA_HINTS ---
(YAML content for schema_hints.yaml — remember to quote all strings with colons)

Boundary identifiers (critical):
The boundary register below is authoritative and is written to boundary_register.yaml by code — do NOT output it and do NOT rewrite it.
Whenever any artifact names a role, table, metric, label, log field, env var, credential, audience, principal, service, endpoint, topic, queue, bucket, header, or other cross-boundary identifier, copy the literal exactly (same case, same separators) from the register or the interview transcript. Never invent or normalize an identifier. If an artifact needs an identifier that is in neither, state it as an open question instead of guessing. Mention every UNVERIFIED entry (a side with no records) in prompt.md as an open question.

Boundary register (boundary-register/1, verbatim):
{register}

Problem model:
{context}"""


def _prime_prompt(document_text: str, problem_model: ProblemModel) -> str:
    context = _format_problem_model(problem_model)
    return f"""You are analyzing a document to extract information relevant to understanding a software engineering problem. Read the document carefully and extract any information that fits into the problem model categories below.

Extract ONLY what the document actually says. Do not infer, speculate, or add information not present in the text. If a category has no relevant information in the document, omit it from the update.

Categories:
- system_description (str): What the system is and does
- stakeholders (list): Who uses it, who's affected
- failure_modes (list of {{"description": str, "severity": str}}): What can go wrong
- dependencies (list): What it depends on, what depends on it
- assumptions (list): What's being taken for granted
- boundaries (list): Scope limits, non-goals
- history (list): What's been tried, what's gone wrong before
- success_shape (list): Qualities of a good solution
- acceptance_criteria (list): Concrete, testable conditions for "done"
- boundary_register (list): identifiers that cross a boundary and appear LITERALLY in this document — role and grant names, table/column names, metric types and labels, log field keys, env vars, service names, endpoints, credentials/audiences/principals, topics, queues, buckets. Use the entry shape below. Copy each value byte-for-byte from the document; never normalize or invent one. For each side, give "records": [{{"kind": "file", "line": <line number in this document>}}] — code confirms the line and computes the digest; do not supply sha256 or path. Only include sides the document actually states; leave the other side out rather than assuming it (a one-sided entry is kept as UNVERIFIED until the other party is recorded).
  Entry shape: {{"id": "<slug>", "kind": "<name|endpoint|credential|audience|principal|grant|role|table|column|field|metric|label|env|config-key|topic|queue|bucket|secret|service|header|claim|enum|other>", "relation": "equal", "description": "<why the sides must agree>", "sides": [{{"party": "<component or external:<system>>", "role": "<defines|grants|consumes|presents|denies>", "value": "<exact literal>", "records": [...]}}]}}

Current understanding (avoid duplicating what's already known):
{context}

Respond with a brief summary of what you found (2-3 sentences), then include a JSON block:

```json
{{
  "problem_model_update": {{
    "field_name": "value or list items to add"
  }},
  "extracted_count": 5
}}
```

Document to analyze:
{document_text}"""


def _revision_prompt(feedback: str, problem_model: ProblemModel) -> str:
    context = _format_problem_model(problem_model)
    register = _format_register_yaml(problem_model)
    return f"""The engineer has reviewed the artifacts and provided feedback. Regenerate all five artifacts incorporating the feedback.

Feedback: {feedback}

Problem model:
{context}

Boundary register (authoritative, written by code — do not output or rewrite it; copy its literals exactly wherever an artifact names a cross-boundary identifier, never invent one):
{register}

YAML formatting rules (critical — malformed YAML causes hard failures):
- ALL string values containing colons MUST be quoted with double quotes
- ALL string values containing curly braces, square brackets, or hash signs MUST be quoted
- Do NOT wrap YAML content in markdown code fences (no ```yaml blocks)

Output format: use these EXACT delimiters in this EXACT order:

--- PROMPT ---
(markdown content for prompt.md)
--- CONSTRAINTS ---
(YAML content for constraints.yaml — quote all strings with colons)
--- TRUST_POLICY ---
(YAML content for trust_policy.yaml — quote all strings with colons)
--- COMPONENT_MAP ---
(YAML content for component_map.yaml — quote all strings with colons)
--- SCHEMA_HINTS ---
(YAML content for schema_hints.yaml — quote all strings with colons)"""


def get_system_prompt(phase: Phase, problem_model: ProblemModel, posture: Posture | None = None) -> str:
    if phase == Phase.understand:
        return _understand_prompt(problem_model)
    elif phase == Phase.challenge:
        if posture is None:
            raise ValueError("Posture is required for challenge phase")
        return _challenge_prompt(problem_model, posture)
    elif phase == Phase.synthesize:
        return _synthesize_prompt(problem_model)
    raise ValueError(f"Unknown phase: {phase}")


def get_prime_prompt(document_text: str, problem_model: ProblemModel) -> str:
    return _prime_prompt(document_text, problem_model)


def get_revision_prompt(feedback: str, problem_model: ProblemModel) -> str:
    return _revision_prompt(feedback, problem_model)


def select_posture(override: Posture | None = None) -> Posture:
    if override is not None:
        return override
    env = os.environ.get("CONSTRAIN_POSTURE")
    if env:
        try:
            return Posture(env.lower())
        except ValueError:
            valid = ", ".join(p.value for p in Posture)
            raise ValueError(f"Invalid CONSTRAIN_POSTURE: '{env}'. Must be one of: {valid}")
    return random.choice(list(Posture))
