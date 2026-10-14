"""Artifact generation: parsing LLM synthesis output and writing files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SynthesisArtifacts:
    """All artifacts produced during synthesis.

    Supports iteration/unpacking for backward compatibility:
        prompt_md, constraints_yaml = parse_synthesis_output(raw)
    """

    prompt_md: str
    constraints_yaml: str
    trust_policy_yaml: str = ""
    component_map_yaml: str = ""
    schema_hints_yaml: str = ""

    def __iter__(self):
        yield self.prompt_md
        yield self.constraints_yaml
        yield self.trust_policy_yaml
        yield self.component_map_yaml
        yield self.schema_hints_yaml

    def __getitem__(self, index):
        return (
            self.prompt_md,
            self.constraints_yaml,
            self.trust_policy_yaml,
            self.component_map_yaml,
            self.schema_hints_yaml,
        )[index]

    def __len__(self):
        return 5


# Ordered delimiters — must appear in this order in the LLM output
_DELIMITERS = [
    "--- PROMPT ---",
    "--- CONSTRAINTS ---",
    "--- TRUST_POLICY ---",
    "--- COMPONENT_MAP ---",
    "--- SCHEMA_HINTS ---",
]


def parse_synthesis_output(raw: str) -> SynthesisArtifacts:
    """Parse delimited LLM output into four artifacts.

    Returns a SynthesisArtifacts with prompt_md, constraints_yaml,
    trust_policy_yaml, and component_map_yaml.

    For backward compatibility, also works when called expecting a 2-tuple:
    the returned SynthesisArtifacts can be iterated to get (prompt_md, constraints_yaml).
    """
    positions = []
    for delim in _DELIMITERS:
        idx = raw.find(delim)
        if idx == -1:
            # PROMPT and CONSTRAINTS are required; others are optional
            if delim in (_DELIMITERS[0], _DELIMITERS[1]):
                raise ValueError(
                    f"Synthesis output missing required delimiter '{delim}'."
                )
            positions.append(-1)
        else:
            positions.append(idx)

    def _extract(start_idx: int, delim: str, next_positions: list[int]) -> str:
        if start_idx == -1:
            return ""
        content_start = start_idx + len(delim)
        # Find the next delimiter that exists
        end = len(raw)
        for np in next_positions:
            if np != -1:
                end = np
                break
        return raw[content_start:end].strip()

    prompt_md = _extract(
        positions[0], _DELIMITERS[0], [positions[1], positions[2], positions[3], positions[4]]
    )
    constraints_yaml = _extract(
        positions[1], _DELIMITERS[1], [positions[2], positions[3], positions[4]]
    )
    trust_policy_yaml = _extract(
        positions[2], _DELIMITERS[2], [positions[3], positions[4]]
    )
    component_map_yaml = _extract(
        positions[3], _DELIMITERS[3], [positions[4]]
    )
    schema_hints_yaml = _extract(
        positions[4], _DELIMITERS[4], []
    )

    return SynthesisArtifacts(
        prompt_md=prompt_md,
        constraints_yaml=constraints_yaml,
        trust_policy_yaml=trust_policy_yaml,
        component_map_yaml=component_map_yaml,
        schema_hints_yaml=schema_hints_yaml,
    )


def sanitize_yaml(content: str) -> str:
    """Fix common LLM YAML generation mistakes before parsing.

    Handles:
    - Markdown code fences wrapping YAML content
    - Unquoted string values containing colons (the most common LLM error)
    """
    # Strip markdown code fences
    content = re.sub(r'^```(?:yaml)?\s*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```\s*$', '', content, flags=re.MULTILINE)

    lines = content.split('\n')
    fixed = []
    for line in lines:
        stripped = line.lstrip()
        # Skip comments, blank lines, block scalars, list-only lines
        if not stripped or stripped.startswith('#') or stripped in ('|', '>', '|-', '>-'):
            fixed.append(line)
            continue

        # Match YAML key-value lines: "  key: value"
        # Only fix lines where the VALUE portion contains an unquoted colon
        m = re.match(r'^(\s*-?\s*)([A-Za-z_][A-Za-z0-9_]*):\s+(.+)$', line)
        if m:
            indent, key, value = m.group(1), m.group(2), m.group(3)
            # Don't touch values that are already quoted, are numbers, booleans,
            # null, lists, or anchors/aliases
            if (value.startswith('"') or value.startswith("'") or
                    value.startswith('[') or value.startswith('{') or
                    value.startswith('&') or value.startswith('*') or
                    re.match(r'^(true|false|null|~|\d+\.?\d*([eE][+-]?\d+)?|0x[0-9a-fA-F]+)$',
                             value, re.IGNORECASE)):
                fixed.append(line)
                continue

            # If the value contains a colon followed by space (or end), it needs quoting
            if re.search(r':\s|:$', value):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                fixed.append(f'{indent}{key}: "{escaped}"')
                continue

        fixed.append(line)

    return '\n'.join(fixed)


def validate_yaml_content(content: str, name: str) -> dict | list | None:
    """Parse YAML content and raise ValueError if invalid.

    Sanitizes common LLM formatting errors before parsing.
    Returns the parsed data structure or None if content is empty.
    """
    if not content.strip():
        return None
    content = sanitize_yaml(content)
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {name}: {e}")


def validate_artifacts(artifacts: SynthesisArtifacts) -> list[str]:
    """Cross-validate all artifacts for internal consistency.

    Returns a list of warning strings. Raises ValueError for hard errors
    (invalid YAML).
    """
    warnings: list[str] = []

    # Validate YAML documents
    constraints_data = validate_yaml_content(
        artifacts.constraints_yaml, "constraints.yaml"
    )
    trust_data = validate_yaml_content(
        artifacts.trust_policy_yaml, "trust_policy.yaml"
    )
    component_data = validate_yaml_content(
        artifacts.component_map_yaml, "component_map.yaml"
    )
    schema_data = validate_yaml_content(
        artifacts.schema_hints_yaml, "schema_hints.yaml"
    )

    # Extract component names from component_map
    cm_components: set[str] = set()
    if isinstance(component_data, dict):
        for comp in component_data.get("components", []):
            if isinstance(comp, dict) and "name" in comp:
                cm_components.add(comp["name"])

    # Extract component names from trust_policy authority_map
    tp_components: set[str] = set()
    if isinstance(trust_data, dict):
        for entry in trust_data.get("authority_map", []):
            if isinstance(entry, dict) and "component" in entry:
                tp_components.add(entry["component"])

    # Cross-validate: authority_map components must exist in component_map
    if tp_components and cm_components:
        orphans = tp_components - cm_components
        if orphans:
            warnings.append(
                f"trust_policy.yaml authority_map references components not in "
                f"component_map.yaml: {sorted(orphans)}"
            )

    # FA-C-013: No two components should claim authority for overlapping domains
    if isinstance(component_data, dict):
        domain_owners: dict[str, str] = {}
        for comp in component_data.get("components", []):
            if not isinstance(comp, dict):
                continue
            name = comp.get("name", "")
            authority = comp.get("authority", {})
            if not isinstance(authority, dict):
                continue
            for domain in authority.get("domains", []):
                if domain in domain_owners:
                    warnings.append(
                        f"Overlapping authority: both '{domain_owners[domain]}' and "
                        f"'{name}' claim domain '{domain}'"
                    )
                else:
                    domain_owners[domain] = name

    # FA-C-015: edges consistent with dependencies
    if isinstance(component_data, dict):
        edges_set: set[tuple[str, str]] = set()
        for edge in component_data.get("edges", []):
            if isinstance(edge, dict):
                edges_set.add((edge.get("from", ""), edge.get("to", "")))
        for comp in component_data.get("components", []):
            if not isinstance(comp, dict):
                continue
            name = comp.get("name", "")
            for dep in comp.get("dependencies", []):
                if (name, dep) not in edges_set:
                    warnings.append(
                        f"Component '{name}' depends on '{dep}' but no edge "
                        f"from '{name}' to '{dep}' exists"
                    )

    # FA-C-020: authority_override_floor >= trust.floor
    if isinstance(trust_data, dict):
        trust_cfg = trust_data.get("trust", {})
        if isinstance(trust_cfg, dict):
            floor = trust_cfg.get("floor")
            override_floor = trust_cfg.get("authority_override_floor")
            if (
                floor is not None
                and override_floor is not None
                and override_floor < floor
            ):
                warnings.append(
                    f"trust_policy.yaml: authority_override_floor ({override_floor}) "
                    f"is less than trust.floor ({floor})"
                )

    # FA-C-023: field_hints with sensitive tiers must have at least one annotation
    _SENSITIVE_TIERS = {"PII", "FINANCIAL", "AUTH", "COMPLIANCE"}
    if isinstance(schema_data, dict):
        for hint in schema_data.get("field_hints", []):
            if not isinstance(hint, dict):
                continue
            tier = hint.get("likely_classification")
            annotations = hint.get("likely_annotations", [])
            if tier in _SENSITIVE_TIERS and not annotations:
                warnings.append(
                    f"schema_hints.yaml: field_hint with classification "
                    f"'{tier}' has no likely_annotations "
                    f"(field: {hint.get('field_description', '?')})"
                )

    return warnings


_ALL_ARTIFACT_FILES = [
    "prompt.md",
    "constraints.yaml",
    "trust_policy.yaml",
    "component_map.yaml",
    "schema_hints.yaml",
    "boundary_register.yaml",
]


_OPTIONAL_ARTIFACT_FILES = {
    "trust_policy.yaml",
    "component_map.yaml",
    "schema_hints.yaml",
    "boundary_register.yaml",
}

# Written by code from the typed register; emitted verbatim (never "sanitized",
# which could alter a literal).
_VERBATIM_ARTIFACT_FILES = {"boundary_register.yaml"}


def write_artifacts(
    prompt_md: str,
    constraints_yaml: str,
    output_dir: str | Path,
    overwrite: bool = False,
    trust_policy_yaml: str = "",
    component_map_yaml: str = "",
    schema_hints_yaml: str = "",
    boundary_register_yaml: str = "",
) -> list[Path]:
    """Write all artifacts to output_dir.

    Returns list of paths written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "prompt.md": prompt_md,
        "constraints.yaml": constraints_yaml,
        "trust_policy.yaml": trust_policy_yaml,
        "component_map.yaml": component_map_yaml,
        "schema_hints.yaml": schema_hints_yaml,
        "boundary_register.yaml": boundary_register_yaml,
    }

    # Check for existing files
    if not overwrite:
        existing = [out / name for name in files if (out / name).exists()]
        if existing:
            names = ", ".join(p.name for p in existing)
            raise FileExistsError(
                f"Artifact(s) already exist: {names}. Use overwrite=True to replace."
            )

    written: list[Path] = []
    for name, content in files.items():
        if not content and name in _OPTIONAL_ARTIFACT_FILES:
            continue  # Skip empty optional artifacts (backward compat for old sessions)
        # Sanitize YAML artifacts before writing
        if name.endswith(".yaml") and name not in _VERBATIM_ARTIFACT_FILES:
            content = sanitize_yaml(content)
        path = out / name
        path.write_text(content, encoding="utf-8")
        written.append(path)

    return written
