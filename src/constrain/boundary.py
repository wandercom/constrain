"""Boundary register: mechanical verification and code-computed evidence.

Invariant: no identifier that crosses a boundary is taken on faith. Every side
of a register entry states the exact literal it uses and cites a record (a
file line or a captured observation) whose bytes are pinned by sha256. This
module checks those claims without an LLM:

- ``validate_boundary_register`` re-reads every record from the project root,
  re-hashes it, and checks the literal is actually there (as a token) — or,
  for ``relation: absent``, that it is not.
- ``anchor_boundary_update`` turns a model-proposed citation (path + line)
  into a record whose line and digest are computed here, never by the model.
  A citation that does not check out is dropped and the side stays
  UNVERIFIED with an open question; nothing is guessed.

Every finding about a register is an error except ``single-source`` (all
sides of an equality attested by one file) and an empty register, which
are warnings. ``unverified`` is an error that synthesis tolerates (the
entry is written and flagged) but ``constrain validate`` does not.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml
from pydantic import ValidationError

from .models import (
    BOUNDARY_REGISTER_SCHEMA,
    BoundaryEntry,
    BoundaryRegister,
    BoundarySide,
    FileRecord,
    ObservationRecord,
    check_record_path,
)

BOUNDARY_REGISTER_FILE = "boundary_register.yaml"

ERROR = "error"
WARNING = "warning"

# Codes that synthesis tolerates: the entry is written but flagged.
NON_BLOCKING_CODES = frozenset({"unverified"})

_TOKEN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
_SEPARATORS_RE = re.compile(r"[-_. ]")
_PLACEHOLDER_WORDS = ("todo", "tbd", "tbc", "fixme", "unknown", "assumed", "placeholder", "changeme")
_ANGLE_PLACEHOLDER_RE = re.compile(r"<[A-Za-z0-9_. -]*>")


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" | "warning"
    entry_id: str
    side: str | None
    code: str
    message: str

    @property
    def blocking(self) -> bool:
        """True if this finding must stop synthesis from completing."""
        return self.severity == ERROR and self.code not in NON_BLOCKING_CODES

    def __str__(self) -> str:
        where = f"{self.entry_id} [{self.side}]" if self.side else self.entry_id
        return f"{self.severity.upper()} {where} {self.code}: {self.message}"


def blocking_findings(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.blocking]


def errors(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == ERROR]


# ── primitives ──────────────────────────────────────────────


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contains_token(haystack: str, value: str) -> bool:
    """True if *value* occurs in *haystack* as a whole token.

    If the value starts (ends) with an identifier character [A-Za-z0-9_-], the
    character before (after) the occurrence must not be one, or be the start
    (end) of the text. So ``mea-auth`` does not match inside
    ``mea-auth-runtime``, while ``level`` matches in ``"level":``.
    """
    if not value:
        return False
    check_start = value[0] in _TOKEN_CHARS
    check_end = value[-1] in _TOKEN_CHARS
    start = 0
    while True:
        i = haystack.find(value, start)
        if i < 0:
            return False
        end = i + len(value)
        ok_start = not check_start or i == 0 or haystack[i - 1] not in _TOKEN_CHARS
        ok_end = not check_end or end == len(haystack) or haystack[end] not in _TOKEN_CHARS
        if ok_start and ok_end:
            return True
        start = i + 1


def token_lines(text: str, value: str) -> list[int]:
    """1-based line numbers whose content contains *value* as a token."""
    return [n for n, line in enumerate(_split_lines(text), start=1) if contains_token(line, value)]


def _split_lines(text: str) -> list[str]:
    # Split on "\n" only so line numbers agree with editors and grep -n.
    return [line.rstrip("\r") for line in text.split("\n")]


def is_placeholder(value: str) -> bool:
    """True for values that stand in for a literal nobody looked up."""
    v = value.strip()
    if not v:
        return True
    if "???" in v or _ANGLE_PLACEHOLDER_RE.search(v):
        return True
    lowered = v.lower()
    return any(contains_token(lowered, word) for word in _PLACEHOLDER_WORDS)


def normalize_near_miss(value: str) -> str:
    return _SEPARATORS_RE.sub("", value).lower()


def resolve_record_path(project_root: Path, path: str) -> Path:
    """Resolve a record path inside *project_root*; raise ValueError otherwise.

    Rejects absolute paths and ``..`` segments, and (after resolving symlinks)
    anything that lands outside the root.
    """
    check_record_path(path)
    root = project_root.resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"record path {path!r} resolves outside the project root")
    return target


def _side_label(side: BoundarySide | dict, index: int) -> str:
    if isinstance(side, BoundarySide):
        return f"sides[{index}] {side.party}/{side.role}"
    party = side.get("party", "?") if isinstance(side, dict) else "?"
    role = side.get("role", "?") if isinstance(side, dict) else "?"
    return f"sides[{index}] {party}/{role}"


# ── verification ────────────────────────────────────────────


def _read_record(
    record: FileRecord | ObservationRecord, project_root: Path
) -> tuple[str | None, list[tuple[str, str]]]:
    """Return (decoded text, problems). Text is None if the record cannot be trusted."""
    try:
        target = resolve_record_path(project_root, record.path)
    except ValueError as e:
        return None, [("path", str(e))]
    if not target.is_file():
        return None, [("missing-record", f"record file {record.path!r} does not exist")]
    try:
        data = target.read_bytes()
    except OSError as e:
        return None, [("missing-record", f"record file {record.path!r} unreadable: {e}")]
    actual = sha256_bytes(data)
    if actual != record.sha256.lower():
        return None, [(
            "stale",
            f"stale record: {record.path} has sha256 {actual}, register pins "
            f"{record.sha256}; the file changed since the value was recorded — re-capture it",
        )]
    return data.decode("utf-8", errors="replace"), []


def _verify_presence(
    record: FileRecord | ObservationRecord, value: str, project_root: Path
) -> list[tuple[str, str]]:
    text, problems = _read_record(record, project_root)
    if text is None:
        return problems
    if isinstance(record, FileRecord):
        lines = _split_lines(text)
        if record.line > len(lines):
            return [(
                "line-out-of-range",
                f"{record.path}:{record.line} is past the end of the file ({len(lines)} lines)",
            )]
        if not contains_token(lines[record.line - 1], value):
            found = token_lines(text, value)
            hint = f" (it does appear at line(s) {found[:5]})" if found else " (it appears nowhere in the file)"
            return [(
                "line-lacks-value",
                f"{record.path}:{record.line} does not contain {value!r} as a token{hint}",
            )]
        return []
    if not contains_token(text, value):
        return [(
            "value-not-observed",
            f"observation {record.path} (from `{record.command}`) does not contain {value!r} as a token",
        )]
    return []


def _verify_absence(
    record: FileRecord | ObservationRecord, subject: str, project_root: Path
) -> list[tuple[str, str]]:
    text, problems = _read_record(record, project_root)
    if text is None:
        return problems
    if contains_token(text, subject):
        found = token_lines(text, subject)
        return [(
            "absence-violated",
            f"{record.path} contains the subject {subject!r} (line(s) {found[:5]}); "
            "the denying side does not deny it",
        )]
    return []


def _verify_entry(entry: BoundaryEntry, project_root: Path) -> list[Finding]:
    out: list[Finding] = []

    def err(side: str | None, code: str, msg: str) -> None:
        out.append(Finding(ERROR, entry.id, side, code, msg))

    labels = [_side_label(s, i) for i, s in enumerate(entry.sides)]

    # Per-side checks common to both relations.
    for side, label in zip(entry.sides, labels):
        if is_placeholder(side.value):
            err(label, "placeholder", f"value {side.value!r} is a placeholder, not a literal")
        if not side.records:
            err(label, "unverified", f"UNVERIFIED: no record shows where {side.value!r} is set")

    if entry.relation == "equal":
        for side, label in zip(entry.sides, labels):
            if side.role == "denies":
                err(label, "relation-shape", "role 'denies' is only valid for relation 'absent'")
        definers = [i for i, s in enumerate(entry.sides) if s.role in ("defines", "grants")]
        users = [i for i, s in enumerate(entry.sides) if s.role in ("consumes", "presents")]
        if not definers:
            err(None, "relation-shape", "relation 'equal' needs at least one side with role defines or grants")
        if not users:
            err(None, "relation-shape", "relation 'equal' needs at least one side with role consumes or presents")
        ref_i = definers[0] if definers else 0
        ref = entry.sides[ref_i]
        for i, side in enumerate(entry.sides):
            if i == ref_i or side.role == "denies" or side.value == ref.value:
                continue
            near = normalize_near_miss(side.value) == normalize_near_miss(ref.value)
            err(
                labels[i],
                "near-miss" if near else "mismatch",
                (f"near-miss: {side.value!r} differs from {ref.party}'s {ref.value!r} only by case or separators"
                 if near else f"{side.value!r} != {ref.party}'s {ref.value!r}"),
            )
        for side, label in zip(entry.sides, labels):
            for rec in side.records:
                for code, msg in _verify_presence(rec, side.value, project_root):
                    err(label, code, msg)
        cited = {r.path for s in entry.sides for r in s.records}
        if len(cited) == 1 and all(s.records for s in entry.sides) and len(entry.sides) > 1:
            out.append(Finding(
                WARNING, entry.id, None, "single-source",
                f"every side is attested only by {next(iter(cited))}; independent records "
                "from each side are stronger evidence",
            ))
    else:  # absent
        presenters = [i for i, s in enumerate(entry.sides) if s.role == "presents"]
        deniers = [i for i, s in enumerate(entry.sides) if s.role == "denies"]
        for side, label in zip(entry.sides, labels):
            if side.role not in ("presents", "denies"):
                err(label, "relation-shape", f"role {side.role!r} is not valid for relation 'absent' (use presents/denies)")
        if len(presenters) != 1:
            err(None, "relation-shape", f"relation 'absent' needs exactly one presents side (subject), found {len(presenters)}")
        if not deniers:
            err(None, "relation-shape", "relation 'absent' needs at least one denies side")
        if presenters:
            subject = entry.sides[presenters[0]]
            for rec in subject.records:
                for code, msg in _verify_presence(rec, subject.value, project_root):
                    err(labels[presenters[0]], code, msg)
            if len(presenters) == 1:
                for i in deniers:
                    for rec in entry.sides[i].records:
                        for code, msg in _verify_absence(rec, subject.value, project_root):
                            err(labels[i], code, msg)
    return out


def validate_boundary_register(
    register: BoundaryRegister | list[BoundaryEntry] | dict | None,
    project_root: str | Path,
) -> list[Finding]:
    """Verify every entry and record of a register against *project_root*.

    Accepts a typed register, a list of entries, or the raw mapping loaded
    from ``boundary_register.yaml``. Raw entries are validated one at a time
    so one malformed entry does not hide findings about the others.
    """
    root = Path(project_root)
    findings: list[Finding] = []
    entries: list[BoundaryEntry] = []

    if register is None:
        register = {}
    if isinstance(register, BoundaryRegister):
        entries = list(register.entries)
    elif isinstance(register, list):
        raw_entries: list = register
        entries, findings = _parse_raw_entries(raw_entries)
    elif isinstance(register, dict):
        schema = register.get("schema")
        if schema != BOUNDARY_REGISTER_SCHEMA:
            findings.append(Finding(
                ERROR, "<register>", None, "schema",
                f"schema must be {BOUNDARY_REGISTER_SCHEMA!r}, got {schema!r}",
            ))
        extra = sorted(set(register) - {"schema", "entries"})
        if extra:
            findings.append(Finding(ERROR, "<register>", None, "schema", f"unknown top-level keys: {extra}"))
        raw = register.get("entries", [])
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            findings.append(Finding(ERROR, "<register>", None, "schema", "entries must be a list"))
            raw = []
        parsed, problems = _parse_raw_entries(raw)
        entries = parsed
        findings.extend(problems)
    else:
        return [Finding(ERROR, "<register>", None, "schema", f"register must be a mapping, got {type(register).__name__}")]

    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            findings.append(Finding(ERROR, entry.id, None, "duplicate-id", f"duplicate entry id {entry.id!r}"))
            continue
        seen.add(entry.id)
        findings.extend(_verify_entry(entry, root))

    if not entries and not findings:
        findings.append(Finding(WARNING, "<register>", None, "empty", "no boundary entries recorded"))
    return findings


def _parse_raw_entries(raw: list) -> tuple[list[BoundaryEntry], list[Finding]]:
    entries: list[BoundaryEntry] = []
    findings: list[Finding] = []
    for i, item in enumerate(raw):
        if isinstance(item, BoundaryEntry):
            entries.append(item)
            continue
        label = item.get("id") if isinstance(item, dict) and isinstance(item.get("id"), str) else f"entries[{i}]"
        try:
            entries.append(BoundaryEntry.model_validate(item))
        except ValidationError as e:
            for problem in e.errors():
                loc = ".".join(str(p) for p in problem["loc"]) or "<entry>"
                code = "path" if loc.endswith(".path") else "invalid"
                findings.append(Finding(ERROR, label, None, code, f"{loc}: {problem['msg']}"))
    return entries, findings


# ── serialization ───────────────────────────────────────────


def render_boundary_register(entries: Iterable[BoundaryEntry]) -> str:
    """Serialize entries as ``boundary-register/1`` YAML, literals unchanged."""
    reg = BoundaryRegister(entries=list(entries))
    data = reg.model_dump(by_alias=True, mode="json")
    return yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, default_flow_style=False, width=float("inf")
    )


def load_boundary_register_text(text: str) -> dict:
    """Parse register YAML into a raw mapping (validate with validate_boundary_register)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {BOUNDARY_REGISTER_FILE}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{BOUNDARY_REGISTER_FILE} must be a mapping")
    return data


def format_entry(entry: BoundaryEntry) -> list[str]:
    """Human/LLM-readable lines for an entry; literals shown JSON-quoted, verbatim."""
    lines = [f"- [{entry.id}] kind={entry.kind} relation={entry.relation}: {entry.description}"]
    for side in entry.sides:
        if side.records:
            cites = ", ".join(
                f"{r.path}:{r.line}" if isinstance(r, FileRecord) else f"{r.path} (observed: {r.command})"
                for r in side.records
            )
            status = f"recorded at {cites}"
        else:
            status = "UNVERIFIED (no record)"
        lines.append(f"    {side.party} ({side.role}): {json.dumps(side.value)} — {status}")
    return lines


# ── anchoring (code computes line + digest, never the model) ──


def _relative_path(path: str, root: Path) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        try:
            return p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            raise ValueError(f"{path!r} is outside the project root")
    return path


def anchor_record(
    raw: dict,
    value: str,
    project_root: Path,
    *,
    default_path: str | None = None,
    expect_value: bool = True,
) -> tuple[FileRecord | ObservationRecord | None, str | None]:
    """Turn a cited location into a record with a code-computed digest.

    Returns (record, None) on success or (None, reason) when the citation does
    not check out. Any sha256 the caller supplied is ignored. With
    *expect_value* (every role except ``denies``) the literal must occur as a
    token: on the cited line if it does, else on the first line where it does.
    """
    if not isinstance(raw, dict):
        return None, "record is not a mapping"
    kind = raw.get("kind", "file")
    if kind not in ("file", "observation"):
        return None, f"unknown record kind {kind!r}"
    path = raw.get("path") or default_path
    if not isinstance(path, str) or not path.strip():
        return None, "no file was cited"
    try:
        rel = _relative_path(path, project_root)
        target = resolve_record_path(project_root, rel)
    except ValueError as e:
        return None, str(e)
    if not target.is_file():
        return None, f"cited file {rel!r} does not exist under the project root"
    try:
        data = target.read_bytes()
    except OSError as e:
        return None, f"cited file {rel!r} unreadable: {e}"
    digest = sha256_bytes(data)
    text = data.decode("utf-8", errors="replace")

    if kind == "observation":
        command = raw.get("command")
        if not isinstance(command, str) or not command.strip():
            return None, f"observation {rel!r} does not state the exact command that produced it"
        if expect_value and not contains_token(text, value):
            return None, f"{value!r} does not occur in observation {rel!r}"
        observed_at = raw.get("observed_at")
        if not isinstance(observed_at, str):
            observed_at = datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc).isoformat()
        try:
            return ObservationRecord(
                kind="observation", path=rel, sha256=digest, command=command, observed_at=observed_at
            ), None
        except ValidationError as e:
            return None, f"observation {rel!r} invalid: {e.errors()[0]['msg']}"

    lines = _split_lines(text)
    hint = raw.get("line")
    line: int | None = None
    if expect_value:
        if isinstance(hint, int) and not isinstance(hint, bool) and 1 <= hint <= len(lines) \
                and contains_token(lines[hint - 1], value):
            line = hint
        else:
            found = token_lines(text, value)
            if not found:
                return None, f"{value!r} does not occur as a token in {rel!r}"
            line = found[0]
    else:
        line = hint if isinstance(hint, int) and not isinstance(hint, bool) and 1 <= hint <= len(lines) else 1
    return FileRecord(kind="file", path=rel, line=line, sha256=digest), None


def anchor_boundary_update(
    raw_entries: object,
    project_root: str | Path,
    *,
    default_path: str | None = None,
) -> tuple[list, list[str]]:
    """Replace model-proposed records with code-computed ones.

    *default_path* is the primed document (project-relative): when set, any
    side without a usable citation is anchored against it, so identifiers
    extracted from a primed file carry a file record. Returns the rewritten
    raw entries (still to be validated by ``ProblemModel.apply_update``) and
    open questions for citations that did not check out.
    """
    root = Path(project_root)
    items = raw_entries if isinstance(raw_entries, list) else [raw_entries]
    out: list = []
    questions: list[str] = []
    for item in items:
        if isinstance(item, BoundaryEntry):
            item = item.model_dump(by_alias=True, mode="json")
        if not isinstance(item, dict) or not isinstance(item.get("sides"), list):
            out.append(item)  # let typed validation reject it
            continue
        entry = dict(item)
        sides = []
        for side in entry["sides"]:
            if not isinstance(side, dict):
                sides.append(side)
                continue
            side = dict(side)
            value = side.get("value")
            raw_records = side.get("records") or []
            if not isinstance(raw_records, list):
                raw_records = [raw_records]
            if not raw_records and default_path:
                raw_records = [{"kind": "file", "path": default_path}]
            anchored = []
            if isinstance(value, str) and value.strip():
                for raw in raw_records:
                    rec, reason = anchor_record(
                        raw, value, root,
                        default_path=default_path,
                        expect_value=side.get("role") != "denies",
                    )
                    if rec is not None:
                        anchored.append(rec.model_dump(mode="json"))
                    else:
                        questions.append(
                            f"Boundary {entry.get('id', '?')}: {side.get('party', '?')} "
                            f"({side.get('role', '?')}) value {json.dumps(value)} is UNVERIFIED — "
                            f"{reason}. Where exactly is it recorded?"
                        )
            side["records"] = anchored
            sides.append(side)
        entry["sides"] = sides
        out.append(entry)
    return out, questions


def unverified_questions(entries: Iterable[BoundaryEntry]) -> list[str]:
    """One deterministic open question per side that has no record."""
    qs = []
    for entry in entries:
        for side in entry.sides:
            if not side.records:
                qs.append(
                    f"Boundary {entry.id}: {side.party} ({side.role}) uses {json.dumps(side.value)} — "
                    "UNVERIFIED: no record cited. Which file line or captured observation shows it?"
                )
    return qs
