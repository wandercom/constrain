"""Core data models for Constrain sessions."""

from __future__ import annotations

import enum
import json
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class Posture(str, enum.Enum):
    adversarial = "adversarial"
    contrarian = "contrarian"
    critic = "critic"
    skeptic = "skeptic"
    collaborator = "collaborator"


class Phase(str, enum.Enum):
    understand = "understand"
    challenge = "challenge"
    synthesize = "synthesize"
    complete = "complete"


class Severity(str, enum.Enum):
    must = "must"
    should = "should"
    may = "may"


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── boundary register (format "boundary-register/1") ────────────────
#
# The register records, for every identifier that crosses a boundary, the
# exact literal each side uses and where that literal is recorded. Field names
# are shared with the factory and Pact; do not rename them.

BOUNDARY_REGISTER_SCHEMA = "boundary-register/1"

BoundaryKind = Literal[
    "name", "endpoint", "credential", "audience", "principal", "grant", "role",
    "table", "column", "field", "metric", "label", "env", "config-key", "topic",
    "queue", "bucket", "secret", "service", "header", "claim", "enum", "other",
]
BoundaryRole = Literal["defines", "grants", "consumes", "presents", "denies"]
BoundaryRelation = Literal["equal", "absent"]

_ENTRY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$"
)


def check_record_path(path: str) -> str:
    """Return *path* if it is a safe project-relative path, else raise ValueError.

    Records are evidence inside the project: no absolute paths, no drive
    letters, no ``..`` segments, no empty path.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("record path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("record path contains a NUL byte")
    if path.startswith(("/", "\\")) or PureWindowsPath(path).drive:
        raise ValueError(f"record path must be relative to the project root: {path!r}")
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if ".." in parts:
        raise ValueError(f"record path must not contain '..': {path!r}")
    return path


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileRecord(_Strict):
    """A line in a project file that states the value. Digest is of the whole file."""

    kind: Literal["file"]
    path: str
    line: int = Field(ge=1)
    sha256: str

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        return check_record_path(v)

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ValueError("sha256 must be 64 hex characters")
        return v.lower()


class ObservationRecord(_Strict):
    """Captured output of a live query or command, stored as a project file."""

    kind: Literal["observation"]
    path: str
    sha256: str
    command: str = Field(min_length=1)
    observed_at: str

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        return check_record_path(v)

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ValueError("sha256 must be 64 hex characters")
        return v.lower()

    @field_validator("command")
    @classmethod
    def _command(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("command must be non-empty")
        return v

    @field_validator("observed_at")
    @classmethod
    def _rfc3339(cls, v: str) -> str:
        if not _RFC3339_RE.match(v):
            raise ValueError(f"observed_at must be RFC3339, got {v!r}")
        return v


BoundaryRecord = Annotated[Union[FileRecord, ObservationRecord], Field(discriminator="kind")]


class BoundarySide(_Strict):
    party: str = Field(min_length=1)
    role: BoundaryRole
    value: str = Field(min_length=1)
    records: list[BoundaryRecord] = Field(default_factory=list)

    @field_validator("party")
    @classmethod
    def _party(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("party must be non-empty")
        return v

    @field_validator("value")
    @classmethod
    def _value(cls, v: str) -> str:
        # The literal is kept byte-for-byte; whitespace-only is not a literal.
        if not v.strip():
            raise ValueError("value must be a non-empty literal")
        return v

    @property
    def verified(self) -> bool:
        return bool(self.records)


class BoundaryEntry(_Strict):
    id: str
    kind: BoundaryKind
    relation: BoundaryRelation
    description: str = Field(min_length=1)
    sides: list[BoundarySide] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ENTRY_ID_RE.match(v):
            raise ValueError(f"id must match [A-Za-z0-9][A-Za-z0-9._-]*, got {v!r}")
        return v

    @field_validator("description")
    @classmethod
    def _description(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("description must be non-empty")
        return v


class BoundaryRegister(_Strict):
    schema_: Literal["boundary-register/1"] = Field(
        default=BOUNDARY_REGISTER_SCHEMA, alias="schema"
    )
    entries: list[BoundaryEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _unique_ids(self) -> "BoundaryRegister":
        seen: set[str] = set()
        for e in self.entries:
            if e.id in seen:
                raise ValueError(f"duplicate boundary entry id {e.id!r}")
            seen.add(e.id)
        return self


def _merge_sides(old: BoundaryEntry, new: BoundaryEntry) -> BoundaryEntry:
    """Carry records forward for sides whose literal did not change.

    A model re-stating an entry without its records must not silently erase
    evidence already captured; a changed literal invalidates the old records
    (they attested a different value).
    """
    old_by_key = {(s.party, s.role): s for s in old.sides}
    sides = []
    for side in new.sides:
        prior = old_by_key.get((side.party, side.role))
        if prior is not None and not side.records and prior.value == side.value:
            side = side.model_copy(update={"records": list(prior.records)})
        sides.append(side)
    return new.model_copy(update={"sides": sides})


def _question_text(item: object) -> str:
    """Coerce an LLM-proposed open question to text (open_questions is list[str])."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for k in ("question", "text", "description"):
            if isinstance(item.get(k), str) and item[k].strip():
                return item[k].strip()
        return json.dumps(item, sort_keys=True, default=str)
    if item is None:
        return ""
    return str(item)


class ProblemModel(BaseModel):
    system_description: str = ""
    stakeholders: list[str] = Field(default_factory=list)
    failure_modes: list[dict] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    success_shape: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    boundary_register: list[BoundaryEntry] = Field(default_factory=list)

    def apply_update(self, update: dict) -> None:
        """Merge an LLM-proposed update in place.

        ``boundary_register`` items are validated as typed ``BoundaryEntry``
        objects and upserted by id. Malformed items are never appended raw:
        each rejection is recorded in ``open_questions`` (prefixed
        "Rejected boundary entry") so the interview can ask for it again.
        """
        fields = type(self).model_fields
        for key, value in update.items():
            if key not in fields:
                continue
            if key == "boundary_register":
                for reason in self._apply_boundary_update(value):
                    if reason not in self.open_questions:
                        self.open_questions.append(reason)
                continue
            if key == "open_questions":
                for item in value if isinstance(value, list) else [value]:
                    q = _question_text(item)
                    if q and q not in self.open_questions:
                        self.open_questions.append(q)
                continue
            current = getattr(self, key)
            if isinstance(current, str):
                setattr(self, key, value)
            elif isinstance(current, list):
                items = value if isinstance(value, list) else [value]
                for item in items:
                    if isinstance(item, str) and item not in current:
                        current.append(item)
                    elif isinstance(item, dict):
                        current.append(item)

    def _apply_boundary_update(self, value: object) -> list[str]:
        rejected: list[str] = []
        items = value if isinstance(value, list) else [value]
        for item in items:
            label = item.get("id", "?") if isinstance(item, dict) else "?"
            if isinstance(item, BoundaryEntry):
                entry = item
            else:
                try:
                    entry = BoundaryEntry.model_validate(item)
                except ValidationError as e:
                    reasons = "; ".join(
                        f"{'.'.join(str(p) for p in err['loc']) or '<entry>'}: {err['msg']}"
                        for err in e.errors()
                    )
                    rejected.append(f"Rejected boundary entry {label!r}: {reasons}")
                    continue
            for i, existing in enumerate(self.boundary_register):
                if existing.id == entry.id:
                    self.boundary_register[i] = _merge_sides(existing, entry)
                    break
            else:
                self.boundary_register.append(entry)
        return rejected


class Constraint(BaseModel):
    id: str
    boundary: str
    condition: str
    violation: str
    severity: Severity
    rationale: str
    classification_tier: str | None = None
    affected_components: list[str] = Field(default_factory=list)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: int = 1
    posture: Posture
    phase: Phase = Phase.understand
    round: int = 0
    understand_rounds: int = 0
    challenge_rounds: int = 0
    conversation: list[Message] = Field(default_factory=list)
    problem_model: ProblemModel = Field(default_factory=ProblemModel)
    prompt_md: str = ""
    constraints_yaml: str = ""
    trust_policy_yaml: str = ""
    component_map_yaml: str = ""
    schema_hints_yaml: str = ""
    boundary_register_yaml: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
