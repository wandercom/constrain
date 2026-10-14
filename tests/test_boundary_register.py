"""Tests for the boundary register (boundary-register/1).

Every identifier that crosses a boundary must be recorded with the exact
literal each side uses and a record (file line or captured observation)
pinned by sha256. Verification is mechanical: no LLM, errors not warnings.
"""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from constrain.boundary import (
    BOUNDARY_REGISTER_FILE,
    anchor_boundary_update,
    anchor_record,
    blocking_findings,
    contains_token,
    is_placeholder,
    load_boundary_register_text,
    render_boundary_register,
    validate_boundary_register,
)
from constrain.engine import ConversationEngine, EngineConfig
from constrain.models import (
    BoundaryEntry,
    BoundaryRegister,
    FileRecord,
    Phase,
    Posture,
    ProblemModel,
    Session,
)
from constrain.posture import _JSON_INSTRUCTION, get_system_prompt
from constrain.synthesizer import write_artifacts


# ============================================================================
# Helpers
# ============================================================================


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _file_record(root: Path, rel: str, line: int) -> dict:
    return {"kind": "file", "path": rel, "line": line, "sha256": _sha(root / rel)}


def _obs_record(root: Path, rel: str, command: str) -> dict:
    return {
        "kind": "observation",
        "path": rel,
        "sha256": _sha(root / rel),
        "command": command,
        "observed_at": "2026-10-14T12:00:00Z",
    }


def _side(party, role, value, records=None):
    return {"party": party, "role": role, "value": value, "records": records or []}


def _entry(id_, sides, relation="equal", kind="role", description="sides must agree"):
    return {"id": id_, "kind": kind, "relation": relation, "description": description, "sides": sides}


def _register(*entries):
    return {"schema": "boundary-register/1", "entries": list(entries)}


def _codes(findings):
    return [f.code for f in findings if f.severity == "error"]


@pytest.fixture
def grant_project(tmp_path):
    """DB grants role gateway_ro; gateway config connects as gateway_ro."""
    _write(tmp_path, "db/grants.sql", "-- grants\nGRANT SELECT ON bookings TO gateway_ro;\n")
    _write(tmp_path, "gateway/config.yaml", "db:\n  role: gateway_ro\n  host: db\n")
    return tmp_path


def _grant_entry(root, consumer_value="gateway_ro", consumer_line=2):
    return _entry("gateway-db-role", [
        _side("db", "grants", "gateway_ro", [_file_record(root, "db/grants.sql", 2)]),
        _side("gateway", "consumes", consumer_value, [_file_record(root, "gateway/config.yaml", consumer_line)]),
    ])


# ============================================================================
# BR-001: token matching
# ============================================================================


class TestBR001TokenMatch:
    def test_prefix_does_not_match_longer_identifier(self):
        assert not contains_token("audience: mea-auth-runtime", "mea-auth")

    def test_suffix_does_not_match(self):
        assert not contains_token("role: xgateway_ro", "gateway_ro")

    def test_quoted_key_matches(self):
        assert contains_token('{"level": "INFO"}', "level")

    def test_bof_and_eof(self):
        assert contains_token("mea-auth", "mea-auth")

    def test_later_occurrence_found_after_rejected_one(self):
        assert contains_token("mea-auth-runtime mea-auth", "mea-auth")

    def test_value_starting_with_punctuation(self):
        assert contains_token("GET x/v1/health", "/v1/health")

    def test_case_sensitive(self):
        assert not contains_token("Gateway_RO", "gateway_ro")


# ============================================================================
# BR-002: relation equal passes when every side is recorded and agrees
# ============================================================================


class TestBR002EqualPass:
    def test_verified_equal_entry_has_no_errors(self, grant_project):
        findings = validate_boundary_register(_register(_grant_entry(grant_project)), grant_project)
        assert _codes(findings) == []

    def test_observation_record_passes(self, tmp_path):
        _write(tmp_path, "obs/metrics.txt", "resource.type=cloud_run_revision\n")
        _write(tmp_path, "alerts/uptime.yaml", 'filter: resource.type="cloud_run_revision"\n')
        entry = _entry("uptime-resource-type", [
            _side("external:cloud-monitoring", "defines", "cloud_run_revision",
                  [_obs_record(tmp_path, "obs/metrics.txt", "gcloud monitoring time-series list")]),
            _side("alerting", "consumes", "cloud_run_revision",
                  [_file_record(tmp_path, "alerts/uptime.yaml", 1)]),
        ], kind="metric")
        assert _codes(validate_boundary_register(_register(entry), tmp_path)) == []

    def test_typed_register_accepted(self, grant_project):
        reg = BoundaryRegister.model_validate(_register(_grant_entry(grant_project)))
        assert _codes(validate_boundary_register(reg, grant_project)) == []


# ============================================================================
# BR-003: mismatch and near-miss are errors
# ============================================================================


class TestBR003Mismatch:
    def test_different_values_are_mismatch(self, tmp_path):
        _write(tmp_path, "checks.txt", "resource.type=cloud_run_revision\n")
        _write(tmp_path, "alert.yaml", 'filter: resource.type="uptime_url"\n')
        entry = _entry("uptime-type", [
            _side("checks", "defines", "cloud_run_revision", [_file_record(tmp_path, "checks.txt", 1)]),
            _side("alert", "consumes", "uptime_url", [_file_record(tmp_path, "alert.yaml", 1)]),
        ], kind="metric")
        findings = validate_boundary_register(_register(entry), tmp_path)
        mism = [f for f in findings if f.code == "mismatch"]
        assert len(mism) == 1
        assert mism[0].severity == "error"
        assert mism[0].entry_id == "uptime-type"
        assert "alert/consumes" in mism[0].side
        assert mism[0].blocking

    def test_near_miss_separator(self, tmp_path):
        _write(tmp_path, "a.txt", "gateway_ro\n")
        _write(tmp_path, "b.txt", "gateway-ro\n")
        entry = _entry("r", [
            _side("db", "grants", "gateway_ro", [_file_record(tmp_path, "a.txt", 1)]),
            _side("gw", "consumes", "gateway-ro", [_file_record(tmp_path, "b.txt", 1)]),
        ])
        findings = validate_boundary_register(_register(entry), tmp_path)
        near = [f for f in findings if f.code == "near-miss"]
        assert near and "near-miss" in near[0].message

    def test_near_miss_case(self, tmp_path):
        _write(tmp_path, "a.txt", "severity\n")
        _write(tmp_path, "b.txt", "SEVERITY\n")
        entry = _entry("log-key", [
            _side("platform", "defines", "severity", [_file_record(tmp_path, "a.txt", 1)]),
            _side("app", "presents", "SEVERITY", [_file_record(tmp_path, "b.txt", 1)]),
        ], kind="field")
        assert "near-miss" in _codes(validate_boundary_register(_register(entry), tmp_path))

    def test_level_vs_severity_is_mismatch_not_near_miss(self, tmp_path):
        _write(tmp_path, "a.txt", "severity\n")
        _write(tmp_path, "b.txt", '{"level": "INFO"}\n')
        entry = _entry("log-key", [
            _side("platform", "defines", "severity", [_file_record(tmp_path, "a.txt", 1)]),
            _side("app", "presents", "level", [_file_record(tmp_path, "b.txt", 1)]),
        ], kind="field")
        codes = _codes(validate_boundary_register(_register(entry), tmp_path))
        assert codes == ["mismatch"]


# ============================================================================
# BR-004: missing records = UNVERIFIED error (written but flagged)
# ============================================================================


class TestBR004Unverified:
    def test_side_without_records_is_error(self, grant_project):
        entry = _grant_entry(grant_project)
        entry["sides"][1]["records"] = []
        findings = validate_boundary_register(_register(entry), grant_project)
        unv = [f for f in findings if f.code == "unverified"]
        assert len(unv) == 1
        assert unv[0].severity == "error"
        assert "UNVERIFIED" in unv[0].message

    def test_unverified_does_not_block_synthesis(self, grant_project):
        entry = _grant_entry(grant_project)
        entry["sides"][1]["records"] = []
        findings = validate_boundary_register(_register(entry), grant_project)
        assert blocking_findings(findings) == []


# ============================================================================
# BR-005: stale digest, missing file, line lacks value
# ============================================================================


class TestBR005Records:
    def test_stale_digest_names_file(self, grant_project):
        entry = _grant_entry(grant_project)
        (grant_project / "gateway/config.yaml").write_text("db:\n  role: gateway_ro\n  host: db2\n")
        findings = validate_boundary_register(_register(entry), grant_project)
        stale = [f for f in findings if f.code == "stale"]
        assert len(stale) == 1
        assert "gateway/config.yaml" in stale[0].message
        assert stale[0].severity == "error"

    def test_missing_record_file(self, grant_project):
        entry = _grant_entry(grant_project)
        (grant_project / "db/grants.sql").unlink()
        assert "missing-record" in _codes(validate_boundary_register(_register(entry), grant_project))

    def test_cited_line_lacks_value(self, grant_project):
        entry = _grant_entry(grant_project, consumer_line=3)  # line 3 is "host: db"
        findings = validate_boundary_register(_register(entry), grant_project)
        lacks = [f for f in findings if f.code == "line-lacks-value"]
        assert len(lacks) == 1
        assert "gateway/config.yaml:3" in lacks[0].message
        assert "line(s) [2]" in lacks[0].message

    def test_line_past_end(self, grant_project):
        entry = _grant_entry(grant_project, consumer_line=99)
        assert "line-out-of-range" in _codes(validate_boundary_register(_register(entry), grant_project))

    def test_token_boundary_on_cited_line(self, tmp_path):
        _write(tmp_path, "idp.yaml", "audience: mea-auth\n")
        _write(tmp_path, "client.yaml", "audience: mea-auth-runtime\n")
        entry = _entry("aud", [
            _side("idp", "defines", "mea-auth", [_file_record(tmp_path, "idp.yaml", 1)]),
            _side("client", "presents", "mea-auth", [_file_record(tmp_path, "client.yaml", 1)]),
        ], kind="audience")
        findings = validate_boundary_register(_register(entry), tmp_path)
        assert _codes(findings) == ["line-lacks-value"]

    def test_observation_lacking_value(self, tmp_path):
        _write(tmp_path, "obs.txt", "resource.type=cloud_run_revision\n")
        _write(tmp_path, "alert.yaml", "uptime_url\n")
        entry = _entry("t", [
            _side("alert", "consumes", "uptime_url", [_file_record(tmp_path, "alert.yaml", 1)]),
            _side("checks", "defines", "uptime_url", [_obs_record(tmp_path, "obs.txt", "gcloud ...")]),
        ], kind="metric")
        assert "value-not-observed" in _codes(validate_boundary_register(_register(entry), tmp_path))


# ============================================================================
# BR-006: relation absent
# ============================================================================


@pytest.fixture
def isolation_project(tmp_path):
    _write(tmp_path, "sa.txt", "serviceAccount:web@proj.iam\n")
    _write(tmp_path, "obs/iam-policy.json",
           '{"bindings": [{"role": "roles/run.invoker", "members": ["serviceAccount:ledger@proj.iam"]}]}\n')
    return tmp_path


def _absent_entry(root):
    return _entry("ledger-isolation", [
        _side("web", "presents", "serviceAccount:web@proj.iam", [_file_record(root, "sa.txt", 1)]),
        _side("ledger", "denies", "roles/run.invoker",
              [_obs_record(root, "obs/iam-policy.json", "gcloud run services get-iam-policy ledger")]),
    ], relation="absent", kind="principal")


class TestBR006Absent:
    def test_absent_passes_when_denier_lacks_subject(self, isolation_project):
        findings = validate_boundary_register(_register(_absent_entry(isolation_project)), isolation_project)
        assert _codes(findings) == []

    def test_absent_fails_when_denier_contains_subject(self, isolation_project):
        p = isolation_project / "obs/iam-policy.json"
        p.write_text('{"members": ["serviceAccount:web@proj.iam"]}\n')
        entry = _absent_entry(isolation_project)  # digest computed after the write
        findings = validate_boundary_register(_register(entry), isolation_project)
        viol = [f for f in findings if f.code == "absence-violated"]
        assert len(viol) == 1 and "ledger/denies" in viol[0].side

    def test_absent_denier_record_must_verify_digest(self, isolation_project):
        entry = _absent_entry(isolation_project)
        (isolation_project / "obs/iam-policy.json").write_text("{}\n")
        assert "stale" in _codes(validate_boundary_register(_register(entry), isolation_project))

    def test_absent_needs_exactly_one_presents(self, isolation_project):
        entry = _absent_entry(isolation_project)
        entry["sides"].append(dict(entry["sides"][0], party="web2"))
        assert "relation-shape" in _codes(validate_boundary_register(_register(entry), isolation_project))

    def test_absent_needs_a_denier(self, isolation_project):
        entry = _absent_entry(isolation_project)
        entry["sides"] = entry["sides"][:1]
        assert "relation-shape" in _codes(validate_boundary_register(_register(entry), isolation_project))


class TestBR006EqualShape:
    def test_equal_needs_definer(self, grant_project):
        entry = _grant_entry(grant_project)
        entry["sides"][0]["role"] = "consumes"
        assert "relation-shape" in _codes(validate_boundary_register(_register(entry), grant_project))

    def test_equal_needs_user(self, grant_project):
        entry = _grant_entry(grant_project)
        entry["sides"][1]["role"] = "defines"
        assert "relation-shape" in _codes(validate_boundary_register(_register(entry), grant_project))


# ============================================================================
# BR-007: placeholders
# ============================================================================


class TestBR007Placeholder:
    @pytest.mark.parametrize("value", ["TODO", "TBD", "<project-id>", "projects/<project>/topics/x",
                                       "???", "unknown", "assumed", "TODO: ask infra", "Assumed"])
    def test_placeholder_values(self, value):
        assert is_placeholder(value)

    @pytest.mark.parametrize("value", ["gateway_ro", "unknown_count", "cloud_run_revision", "Name <a@b.com>"])
    def test_real_values(self, value):
        assert not is_placeholder(value)

    def test_placeholder_is_error(self, tmp_path):
        _write(tmp_path, "a.txt", "TODO\n")
        entry = _entry("x", [
            _side("db", "grants", "TODO", [_file_record(tmp_path, "a.txt", 1)]),
            _side("gw", "consumes", "TODO", [_file_record(tmp_path, "a.txt", 1)]),
        ])
        findings = validate_boundary_register(_register(entry), tmp_path)
        assert _codes(findings).count("placeholder") == 2
        assert blocking_findings(findings)


# ============================================================================
# BR-008: path safety
# ============================================================================


class TestBR008PathSafety:
    @pytest.mark.parametrize("path", ["../secrets.txt", "a/../../b", "/etc/passwd", "C:\\x.txt", ""])
    def test_model_rejects_unsafe_paths(self, path):
        with pytest.raises(Exception):
            FileRecord(kind="file", path=path, line=1, sha256="a" * 64)

    def test_validator_reports_traversal(self, grant_project):
        entry = _grant_entry(grant_project)
        entry["sides"][0]["records"][0]["path"] = "../outside.sql"
        findings = validate_boundary_register(_register(entry), grant_project)
        path_errs = [f for f in findings if f.code == "path"]
        assert path_errs and path_errs[0].entry_id == "gateway-db-role"

    def test_symlink_escaping_root_rejected(self, tmp_path):
        outside = tmp_path / "outside"
        root = tmp_path / "proj"
        root.mkdir()
        _write(outside, "grants.sql", "gateway_ro\n")
        os.symlink(outside / "grants.sql", root / "link.sql")
        entry = _entry("r", [
            _side("db", "grants", "gateway_ro", [_file_record(root, "link.sql", 1)]),
            _side("gw", "consumes", "gateway_ro", [_file_record(root, "link.sql", 1)]),
        ])
        assert "path" in _codes(validate_boundary_register(_register(entry), root))


# ============================================================================
# BR-009: register schema and serialization
# ============================================================================


class TestBR009Schema:
    def test_wrong_schema_is_error(self, grant_project):
        reg = _register(_grant_entry(grant_project))
        reg["schema"] = "boundary-register/2"
        assert "schema" in _codes(validate_boundary_register(reg, grant_project))

    def test_duplicate_ids(self, grant_project):
        e = _grant_entry(grant_project)
        assert "duplicate-id" in _codes(validate_boundary_register(_register(e, e), grant_project))

    def test_malformed_entry_reported_others_still_checked(self, grant_project):
        bad = {"id": "bad", "kind": "nonsense", "relation": "equal", "description": "x", "sides": []}
        good = _grant_entry(grant_project, consumer_line=3)
        codes = _codes(validate_boundary_register(_register(bad, good), grant_project))
        assert "invalid" in codes and "line-lacks-value" in codes

    def test_render_round_trip_preserves_literals(self, grant_project):
        entry = _grant_entry(grant_project)
        entry["sides"][0]["value"] = "a: b # not a comment"
        entry["sides"][1]["value"] = "yes"
        typed = [BoundaryEntry.model_validate(entry)]
        text = render_boundary_register(typed)
        data = yaml.safe_load(text)
        assert data["schema"] == "boundary-register/1"
        assert data["entries"][0]["sides"][0]["value"] == "a: b # not a comment"
        assert data["entries"][0]["sides"][1]["value"] == "yes"
        assert set(data["entries"][0]) == {"id", "kind", "relation", "description", "sides"}
        assert set(data["entries"][0]["sides"][0]["records"][0]) == {"kind", "path", "line", "sha256"}

    def test_write_artifacts_writes_register_verbatim(self, tmp_path, grant_project):
        entry = _grant_entry(grant_project)
        entry["sides"][0]["value"] = "Free tier: 100/day"
        text = render_boundary_register([BoundaryEntry.model_validate(entry)])
        written = write_artifacts("# p", "constraints: []", tmp_path, boundary_register_yaml=text)
        assert tmp_path / BOUNDARY_REGISTER_FILE in written
        assert (tmp_path / BOUNDARY_REGISTER_FILE).read_text() == text

    def test_empty_register_is_warning_only(self, tmp_path):
        findings = validate_boundary_register(_register(), tmp_path)
        assert _codes(findings) == [] and findings[0].severity == "warning"


# ============================================================================
# BR-010: typed apply_update
# ============================================================================


class TestBR010ApplyUpdate:
    def _valid(self, value="gateway_ro"):
        return _entry("gw-role", [_side("db", "grants", value), _side("gw", "consumes", value)])

    @pytest.mark.parametrize("bad", [
        {"id": "x"},  # missing fields
        _entry("-bad-id", [_side("a", "defines", "v")]),
        _entry("x", [_side("a", "defines", "v")], kind="nonsense"),
        _entry("x", [_side("a", "owns", "v")]),
        _entry("x", [_side("a", "defines", "")]),
        _entry("x", [_side("a", "defines", "v", [{"kind": "file", "path": "a", "line": 1, "sha256": "zz"}])]),
        _entry("x", [_side("a", "defines", "v")], relation="maybe"),
        dict(_entry("x", [_side("a", "defines", "v")]), extra_field=1),
        "not a dict",
    ])
    def test_malformed_entries_rejected(self, bad):
        pm = ProblemModel()
        result = pm.apply_update({"boundary_register": [bad]})
        assert result is None
        assert pm.boundary_register == []
        assert any(q.startswith("Rejected boundary entry") for q in pm.open_questions)

    def test_valid_entry_is_typed(self):
        pm = ProblemModel()
        pm.apply_update({"boundary_register": [self._valid()]})
        assert isinstance(pm.boundary_register[0], BoundaryEntry)
        assert pm.open_questions == []

    def test_upsert_by_id_keeps_records_when_value_unchanged(self):
        pm = ProblemModel()
        rec = {"kind": "file", "path": "a.sql", "line": 1, "sha256": "a" * 64}
        first = self._valid()
        first["sides"][0]["records"] = [rec]
        pm.apply_update({"boundary_register": [first]})
        pm.apply_update({"boundary_register": [self._valid()]})
        assert len(pm.boundary_register) == 1
        assert pm.boundary_register[0].sides[0].records[0].path == "a.sql"

    def test_upsert_drops_records_when_value_changes(self):
        pm = ProblemModel()
        rec = {"kind": "file", "path": "a.sql", "line": 1, "sha256": "a" * 64}
        first = self._valid()
        first["sides"][0]["records"] = [rec]
        pm.apply_update({"boundary_register": [first]})
        pm.apply_update({"boundary_register": [self._valid("gateway_rw")]})
        assert pm.boundary_register[0].sides[0].records == []

    def test_session_round_trips_register(self):
        s = Session(posture=Posture.critic)
        s.problem_model.apply_update({"boundary_register": [self._valid()]})
        s2 = Session.model_validate_json(s.model_dump_json())
        assert s2.problem_model.boundary_register == s.problem_model.boundary_register


# ============================================================================
# BR-011: anchoring — code computes line and digest, never the model
# ============================================================================


class TestBR011Anchoring:
    def test_model_sha_ignored_and_line_corrected(self, grant_project):
        rec, reason = anchor_record(
            {"kind": "file", "path": "gateway/config.yaml", "line": 3, "sha256": "f" * 64},
            "gateway_ro", grant_project,
        )
        assert reason is None
        assert rec.sha256 == _sha(grant_project / "gateway/config.yaml")
        assert rec.line == 2

    def test_value_absent_from_file_is_not_anchored(self, grant_project):
        rec, reason = anchor_record({"kind": "file", "path": "gateway/config.yaml"}, "gateway_rw", grant_project)
        assert rec is None and "does not occur" in reason

    def test_missing_file_not_anchored(self, grant_project):
        rec, reason = anchor_record({"kind": "file", "path": "nope.txt"}, "x", grant_project)
        assert rec is None and "does not exist" in reason

    def test_absolute_path_inside_root_made_relative(self, grant_project):
        rec, _ = anchor_record({"kind": "file", "path": str(grant_project / "db/grants.sql")},
                               "gateway_ro", grant_project)
        assert rec.path == "db/grants.sql"

    def test_observation_requires_command(self, grant_project):
        rec, reason = anchor_record({"kind": "observation", "path": "db/grants.sql"}, "gateway_ro", grant_project)
        assert rec is None and "command" in reason

    def test_failed_citation_leaves_side_unverified_with_question(self, grant_project):
        raw = [_entry("r", [_side("gw", "consumes", "gateway_rw", [{"kind": "file", "path": "gateway/config.yaml", "line": 2}])])]
        out, questions = anchor_boundary_update(raw, grant_project)
        assert out[0]["sides"][0]["records"] == []
        assert questions and "UNVERIFIED" in questions[0]


class _Backend:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def complete(self, system, messages):
        self.calls.append((system, [dict(m) for m in messages]))
        if self.responses:
            return self.responses.pop(0)
        return (
            "--- PROMPT ---\n# Generated prompt\n"
            "--- CONSTRAINTS ---\nconstraints: []\n"
            "--- TRUST_POLICY ---\ntrust: {}\n"
            "--- COMPONENT_MAP ---\ncomponents: []\n"
            "--- SCHEMA_HINTS ---\nfield_hints: []\n"
        )


class _IO:
    def __init__(self):
        self.messages = []

    def display(self, text):
        self.messages.append(text)

    def prompt(self, prefix):
        raise EOFError


class _SessionManager:
    def __init__(self):
        self.saved = 0

    def transition_phase(self, session, phase):
        session.phase = phase

    def save(self, session):
        self.saved += 1


def _json_block(update):
    return "Summary.\n\n```json\n" + json.dumps({"problem_model_update": update}) + "\n```"


class TestBR012Priming:
    def test_primed_identifiers_carry_code_computed_file_record(self, tmp_path):
        doc = _write(tmp_path, "docs/design.md",
                     "# Design\n\nThe gateway connects as role `gateway_ro`.\nLogs use key severity.\n")
        update = {"boundary_register": [_entry("gw-role", [
            # Model claims line 1 and a bogus sha — code must fix both.
            _side("gateway", "consumes", "gateway_ro",
                  [{"kind": "file", "line": 1, "sha256": "0" * 64}]),
        ])]}
        session = Session(posture=Posture.critic)
        engine = ConversationEngine(session, _SessionManager(), backend=_Backend([_json_block(update)]),
                                    io=_IO(), project_root=tmp_path)
        engine.prime_with_document(doc)
        rec = session.problem_model.boundary_register[0].sides[0].records[0]
        assert rec.path == "docs/design.md"
        assert rec.line == 3
        assert rec.sha256 == _sha(doc)

    def test_side_without_hint_anchored_to_primed_doc(self, tmp_path):
        doc = _write(tmp_path, "spec.md", "alert filter uses uptime_url\n")
        update = {"boundary_register": [_entry("t", [_side("alert", "consumes", "uptime_url")], kind="metric")]}
        session = Session(posture=Posture.critic)
        engine = ConversationEngine(session, _SessionManager(), backend=_Backend([_json_block(update)]),
                                    io=_IO(), project_root=tmp_path)
        engine.prime_with_document(doc)
        rec = session.problem_model.boundary_register[0].sides[0].records[0]
        assert (rec.path, rec.line) == ("spec.md", 1)

    def test_invented_value_stays_unverified(self, tmp_path):
        doc = _write(tmp_path, "spec.md", "the gateway reads bookings\n")
        update = {"boundary_register": [_entry("gw-role", [_side("gateway", "consumes", "gateway_ro")])]}
        session = Session(posture=Posture.critic)
        engine = ConversationEngine(session, _SessionManager(), backend=_Backend([_json_block(update)]),
                                    io=_IO(), project_root=tmp_path)
        engine.prime_with_document(doc)
        side = session.problem_model.boundary_register[0].sides[0]
        assert side.records == []
        assert any("gw-role" in q and "UNVERIFIED" in q for q in session.problem_model.open_questions)

    def test_interview_citation_anchored(self, grant_project):
        update = {"boundary_register": [_entry("gw-role", [
            _side("db", "grants", "gateway_ro", [{"kind": "file", "path": "db/grants.sql", "line": 2}]),
        ])]}
        session = Session(posture=Posture.critic, phase=Phase.challenge)
        engine = ConversationEngine(session, _SessionManager(), backend=_Backend(), io=_IO(),
                                    project_root=grant_project)
        engine._apply_model_update(update)
        rec = session.problem_model.boundary_register[0].sides[0].records[0]
        assert rec.sha256 == _sha(grant_project / "db/grants.sql")


# ============================================================================
# BR-013: synthesis — register verbatim, transcript passed, blocking gate
# ============================================================================


def _synth_session(root, entry):
    session = Session(posture=Posture.critic, phase=Phase.synthesize)
    session.problem_model.apply_update({"boundary_register": [entry]})
    session.conversation = []
    return session


class TestBR013Synthesis:
    def test_register_written_by_code_and_complete(self, grant_project):
        session = _synth_session(grant_project, _grant_entry(grant_project))
        backend = _Backend()
        engine = ConversationEngine(session, _SessionManager(), backend=backend, io=_IO(),
                                    project_root=grant_project)
        engine._run_synthesis()
        assert session.phase is Phase.complete
        data = load_boundary_register_text(session.boundary_register_yaml)
        assert data["entries"][0]["sides"][1]["value"] == "gateway_ro"
        assert _codes(validate_boundary_register(data, grant_project)) == []

    def test_synthesis_prompt_contains_register_verbatim(self, grant_project):
        session = _synth_session(grant_project, _grant_entry(grant_project))
        backend = _Backend()
        engine = ConversationEngine(session, _SessionManager(), backend=backend, io=_IO(),
                                    project_root=grant_project)
        engine._run_synthesis()
        system, _ = backend.calls[0]
        assert render_boundary_register(session.problem_model.boundary_register) in system
        assert "do NOT rewrite it" in system

    def test_transcript_passed_to_synthesis(self, grant_project):
        session = _synth_session(grant_project, _grant_entry(grant_project))
        session.conversation = []
        engine = ConversationEngine(session, _SessionManager(), backend=_Backend(), io=_IO(),
                                    project_root=grant_project)
        engine._add_message("user", "The alert filters on resource.type=uptime_url_exact_literal")
        engine._run_synthesis()
        _, messages = engine.backend.calls[0]
        assert "uptime_url_exact_literal" in messages[0]["content"]

    def test_mismatch_blocks_completion(self, grant_project):
        _write(grant_project, "gateway/config.yaml", "db:\n  role: gateway\n")
        entry = _entry("gateway-db-role", [
            _side("db", "grants", "gateway_ro", [_file_record(grant_project, "db/grants.sql", 2)]),
            _side("gateway", "consumes", "gateway", [_file_record(grant_project, "gateway/config.yaml", 2)]),
        ])
        session = _synth_session(grant_project, entry)
        backend = _Backend(["I need the exact role. Which is right?"])
        io = _IO()
        engine = ConversationEngine(session, _SessionManager(), backend=backend, io=io,
                                    config=EngineConfig(challenge_max=2), project_root=grant_project)
        engine._run_synthesis()
        assert session.phase is Phase.synthesize
        assert session.prompt_md == ""
        assert session.boundary_register_yaml == ""
        # Only the resolution turn ran — no synthesis call.
        assert all("--- PROMPT ---" not in s for s, _ in backend.calls)
        assert any("Synthesis blocked" in m for m in io.messages)
        assert any("mismatch" in m for m in io.messages)

    def test_resolution_turn_can_fix_and_continue(self, grant_project):
        _write(grant_project, "gateway/config.yaml", "db:\n  role: gateway_ro\n")
        entry = _entry("gateway-db-role", [
            _side("db", "grants", "gateway_ro", [_file_record(grant_project, "db/grants.sql", 2)]),
            _side("gateway", "consumes", "gateway"),  # wrong literal, no record
        ])
        session = _synth_session(grant_project, entry)
        fix = _json_block({"boundary_register": [_entry("gateway-db-role", [
            _side("db", "grants", "gateway_ro", []),
            _side("gateway", "consumes", "gateway_ro", [{"kind": "file", "path": "gateway/config.yaml", "line": 2}]),
        ])]})
        backend = _Backend([fix])
        engine = ConversationEngine(session, _SessionManager(), backend=backend, io=_IO(),
                                    project_root=grant_project)
        engine._run_synthesis()
        assert session.phase is Phase.complete
        assert _codes(validate_boundary_register(
            load_boundary_register_text(session.boundary_register_yaml), grant_project)) == []

    def test_unverified_written_and_flagged(self, grant_project):
        entry = _grant_entry(grant_project)
        entry["sides"][1]["records"] = []
        session = _synth_session(grant_project, entry)
        io = _IO()
        engine = ConversationEngine(session, _SessionManager(), backend=_Backend(), io=io,
                                    project_root=grant_project)
        engine._run_synthesis()
        assert session.phase is Phase.complete
        data = load_boundary_register_text(session.boundary_register_yaml)
        assert data["entries"][0]["sides"][1]["records"] == []
        assert any("UNVERIFIED" in m for m in io.messages)


# ============================================================================
# BR-014: prompts
# ============================================================================


class TestBR014Prompts:
    def test_challenge_prompt_has_boundary_probe(self):
        prompt = get_system_prompt(Phase.challenge, ProblemModel(), Posture.critic)
        assert "boundary probe" in prompt
        assert "what exact literal does each side use, and where is it recorded?" in prompt.lower()
        assert '"records": [] (UNVERIFIED)' in prompt

    def test_json_instruction_documents_register(self):
        assert "boundary_register" in _JSON_INSTRUCTION
        assert "NEVER invent a value" in _JSON_INSTRUCTION
        assert "NEVER supply sha256" in _JSON_INSTRUCTION

    def test_problem_model_context_shows_literals(self, grant_project):
        pm = ProblemModel()
        pm.apply_update({"boundary_register": [_grant_entry(grant_project)]})
        prompt = get_system_prompt(Phase.understand, pm)
        assert '"gateway_ro"' in prompt and "db/grants.sql:2" in prompt


# ============================================================================
# BR-015: CLI — validate, boundary verify, export, logger
# ============================================================================


def _cli_session(register_yaml):
    session = Session(posture=Posture.critic, phase=Phase.complete)
    session.prompt_md = "# S\n\n## Trust and Authority Model\nx\n\n## Component Topology\ny"
    session.constraints_yaml = "constraints: []"
    session.boundary_register_yaml = register_yaml
    mgr = Mock()
    mgr.list_all.return_value = [{"id": session.id, "is_complete": True}]
    mgr.load.return_value = session
    return session, mgr


class TestBR015CLI:
    def _run_in(self, root, command, args, mgr=None):
        runner = CliRunner()
        cwd = os.getcwd()
        os.chdir(root)
        try:
            if mgr is not None:
                with patch("constrain.cli.SessionManager", return_value=mgr):
                    return runner.invoke(command, args)
            return runner.invoke(command, args)
        finally:
            os.chdir(cwd)

    def test_validate_passes_with_verified_register(self, grant_project):
        from constrain.cli import cmd_validate
        text = yaml.safe_dump(_register(_grant_entry(grant_project)))
        _, mgr = _cli_session(text)
        result = self._run_in(grant_project, cmd_validate, [], mgr)
        assert result.exit_code == 0, result.output
        assert "1 entry verified" in result.output

    def test_validate_exits_nonzero_on_mismatch(self, grant_project):
        from constrain.cli import cmd_validate
        _write(grant_project, "gateway/config.yaml", "db:\n  role: gateway\n")
        entry = _grant_entry(grant_project, consumer_value="gateway")
        _, mgr = _cli_session(yaml.safe_dump(_register(entry)))
        result = self._run_in(grant_project, cmd_validate, [], mgr)
        assert result.exit_code != 0
        assert "gateway-db-role" in result.output
        assert "gateway/consumes" in result.output
        assert "mismatch" in result.output

    def test_validate_exits_nonzero_on_unverified(self, grant_project):
        from constrain.cli import cmd_validate
        entry = _grant_entry(grant_project)
        entry["sides"][0]["records"] = []
        _, mgr = _cli_session(yaml.safe_dump(_register(entry)))
        result = self._run_in(grant_project, cmd_validate, [], mgr)
        assert result.exit_code != 0
        assert "UNVERIFIED" in result.output

    def test_validate_exits_nonzero_on_stale(self, grant_project):
        from constrain.cli import cmd_validate
        text = yaml.safe_dump(_register(_grant_entry(grant_project)))
        (grant_project / "db/grants.sql").write_text("GRANT SELECT ON bookings TO gateway_ro; -- edited\n")
        _, mgr = _cli_session(text)
        result = self._run_in(grant_project, cmd_validate, [], mgr)
        assert result.exit_code != 0
        assert "stale" in result.output and "db/grants.sql" in result.output

    def test_boundary_verify_pass_and_fail(self, grant_project):
        from constrain.cli import cli
        (grant_project / BOUNDARY_REGISTER_FILE).write_text(yaml.safe_dump(_register(_grant_entry(grant_project))))
        ok = self._run_in(grant_project, cli, ["boundary", "verify"])
        assert ok.exit_code == 0, ok.output

        (grant_project / BOUNDARY_REGISTER_FILE).write_text(
            yaml.safe_dump(_register(_grant_entry(grant_project, consumer_line=3))))
        bad = self._run_in(grant_project, cli, ["boundary", "verify"])
        assert bad.exit_code != 0
        assert "line-lacks-value" in bad.output

    def test_boundary_verify_root_option(self, grant_project, tmp_path_factory):
        from constrain.cli import cli
        elsewhere = tmp_path_factory.mktemp("ci")
        reg = elsewhere / "reg.yaml"
        reg.write_text(yaml.safe_dump(_register(_grant_entry(grant_project))))
        result = CliRunner().invoke(cli, ["boundary", "verify", str(reg), "--root", str(grant_project)])
        assert result.exit_code == 0, result.output

    def test_boundary_verify_missing_file(self, tmp_path):
        from constrain.cli import cli
        result = self._run_in(tmp_path, cli, ["boundary", "verify"])
        assert result.exit_code != 0

    def test_export_pact_includes_register(self, grant_project):
        from constrain.cli import cmd_export
        entry = _grant_entry(grant_project)
        session, mgr = _cli_session(render_boundary_register([BoundaryEntry.model_validate(entry)]))
        result = self._run_in(grant_project, cmd_export, ["--format", "pact"], mgr)
        assert result.exit_code == 0, result.output
        out = grant_project / BOUNDARY_REGISTER_FILE
        assert out.read_text() == session.boundary_register_yaml
        assert "boundary_register.yaml" in (grant_project / "task.md").read_text()

    def test_artifact_names_include_register(self):
        from constrain.cli import _ARTIFACT_NAMES
        assert BOUNDARY_REGISTER_FILE in _ARTIFACT_NAMES

    def test_show_prints_register(self, grant_project):
        from constrain.cli import cmd_show
        text = render_boundary_register([BoundaryEntry.model_validate(_grant_entry(grant_project))])
        _, mgr = _cli_session(text)
        result = self._run_in(grant_project, cmd_show, [], mgr)
        assert "=== boundary_register.yaml ===" in result.output

    def test_logger_defined_for_kindex_publish_failure(self):
        from constrain import cli as cli_mod
        session = Session(posture=Posture.critic)
        session.constraints_yaml = "constraints: []"
        with patch.object(cli_mod.kindex, "is_available", return_value=True), \
             patch.object(cli_mod.kindex, "publish_constraints", side_effect=RuntimeError("down")):
            cli_mod._kindex_publish_artifacts(session)  # must not raise NameError


class TestBR016OpenQuestions:
    def test_one_question_per_unverified_side_and_cleared_when_verified(self, grant_project):
        session = Session(posture=Posture.critic, phase=Phase.challenge)
        engine = ConversationEngine(session, _SessionManager(), backend=_Backend(), io=_IO(),
                                    project_root=grant_project)
        bad_cite = _entry("gw-role", [
            _side("gateway", "consumes", "gateway_ro", [{"kind": "file", "path": "missing.txt"}]),
        ])
        engine._apply_model_update({"boundary_register": [bad_cite]})
        engine._apply_model_update({"boundary_register": [bad_cite]})
        qs = [q for q in session.problem_model.open_questions if q.startswith("Boundary gw-role")]
        assert len(qs) == 1

        good = _entry("gw-role", [
            _side("gateway", "consumes", "gateway_ro",
                  [{"kind": "file", "path": "gateway/config.yaml", "line": 2}]),
        ])
        engine._apply_model_update({"boundary_register": [good]})
        assert not [q for q in session.problem_model.open_questions if q.startswith("Boundary gw-role")]
