"""Conversation engine: three-phase interview orchestration."""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .backends import (
    Backend,
    BackendAuthError,
    BackendConnectionError,
    BackendRateLimitError,
    BackendTimeoutError,
    create_backend,
)
from .boundary import (
    anchor_boundary_update,
    blocking_findings,
    errors as boundary_errors,
    load_boundary_register_text,
    render_boundary_register,
    unverified_questions,
    validate_boundary_register,
)
from .models import Message, Phase, Session
from .posture import get_prime_prompt, get_revision_prompt, get_system_prompt
from .session import SessionManager
from .synthesizer import parse_synthesis_output, validate_artifacts, validate_yaml_content


class TerminalIO(Protocol):
    def display(self, text: str) -> None: ...
    def prompt(self, prefix: str) -> str: ...


class DefaultIO:
    """Production terminal I/O using print/input."""

    def display(self, text: str) -> None:
        print(text)

    def prompt(self, prefix: str = "> ") -> str:
        return input(prefix)


@dataclass
class EngineConfig:
    understand_min: int = 2
    understand_max: int = 10
    challenge_min: int = 2
    challenge_max: int = 10


class ConversationEngine:
    """Orchestrates the three-phase constrain interview."""

    def __init__(
        self,
        session: Session,
        session_mgr: SessionManager,
        backend: Backend | None = None,
        io: TerminalIO | None = None,
        config: EngineConfig | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.session = session
        self.session_mgr = session_mgr
        self.backend: Backend = backend or create_backend()
        self.io: TerminalIO = io or DefaultIO()
        self.config = config or EngineConfig()
        # Root that boundary-register record paths are relative to.
        self.project_root = Path(project_root) if project_root else Path.cwd()

    # ── public API ────────────────────────────────────────

    def run_session(self) -> Session:
        """Run all remaining phases. Saves on interrupt."""
        try:
            if self.session.round > 0:
                self._display_resume_summary()

            phases = [Phase.understand, Phase.challenge, Phase.synthesize]
            for phase in phases:
                if self._phase_done(phase):
                    continue
                if self.session.phase != phase:
                    continue

                if phase == Phase.synthesize:
                    self._run_synthesis()
                else:
                    self._run_phase(phase)

                # advance to next phase if not already there
                if phase != Phase.synthesize:
                    next_phase = phases[phases.index(phase) + 1]
                    self.session_mgr.transition_phase(self.session, next_phase)
                    self.session_mgr.save(self.session)
                    self.io.display(f"\n--- Moving to {next_phase.value} phase ---\n")

        except KeyboardInterrupt:
            self.io.display("\n\nSession saved. You can resume later with 'constrain resume'.")
            self.session_mgr.save(self.session)
            sys.exit(0)

        return self.session

    # ── document priming ───────────────────────────────────

    def prime_with_document(self, path: Path) -> dict:
        """Ingest a document and extract problem model information.

        Returns a dict with keys: 'extracted_count', 'summary'.
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        word_count = len(text.split())

        # Truncate very large documents to avoid blowing context
        max_chars = 100_000
        if len(text) > max_chars:
            text = text[:max_chars]
            self.io.display(f"  (Truncated to ~{max_chars // 1000}k chars)")

        system = get_prime_prompt(text, self.session.problem_model)
        messages = [{"role": "user", "content": "Analyze this document."}]

        raw = self._call_api(system, messages)
        display_text, model_update, _ = self._parse_response(raw)

        if model_update:
            self._apply_model_update(model_update, primed_path=path)
            self.session.touch()
            self.session_mgr.save(self.session)

        # Try to get extracted_count from the JSON
        count = 0
        if model_update:
            count = sum(
                len(v) if isinstance(v, list) else 1
                for v in model_update.values()
            )

        return {
            "word_count": word_count,
            "extracted_count": count,
            "summary": display_text[:200],
            "fields": list(model_update.keys()) if model_update else [],
        }

    def load_prior_register(self, text: str) -> int:
        """Load a previous session's boundary register without an LLM.

        Each prior record is re-anchored against the current files (line
        re-confirmed, digest recomputed); a record whose literal is gone is
        dropped and its side becomes UNVERIFIED with an open question.
        Returns the number of entries loaded.
        """
        try:
            data = load_boundary_register_text(text)
        except ValueError as e:
            self.io.display(f"  Skipping previous {e}")
            return 0
        entries = data.get("entries") or []
        if not isinstance(entries, list) or not entries:
            return 0
        before = len(self.session.problem_model.boundary_register)
        self._apply_model_update({"boundary_register": entries})
        self.session.touch()
        self.session_mgr.save(self.session)
        return len(self.session.problem_model.boundary_register) - before

    def prime_interactive(self, initial_paths: list[Path] | None = None) -> int:
        """Interactive priming loop. Returns number of documents ingested."""
        docs_ingested = 0

        # Process initial paths from CLI
        if initial_paths:
            for p in initial_paths:
                if not p.exists():
                    self.io.display(f"  Skipping {p} (not found)")
                    continue
                self.io.display(f"Ingesting {p.name}...")
                result = self.prime_with_document(p)
                docs_ingested += 1
                fields = ", ".join(result["fields"]) if result["fields"] else "none"
                self.io.display(
                    f"  {result['word_count']} words -> "
                    f"{result['extracted_count']} items extracted ({fields})"
                )

        # Interactive loop
        while True:
            try:
                response = self.io.prompt(
                    "\nAdd document (path or Enter to start interview): "
                )
            except EOFError:
                break

            response = response.strip()
            if not response:
                break

            p = Path(response).expanduser()
            if not p.exists():
                self.io.display(f"  Not found: {p}")
                continue

            self.io.display(f"Ingesting {p.name}...")
            result = self.prime_with_document(p)
            docs_ingested += 1
            fields = ", ".join(result["fields"]) if result["fields"] else "none"
            self.io.display(
                f"  {result['word_count']} words -> "
                f"{result['extracted_count']} items extracted ({fields})"
            )

        if docs_ingested > 0:
            from .posture import _format_problem_model
            self.io.display(f"\nPrimed with {docs_ingested} document(s). Current understanding:")
            self.io.display(_format_problem_model(self.session.problem_model))
            self.io.display("")

        return docs_ingested

    # ── phase runners ─────────────────────────────────────

    def _run_phase(self, phase: Phase) -> None:
        """Run understand or challenge phase conversation loop."""
        min_rounds, max_rounds = self._round_limits(phase)

        while True:
            current = self._current_rounds(phase)
            if current >= max_rounds:
                break

            system = get_system_prompt(
                phase, self.session.problem_model, self.session.posture
            )
            messages = self._api_messages()

            # Ensure messages end with a user message before calling the API.
            # This handles: (a) brand-new phase with no messages,
            # (b) phase transition where prior phase left an assistant message last,
            # (c) resume where session was saved after assistant response.
            if not messages or messages[-1]["role"] == "assistant":
                if self._current_rounds(phase) == 0:
                    # New or freshly-transitioned phase — add an opening prompt
                    if phase == Phase.understand:
                        opening = "I'd like help articulating a problem I'm working on."
                    else:
                        opening = (
                            "Please challenge my problem description. "
                            "Here's what I have so far."
                        )
                    self._add_message("user", opening)
                    messages = self._api_messages()
                else:
                    # Resuming mid-phase — get user input before next API call
                    try:
                        user_input = self.io.prompt("\n> ")
                        if not user_input.strip():
                            continue
                        self._add_message("user", user_input.strip())
                        self.session_mgr.save(self.session)
                        messages = self._api_messages()
                    except EOFError:
                        self.io.display("\n(Advancing to next phase)")
                        break

            raw = self._call_api(system, messages)
            display_text, model_update, ready = self._parse_response(raw)

            # Store full response, display clean text
            self._add_message("assistant", raw)
            self._increment_round(phase)
            self.session_mgr.save(self.session)

            self.io.display(f"\n{display_text}\n")

            if model_update:
                self._apply_model_update(model_update)
                self.session.touch()

            current = self._current_rounds(phase)
            if ready and current >= min_rounds:
                break
            if current >= max_rounds:
                break

            # Get user input
            try:
                user_input = self.io.prompt("\n> ")
                if not user_input.strip():
                    continue
                self._add_message("user", user_input.strip())
                self.session_mgr.save(self.session)
            except EOFError:
                self.io.display("\n(Advancing to next phase)")
                break

    def _run_synthesis(self) -> None:
        """Run synthesis phase: generate artifacts, allow one revision.

        Blocked (session stays in synthesize, nothing is marked complete)
        while the boundary register has errors other than UNVERIFIED sides.
        """
        if not self._boundary_gate():
            return

        system = get_system_prompt(
            Phase.synthesize, self.session.problem_model
        )
        # Fresh messages for synthesis — not continued conversation. The
        # transcript is passed verbatim so identifiers are copied, not
        # re-guessed from the lossy problem-model summary.
        synth_messages = [{"role": "user", "content": self._synthesis_request()}]

        self.io.display("\nGenerating artifacts...\n")

        raw = self._call_api(system, synth_messages)

        try:
            artifacts = parse_synthesis_output(raw)
        except ValueError as e:
            self.io.display(f"Warning: {e}")
            self.io.display("Raw synthesis output:")
            self.io.display(raw)
            self.session.prompt_md = raw
            self.session.constraints_yaml = ""
            self.session.trust_policy_yaml = ""
            self.session.component_map_yaml = ""
            self.session.schema_hints_yaml = ""
            self._finalize_boundary_register()
            self.session_mgr.transition_phase(self.session, Phase.complete)
            self.session_mgr.save(self.session)
            return

        artifacts = self._repair_artifacts_if_needed(artifacts)
        self._warn_about_artifact_validation(artifacts)

        self._display_synthesis_preview(artifacts)

        # One revision cycle
        try:
            self.io.display(
                "\nReview the artifacts above. Enter feedback to revise, "
                "or press Enter/Ctrl+D to accept."
            )
            feedback = self.io.prompt("\nFeedback> ")
        except EOFError:
            feedback = ""

        if feedback.strip():
            revision_prompt = get_revision_prompt(
                feedback.strip(), self.session.problem_model
            )
            revision_messages = [{"role": "user", "content": revision_prompt}]
            raw2 = self._call_api(
                "You are a technical writer revising artifacts based on feedback.",
                revision_messages,
            )
            try:
                artifacts = parse_synthesis_output(raw2)
            except ValueError:
                self.io.display("Could not parse revised output. Keeping original artifacts.")
            else:
                artifacts = self._repair_artifacts_if_needed(artifacts)
                self._warn_about_artifact_validation(artifacts)

            self._display_synthesis_preview(artifacts, prefix="Revised ")

        self.session.prompt_md = artifacts.prompt_md
        self.session.constraints_yaml = artifacts.constraints_yaml
        self.session.trust_policy_yaml = artifacts.trust_policy_yaml
        self.session.component_map_yaml = artifacts.component_map_yaml
        self.session.schema_hints_yaml = artifacts.schema_hints_yaml
        self._finalize_boundary_register()
        self.session_mgr.transition_phase(self.session, Phase.complete)
        self.session_mgr.save(self.session)

    # ── boundary register ─────────────────────────────────

    def _apply_model_update(self, update: dict, primed_path: Path | None = None) -> None:
        """Apply an LLM update; boundary records are anchored by code first."""
        pm = self.session.problem_model
        questions: list[str] = []
        if "boundary_register" in update:
            default_path = None
            if primed_path is not None:
                try:
                    default_path = (
                        Path(primed_path).resolve()
                        .relative_to(self.project_root.resolve()).as_posix()
                    )
                except ValueError:
                    questions.append(
                        f"Primed document {primed_path} is outside the project root; "
                        "identifiers taken from it are UNVERIFIED until a project file records them."
                    )
            anchored, notes = anchor_boundary_update(
                update["boundary_register"], self.project_root, default_path=default_path
            )
            questions.extend(notes)
            update = {**update, "boundary_register": anchored}
        before = set(pm.open_questions)
        pm.apply_update(update)
        for q in pm.open_questions:
            if q not in before and q.startswith("Rejected boundary entry"):
                self.io.display(f"  {q}")
        questions.extend(unverified_questions(pm.boundary_register))

        # One open question per unverified side; drop questions for sides
        # that now have a record. Keys look like "Boundary <id>: <party> (<role>)".
        def key(q: str) -> str | None:
            return q.split(") ", 1)[0] + ")" if q.startswith("Boundary ") else None

        verified = {
            f"Boundary {e.id}: {s.party} ({s.role})"
            for e in pm.boundary_register for s in e.sides if s.records
        }
        pm.open_questions[:] = [q for q in pm.open_questions if key(q) not in verified]
        for q in questions:
            k = key(q)
            if k in verified:
                continue
            if not any(key(existing) == k for existing in pm.open_questions):
                pm.open_questions.append(q)

    def _boundary_findings(self):
        return validate_boundary_register(
            self.session.problem_model.boundary_register, self.project_root
        )

    def _boundary_gate(self) -> bool:
        """Return True if synthesis may proceed; otherwise try to resolve.

        Errors other than UNVERIFIED (mismatch, near-miss, stale or missing
        records, placeholders, malformed relations) block synthesis. The
        engineer gets a bounded resolution conversation; if errors remain the
        session is saved in the synthesize phase and nothing is written.
        """
        blocking = blocking_findings(self._boundary_findings())
        if not blocking:
            return True
        self.io.display("\nBoundary verification failed:")
        for f in blocking:
            self.io.display(f"  {f}")
        if self._resolve_boundaries(blocking):
            self.io.display("Boundary register now verifies. Continuing synthesis.")
            return True
        self.io.display(
            "\nSynthesis blocked: the boundary register still has errors. Fix the system "
            "or the cited records, then run 'constrain resume'."
        )
        self.session_mgr.save(self.session)
        return False

    def _resolve_boundaries(self, blocking) -> bool:
        """Challenge-posture conversation focused on failing boundary entries."""
        self._add_message(
            "user",
            "Boundary verification failed. For each finding below, ask me for the exact "
            "literal each side uses and where it is recorded, then re-send the corrected "
            "boundary_register entry (same id). Do not guess values.\n"
            + "\n".join(f"- {f}" for f in blocking),
        )
        for _ in range(self.config.challenge_max):
            system = get_system_prompt(
                Phase.challenge, self.session.problem_model, self.session.posture
            )
            raw = self._call_api(system, self._api_messages())
            display_text, model_update, _ = self._parse_response(raw)
            self._add_message("assistant", raw)
            self.session.round += 1
            self.io.display(f"\n{display_text}\n")
            if model_update:
                self._apply_model_update(model_update)
            self.session_mgr.save(self.session)
            blocking = blocking_findings(self._boundary_findings())
            if not blocking:
                return True
            for f in blocking:
                self.io.display(f"  {f}")
            try:
                user_input = self.io.prompt("\n> ")
            except EOFError:
                return False
            if user_input.strip():
                self._add_message("user", user_input.strip())
                self.session_mgr.save(self.session)
        return False

    def _finalize_boundary_register(self) -> None:
        """Write the register artifact from the typed model (never from LLM text)."""
        pm = self.session.problem_model
        self.session.boundary_register_yaml = render_boundary_register(pm.boundary_register)
        unverified = [f for f in boundary_errors(self._boundary_findings()) if not f.blocking]
        if unverified:
            self.io.display(
                f"\nboundary_register.yaml: {len(unverified)} UNVERIFIED side(s) written and "
                "flagged; 'constrain validate' fails until each has a record:"
            )
            for f in unverified:
                self.io.display(f"  {f}")

    def _synthesis_request(self, max_chars: int = 150_000) -> str:
        transcript = "\n\n".join(
            f"[{m.role}]\n{m.content}" for m in self.session.conversation
        )
        if len(transcript) > max_chars:
            transcript = "(earlier transcript truncated)\n" + transcript[-max_chars:]
        if not transcript:
            return "Generate the artifacts now."
        return (
            "Generate the artifacts now.\n\n"
            "Interview transcript (verbatim — copy identifiers exactly as written here "
            "and in the boundary register; never re-invent them):\n\n"
            + transcript
        )

    def _display_synthesis_preview(self, artifacts, prefix: str = "") -> None:
        """Display a truncated preview of all synthesis artifacts."""
        limit = 2000
        for label, content in [
            ("prompt.md", artifacts.prompt_md),
            ("constraints.yaml", artifacts.constraints_yaml),
            ("trust_policy.yaml", artifacts.trust_policy_yaml),
            ("component_map.yaml", artifacts.component_map_yaml),
            ("schema_hints.yaml", artifacts.schema_hints_yaml),
            (
                "boundary_register.yaml (written by code)",
                render_boundary_register(self.session.problem_model.boundary_register)
                if self.session.problem_model.boundary_register else "",
            ),
        ]:
            if content:
                self.io.display(f"\n--- {prefix}{label} ---")
                self.io.display(
                    content[:limit] + ("..." if len(content) > limit else "")
                )

    def _repair_artifacts_if_needed(self, artifacts):
        """Ask the backend for a YAML-only repair pass when artifacts are invalid."""
        yaml_errors = self._collect_yaml_errors(artifacts)
        if not yaml_errors:
            return artifacts

        self.io.display("Generated artifacts contain invalid YAML. Attempting automatic repair...")
        for err in yaml_errors:
            self.io.display(f"YAML validation error: {err}")

        repair_system = (
            "You repair generated Constrain artifacts. Return the complete artifact set "
            "using exactly these delimiters: --- PROMPT ---, --- CONSTRAINTS ---, "
            "--- TRUST_POLICY ---, --- COMPONENT_MAP ---, --- SCHEMA_HINTS ---. "
            "Fix YAML syntax only unless a minimal structural correction is required. "
            "Do not add commentary outside the delimited artifacts."
        )
        repair_user = (
            "Repair the YAML errors below and return all artifacts.\n\n"
            "Errors:\n"
            + "\n".join(f"- {err}" for err in yaml_errors)
            + "\n\n--- PROMPT ---\n"
            + artifacts.prompt_md
            + "\n--- CONSTRAINTS ---\n"
            + artifacts.constraints_yaml
            + "\n--- TRUST_POLICY ---\n"
            + artifacts.trust_policy_yaml
            + "\n--- COMPONENT_MAP ---\n"
            + artifacts.component_map_yaml
            + "\n--- SCHEMA_HINTS ---\n"
            + artifacts.schema_hints_yaml
        )

        try:
            repaired_raw = self._call_api(
                repair_system,
                [{"role": "user", "content": repair_user}],
            )
            repaired = parse_synthesis_output(repaired_raw)
        except Exception as e:
            self.io.display(f"Automatic repair failed: {e}")
            return artifacts

        repaired_errors = self._collect_yaml_errors(repaired)
        if repaired_errors:
            self.io.display("Automatic repair did not produce valid YAML.")
            return repaired

        self.io.display("Automatic YAML repair succeeded.")
        return repaired

    def _collect_yaml_errors(self, artifacts) -> list[str]:
        """Return YAML validation errors for all generated YAML artifacts."""
        yaml_errors = []
        for name, content in [
            ("constraints.yaml", artifacts.constraints_yaml),
            ("trust_policy.yaml", artifacts.trust_policy_yaml),
            ("component_map.yaml", artifacts.component_map_yaml),
            ("schema_hints.yaml", artifacts.schema_hints_yaml),
        ]:
            if content:
                try:
                    validate_yaml_content(content, name)
                except ValueError as e:
                    yaml_errors.append(str(e))
        return yaml_errors

    def _warn_about_artifact_validation(self, artifacts) -> None:
        """Report artifact validation problems without aborting synthesis."""
        yaml_errors = self._collect_yaml_errors(artifacts)
        if yaml_errors:
            for err in yaml_errors:
                self.io.display(f"YAML validation error: {err}")
            self.io.display("Skipping cross-validation until YAML is valid.")
            return

        try:
            warnings = validate_artifacts(artifacts)
            for w in warnings:
                self.io.display(f"Warning: {w}")
        except ValueError as e:
            self.io.display(f"Cross-validation error (non-fatal): {e}")

    # ── API layer ─────────────────────────────────────────

    def _call_api(
        self, system: str, messages: list[dict]
    ) -> str:
        """Call the backend with exponential backoff on transient errors."""
        # Optional register normalization via Transmogrifier
        try:
            from transmogrifier.core import Transmogrifier
            from transmogrifier.system_prompts import inject_system_prompt
            _transmog = Transmogrifier()
            # Translate the last user message
            for msg in reversed(messages):
                if msg["role"] == "user":
                    result = _transmog.translate(msg["content"])
                    msg["content"] = result.output_text
                    if result.system_prompt:
                        system = inject_system_prompt(system, result.system_prompt)
                    break
        except ImportError:
            pass
        except Exception:
            pass  # fail-safe: never let translation break the pipeline

        delays = [1, 2, 4]
        last_err: Exception | None = None

        for attempt in range(len(delays) + 1):
            try:
                return self.backend.complete(system, messages)
            except (
                BackendRateLimitError,
                BackendTimeoutError,
                BackendConnectionError,
            ) as e:
                last_err = e
                if attempt < len(delays):
                    time.sleep(delays[attempt])
            except BackendAuthError:
                raise
        raise RuntimeError(f"API unavailable after retries: {last_err}")

    # ── response parsing ──────────────────────────────────

    def _parse_response(self, raw: str) -> tuple[str, dict, bool]:
        """Extract display text, model update dict, and ready flag from LLM response.

        Returns (display_text, model_update, ready_to_proceed).
        """
        # Find last ```json ... ``` block
        pattern = r"```json\s*\n(.*?)```"
        matches = list(re.finditer(pattern, raw, re.DOTALL))

        if not matches:
            return raw.strip(), {}, False

        last_match = matches[-1]
        json_str = last_match.group(1).strip()
        display = (raw[: last_match.start()] + raw[last_match.end() :]).strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return raw.strip(), {}, False

        ready = bool(data.get("ready_to_proceed", False))
        update = data.get("problem_model_update", {})
        if not isinstance(update, dict):
            update = {}

        return display, update, ready

    # ── helpers ───────────────────────────────────────────

    def _api_messages(self) -> list[dict]:
        """Build API message list from session conversation history."""
        return [{"role": m.role, "content": m.content} for m in self.session.conversation]

    def _add_message(self, role: str, content: str) -> None:
        self.session.conversation.append(Message(role=role, content=content))
        self.session.touch()

    def _increment_round(self, phase: Phase) -> None:
        self.session.round += 1
        if phase == Phase.understand:
            self.session.understand_rounds += 1
        elif phase == Phase.challenge:
            self.session.challenge_rounds += 1

    def _current_rounds(self, phase: Phase) -> int:
        if phase == Phase.understand:
            return self.session.understand_rounds
        elif phase == Phase.challenge:
            return self.session.challenge_rounds
        return 0

    def _round_limits(self, phase: Phase) -> tuple[int, int]:
        if phase == Phase.understand:
            return self.config.understand_min, self.config.understand_max
        elif phase == Phase.challenge:
            return self.config.challenge_min, self.config.challenge_max
        return 1, 1

    def _phase_done(self, phase: Phase) -> bool:
        """Check if a phase is already completed based on session state."""
        phase_order = [Phase.understand, Phase.challenge, Phase.synthesize, Phase.complete]
        current_idx = phase_order.index(self.session.phase)
        target_idx = phase_order.index(phase)
        return current_idx > target_idx

    def _display_resume_summary(self) -> None:
        pm = self.session.problem_model
        self.io.display(f"Resuming session {self.session.id[:8]}...")
        self.io.display(f"  Phase: {self.session.phase.value}")
        self.io.display(
            f"  Rounds: {self.session.understand_rounds} understand, "
            f"{self.session.challenge_rounds} challenge"
        )
        if pm.system_description:
            self.io.display(f"  System: {pm.system_description[:80]}")
        if pm.stakeholders:
            self.io.display(f"  Stakeholders: {', '.join(pm.stakeholders[:5])}")
        self.io.display("")
