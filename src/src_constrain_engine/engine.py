"""Conversation engine: three-phase interview orchestration."""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import anthropic

from .models import Message, Phase, Session
from .posture import get_revision_prompt, get_system_prompt
from .session import SessionManager
from .synthesizer import parse_synthesis_output

# NOTE: this package is NOT shipped (pyproject packages only `src/constrain`)
# and nothing under `src/constrain` imports it — the live model default is
# `constrain.backends.anthropic.DEFAULT_MODEL`. The literal here was
# `claude-sonnet-4-20250514`, which Anthropic has retired; it is corrected so
# no live source names a dead model, but this duplicate constant is the real
# defect. The component and its generated contracts under
# `contracts/src_constrain_engine/` still assert the retired string and need
# regenerating or deleting.
MODEL = "claude-opus-5"


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
        client: anthropic.Anthropic | None = None,
        io: TerminalIO | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.session = session
        self.session_mgr = session_mgr
        self.client = client or anthropic.Anthropic()
        self.io: TerminalIO = io or DefaultIO()
        self.config = config or EngineConfig()

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
                self.session.problem_model.apply_update(model_update)
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
        """Run synthesis phase: generate artifacts, allow one revision."""
        system = get_system_prompt(
            Phase.synthesize, self.session.problem_model
        )
        # Fresh messages for synthesis — not continued conversation
        synth_messages = [{"role": "user", "content": "Generate the artifacts now."}]

        self.io.display("\nGenerating artifacts...\n")

        raw = self._call_api(system, synth_messages)

        try:
            prompt_md, constraints_yaml = parse_synthesis_output(raw)
        except ValueError as e:
            self.io.display(f"Warning: {e}")
            self.io.display("Raw synthesis output:")
            self.io.display(raw)
            self.session.prompt_md = raw
            self.session.constraints_yaml = ""
            self.session_mgr.transition_phase(self.session, Phase.complete)
            self.session_mgr.save(self.session)
            return

        self.io.display("--- prompt.md ---")
        self.io.display(prompt_md[:2000] + ("..." if len(prompt_md) > 2000 else ""))
        self.io.display("\n--- constraints.yaml ---")
        self.io.display(constraints_yaml[:2000] + ("..." if len(constraints_yaml) > 2000 else ""))

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
                prompt_md, constraints_yaml = parse_synthesis_output(raw2)
            except ValueError:
                self.io.display("Could not parse revised output. Keeping original artifacts.")

            self.io.display("\n--- Revised prompt.md ---")
            self.io.display(prompt_md[:2000])
            self.io.display("\n--- Revised constraints.yaml ---")
            self.io.display(constraints_yaml[:2000])

        self.session.prompt_md = prompt_md
        self.session.constraints_yaml = constraints_yaml
        self.session_mgr.transition_phase(self.session, Phase.complete)
        self.session_mgr.save(self.session)

    # ── API layer ─────────────────────────────────────────

    def _call_api(
        self, system: str, messages: list[dict]
    ) -> str:
        """Call Claude with exponential backoff on transient errors."""
        delays = [1, 2, 4]
        last_err: Exception | None = None

        for attempt in range(len(delays) + 1):
            try:
                resp = self.client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=system,
                    messages=messages,
                )
                if not resp.content:
                    raise RuntimeError("API returned an empty response")
                return resp.content[0].text
            except (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
            ) as e:
                last_err = e
                if attempt < len(delays):
                    time.sleep(delays[attempt])
            except anthropic.AuthenticationError:
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
