"""Local agent CLI backend helpers."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess

from . import BackendConnectionError


class LocalAgentBackend:
    """Backend adapter for non-interactive local agent CLIs."""

    def __init__(
        self,
        *,
        name: str,
        command: str,
        args_template: list[str],
        model: str | None = None,
        default_model: str = "",
        cwd: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        env_prefix = f"CONSTRAIN_{name.upper()}"
        self.name = name
        self.command = os.environ.get(f"{env_prefix}_COMMAND", command)
        self.model = model or os.environ.get(f"{env_prefix}_MODEL") or default_model
        self.cwd = cwd
        self.max_tokens = max_tokens
        configured_args = os.environ.get(f"{env_prefix}_ARGS")
        self.args_template = (
            shlex.split(configured_args) if configured_args else args_template
        )
        if shutil.which(self.command) is None:
            raise BackendConnectionError(
                f"{name} command not found: {self.command!r}. "
                f"Install the CLI or set {env_prefix}_COMMAND."
            )

    def complete(self, system: str, messages: list[dict], max_tokens: int | None = None) -> str:
        prompt = self._format_prompt(system, messages, max_tokens or self.max_tokens)
        values = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "max_tokens": str(max_tokens or self.max_tokens),
        }
        cmd = [self.command] + [
            arg.format(**values)
            for arg in self.args_template
            if arg != "{model}" or self.model
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.cwd,
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise BackendConnectionError(str(exc)) from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise BackendConnectionError(
                f"{self.name} exited {result.returncode}: {detail}"
            )
        output = result.stdout.strip()
        if not output:
            raise RuntimeError(f"{self.name} returned an empty response")
        return output

    @staticmethod
    def _format_prompt(system: str, messages: list[dict], max_tokens: int) -> str:
        parts = [
            "System instructions:",
            system,
            "",
            f"Return only the requested content. Target maximum output tokens: {max_tokens}.",
            "",
            "Conversation:",
        ]
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
