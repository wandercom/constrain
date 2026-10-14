"""Constrain MCP server: expose session state to AI assistants."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .models import Phase, Session
from .session import SessionManager


class ConstrainMCPServer:
    """Backend-agnostic MCP handler. No dependency on the mcp package."""

    def __init__(self, project_dir: Path | None = None) -> None:
        self._project_dir = project_dir or _find_project_dir()
        self._mgr: SessionManager | None = None

    def _get_mgr(self) -> SessionManager | None:
        if self._mgr is not None:
            return self._mgr
        if self._project_dir is None:
            return None
        self._mgr = SessionManager(self._project_dir)
        return self._mgr

    # ── Resources ──────────────────────────────────────────

    def resource_session_list(self) -> dict:
        mgr = self._get_mgr()
        if mgr is None:
            return {"error": "No project found. Run 'constrain' first."}
        sessions = mgr.list_all()
        return {"sessions": sessions, "count": len(sessions)}

    def resource_session(self, session_id: str) -> dict:
        mgr = self._get_mgr()
        if mgr is None:
            return {"error": "No project found."}
        try:
            s = mgr.load(session_id)
        except (FileNotFoundError, ValueError) as e:
            return {"error": str(e)}
        return _session_detail(s)

    def resource_artifacts(self, session_id: str) -> dict:
        mgr = self._get_mgr()
        if mgr is None:
            return {"error": "No project found."}
        try:
            s = mgr.load(session_id)
        except (FileNotFoundError, ValueError) as e:
            return {"error": str(e)}
        if s.phase != Phase.complete or not s.prompt_md:
            return {
                "session_id": session_id,
                "error": "Session is not complete or has no artifacts.",
                "phase": s.phase.value,
            }
        result = {
            "session_id": session_id,
            "prompt_md": s.prompt_md,
            "constraints_yaml": s.constraints_yaml,
            "is_complete": True,
        }
        if s.trust_policy_yaml:
            result["trust_policy_yaml"] = s.trust_policy_yaml
        if s.component_map_yaml:
            result["component_map_yaml"] = s.component_map_yaml
        if s.schema_hints_yaml:
            result["schema_hints_yaml"] = s.schema_hints_yaml
        if s.boundary_register_yaml:
            result["boundary_register_yaml"] = s.boundary_register_yaml
        return result

    # ── Tools ──────────────────────────────────────────────

    def tool_list_sessions(self) -> dict:
        return self.resource_session_list()

    def tool_show_session(self, session_id: str) -> dict:
        return self.resource_session(session_id)

    def tool_show_artifacts(self, session_id: str) -> dict:
        return self.resource_artifacts(session_id)

    def tool_search_sessions(self, query: str) -> dict:
        mgr = self._get_mgr()
        if mgr is None:
            return {"error": "No project found."}
        query_lower = query.lower()
        results = []
        sessions = mgr.list_all()
        for info in sessions:
            try:
                s = mgr.load(info["id"])
            except Exception:
                continue
            pm = s.problem_model
            searchable = " ".join([
                pm.system_description,
                " ".join(pm.stakeholders),
                " ".join(pm.assumptions),
                " ".join(pm.boundaries),
                " ".join(pm.dependencies),
                " ".join(pm.success_shape),
                " ".join(pm.acceptance_criteria),
            ]).lower()
            if query_lower in searchable:
                results.append({
                    "id": s.id,
                    "phase": s.phase.value,
                    "system_description": pm.system_description[:200],
                    "updated_at": s.updated_at,
                })
        return {"query": query, "matches": results, "count": len(results)}


def _session_detail(s: Session) -> dict:
    pm = s.problem_model
    return {
        "id": s.id,
        "phase": s.phase.value,
        "posture": s.posture.value,
        "round": s.round,
        "understand_rounds": s.understand_rounds,
        "challenge_rounds": s.challenge_rounds,
        "problem_model": {
            "system_description": pm.system_description,
            "stakeholders": pm.stakeholders,
            "failure_modes": pm.failure_modes,
            "dependencies": pm.dependencies,
            "assumptions": pm.assumptions,
            "boundaries": pm.boundaries,
            "history": pm.history,
            "success_shape": pm.success_shape,
            "acceptance_criteria": pm.acceptance_criteria,
            "open_questions": pm.open_questions,
            "boundary_register": [
                e.model_dump(by_alias=True, mode="json") for e in pm.boundary_register
            ],
        },
        "has_prompt_md": bool(s.prompt_md),
        "has_constraints_yaml": bool(s.constraints_yaml),
        "has_trust_policy_yaml": bool(s.trust_policy_yaml),
        "has_component_map_yaml": bool(s.component_map_yaml),
        "has_schema_hints_yaml": bool(s.schema_hints_yaml),
        "has_boundary_register_yaml": bool(s.boundary_register_yaml),
        "created_at": s.created_at,
        "updated_at": s.updated_at,
    }


def _find_project_dir() -> Path | None:
    env = os.environ.get("CONSTRAIN_PROJECT_DIR")
    if env:
        return Path(env)
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        if (d / ".constrain").is_dir():
            return d
    return cwd


def _json_str(obj: dict) -> str:
    return json.dumps(obj, indent=2, default=str)


def _create_mcp_app():
    from mcp.server.fastmcp import FastMCP

    app = FastMCP(
        "constrain",
        instructions=(
            "Constrain helps engineers articulate problems through a three-phase "
            "interview (understand, challenge, synthesize). It produces prompt.md "
            "(an induced-understanding briefing), constraints.yaml (boundary "
            "conditions), trust_policy.yaml (trust and authority model), "
            "component_map.yaml (component topology), schema_hints.yaml "
            "(storage schema hints for Ledger), and boundary_register.yaml "
            "(exact cross-boundary identifiers with verifiable records). Use the tools below to "
            "inspect sessions and artifacts."
        ),
    )

    server = ConstrainMCPServer()

    # ── Resources ──────────────────────────────────────

    @app.resource("constrain://sessions")
    def res_sessions() -> str:
        return _json_str(server.resource_session_list())

    @app.resource("constrain://session/{session_id}")
    def res_session(session_id: str) -> str:
        return _json_str(server.resource_session(session_id))

    @app.resource("constrain://artifacts/{session_id}")
    def res_artifacts(session_id: str) -> str:
        return _json_str(server.resource_artifacts(session_id))

    # ── Tools ──────────────────────────────────────────

    @app.tool()
    def constrain_list_sessions(project_dir: str | None = None) -> str:
        """List all Constrain sessions with their status."""
        srv = ConstrainMCPServer(Path(project_dir) if project_dir else None)
        return _json_str(srv.tool_list_sessions())

    @app.tool()
    def constrain_show_session(session_id: str, project_dir: str | None = None) -> str:
        """Show full details for a specific session including the problem model."""
        srv = ConstrainMCPServer(Path(project_dir) if project_dir else None)
        return _json_str(srv.tool_show_session(session_id))

    @app.tool()
    def constrain_show_artifacts(session_id: str, project_dir: str | None = None) -> str:
        """Get the prompt.md and constraints.yaml artifacts from a completed session."""
        srv = ConstrainMCPServer(Path(project_dir) if project_dir else None)
        return _json_str(srv.tool_show_artifacts(session_id))

    @app.tool()
    def constrain_search_sessions(query: str, project_dir: str | None = None) -> str:
        """Search sessions by keyword across problem model fields."""
        srv = ConstrainMCPServer(Path(project_dir) if project_dir else None)
        return _json_str(srv.tool_search_sessions(query))

    # ── Prompts ────────────────────────────────────────

    @app.prompt()
    def session_overview() -> str:
        """Get an overview of all sessions and their current state."""
        data = server.resource_session_list()
        if "error" in data:
            return data["error"]
        lines = [f"Constrain: {data['count']} session(s)\n"]
        for s in data.get("sessions", []):
            sid = s["id"][:8]
            phase = s["phase"]
            rounds = s["understand_rounds"] + s["challenge_rounds"]
            lines.append(f"  {sid}  {phase:<12}  {rounds} rounds  {s['updated_at'][:19]}")
        return "\n".join(lines)

    return app


def main() -> None:
    try:
        app = _create_mcp_app()
    except ImportError:
        print(
            "Error: the 'mcp' package is not installed.\n"
            "Install with: pip install constrain[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)
    app.run()
