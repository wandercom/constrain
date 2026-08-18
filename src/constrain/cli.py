"""Click CLI for Constrain."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from .backends import create_backend
from .engine import ConversationEngine, EngineConfig
from .models import Phase, Posture, Session
from .session import SessionManager
from .archive import archive_artifacts, list_archived_sessions, load_archived_artifacts
from . import kindex_integration as kindex
from .synthesizer import validate_artifacts, validate_yaml_content, write_artifacts, SynthesisArtifacts


class SafeGroup(click.Group):
    """Click group with centralized exception handling."""

    def invoke(self, ctx: click.Context) -> None:
        try:
            super().invoke(ctx)
        except click.ClickException:
            raise
        except click.Abort:
            click.echo("Aborted.", err=True)
            sys.exit(1)
        except KeyboardInterrupt:
            click.echo("\nInterrupted.", err=True)
            sys.exit(130)
        except SystemExit:
            raise
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


def ensure_api_key() -> None:
    """Validate that the required API key is set for the active backend."""
    backend_name = os.environ.get("CONSTRAIN_BACKEND", "anthropic")
    if backend_name == "anthropic":
        key_names = (
            "WANDER_ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY",
            "JMC_ANTHROPIC_API_KEY",
        )
        if not any(os.environ.get(name, "").strip() for name in key_names):
            raise click.ClickException(
                "No Anthropic API key set. Export WANDER_ANTHROPIC_API_KEY "
                "(preferred), ANTHROPIC_API_KEY, or JMC_ANTHROPIC_API_KEY."
            )
    elif backend_name == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        base = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not key and not base:
            raise click.ClickException(
                "OPENAI_API_KEY not set (or set OPENAI_BASE_URL for local models)."
            )
    elif backend_name in {"codex", "claude", "opencode", "cursor"}:
        return


def _round_options(f):
    """Shared round-limit options for commands that run sessions."""
    f = click.option(
        "--min-understand", type=int, default=None,
        help="Minimum rounds in understand phase (default: 2).",
    )(f)
    f = click.option(
        "--max-understand", type=int, default=None,
        help="Maximum rounds in understand phase (default: 10).",
    )(f)
    f = click.option(
        "--min-challenge", type=int, default=None,
        help="Minimum rounds in challenge phase (default: 2).",
    )(f)
    f = click.option(
        "--max-challenge", type=int, default=None,
        help="Maximum rounds in challenge phase (default: 10).",
    )(f)
    return f


def _resolve_int(cli_val: int | None, env_var: str, default: int) -> int:
    if cli_val is not None:
        return cli_val
    env = os.environ.get(env_var)
    if env:
        try:
            v = int(env)
            if v < 1:
                raise ValueError
            return v
        except ValueError:
            raise click.ClickException(
                f"Invalid value for {env_var}: expected a positive integer, got '{env}'."
            )
    return default


def _resolve_optional_int_env(env_var: str) -> int | None:
    env = os.environ.get(env_var)
    if not env:
        return None
    try:
        value = int(env)
    except ValueError as exc:
        raise click.ClickException(
            f"Invalid value for {env_var}: expected a positive integer, got '{env}'."
        ) from exc
    if value < 1:
        raise click.ClickException(
            f"Invalid value for {env_var}: expected a positive integer, got '{env}'."
        )
    return value


def resolve_config(
    min_understand: int | None,
    max_understand: int | None,
    min_challenge: int | None,
    max_challenge: int | None,
) -> EngineConfig:
    cfg = EngineConfig(
        understand_min=_resolve_int(min_understand, "CONSTRAIN_MIN_UNDERSTAND", 2),
        understand_max=_resolve_int(max_understand, "CONSTRAIN_MAX_UNDERSTAND", 10),
        challenge_min=_resolve_int(min_challenge, "CONSTRAIN_MIN_CHALLENGE", 2),
        challenge_max=_resolve_int(max_challenge, "CONSTRAIN_MAX_CHALLENGE", 10),
    )
    if cfg.understand_min > cfg.understand_max:
        raise click.ClickException(
            f"min-understand ({cfg.understand_min}) cannot exceed max-understand ({cfg.understand_max})."
        )
    if cfg.challenge_min > cfg.challenge_max:
        raise click.ClickException(
            f"min-challenge ({cfg.challenge_min}) cannot exceed max-challenge ({cfg.challenge_max})."
        )
    return cfg


_ARTIFACT_NAMES = ["prompt.md", "constraints.yaml", "trust_policy.yaml", "component_map.yaml", "schema_hints.yaml"]


def _archive_dir(cwd: Path) -> Path:
    """Archive directory: .constrain/archive/ under the working directory."""
    return cwd / ".constrain" / "archive"


def _kindex_prompt_and_index(cwd: Path) -> None:
    """Check kindex availability and offer code indexing on first run."""
    if not kindex.is_available():
        return

    auto = kindex.should_auto_index(cwd)
    if auto is True:
        click.echo("Kindex: auto-indexing codebase...")
        kindex.index_codebase(cwd)
    elif auto is None:
        # Not configured — prompt
        choice = click.prompt(
            "Kindex detected. Index this codebase for cross-session context?",
            type=click.Choice(["y", "n", "always", "never"], case_sensitive=False),
            default="y",
        )
        if choice in ("y", "always"):
            click.echo("Indexing codebase...")
            kindex.index_codebase(cwd)
        if choice == "always":
            kindex.write_kin_config(cwd, {"auto_index": True})
            click.echo("  Saved to .kin/config (auto_index: true)")
        elif choice == "never":
            kindex.write_kin_config(cwd, {"auto_index": False})
            click.echo("  Saved to .kin/config (auto_index: false)")
    # auto is False → skip silently


def _auto_prime_previous(engine: ConversationEngine, cwd: Path) -> None:
    """Prime the engine with previous artifacts and kindex context.

    Sources (in order):
    1. Kindex graph context for the project topic
    2. Most recent archived constrain session artifacts
    """
    parts = []

    # 1. Kindex graph context
    if kindex.is_available():
        kin_config = kindex.read_kin_config(cwd)
        topic = kin_config.get("name", cwd.name)
        graph_context = kindex.fetch_context(f"{topic} constraints boundaries")
        if graph_context.strip():
            parts.append(f"=== Kindex Graph Context ===\n{graph_context}")
            click.echo(f"Loaded kindex context for '{topic}'.")

    # 2. Archived artifacts
    archive_base = _archive_dir(cwd)
    previous = load_archived_artifacts(archive_base)
    if previous:
        for name in ("constraints.yaml", "prompt.md", "trust_policy.yaml",
                     "component_map.yaml", "schema_hints.yaml"):
            content = previous.get(name, "")
            if content.strip():
                parts.append(f"=== Previous {name} ===\n{content}")

    if not parts:
        return

    sessions = list_archived_sessions(archive_base)
    slug = sessions[0]["slug"] if sessions else "project"
    if previous:
        click.echo(f"Found previous artifacts ({slug}). Priming with prior context...")

    import tempfile
    context = (
        f"# Prior Context for {slug}\n\n"
        "The following context comes from the kindex knowledge graph and/or a previous\n"
        "constrain session for this project. Reference it when relevant — build on\n"
        "existing constraints rather than contradicting them, unless the user indicates\n"
        "otherwise.\n\n"
        + "\n\n".join(parts)
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="prior_", delete=False, encoding="utf-8"
    ) as f:
        f.write(context)
        tmp_path = Path(f.name)
    try:
        result = engine.prime_with_document(tmp_path)
        click.echo(
            f"  Primed with {result['extracted_count']} items from prior context."
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _kindex_publish_artifacts(session: Session) -> None:
    """Publish completed session artifacts to kindex (if available)."""
    if not kindex.is_available():
        return
    try:
        published = 0
        # Publish individual constraints as constraint nodes
        if session.constraints_yaml:
            published += kindex.publish_constraints(session.constraints_yaml)
        # Publish components from component_map
        if session.component_map_yaml:
            published += kindex.publish_components(session.component_map_yaml)
        # Publish system description as a concept
        if session.problem_model.system_description:
            kindex.publish_node(
                title=f"System: {session.problem_model.system_description}",
                content=session.prompt_md[:2000] if session.prompt_md else "",
                node_type="concept",
                tags=["constrain", "system"],
            )
            published += 1
        if published:
            click.echo(f"\nPublished {published} item(s) to kindex.")
    except Exception as e:
        logger.debug("Kindex publish failed: %s", e)


def _run_engine(
    session: Session,
    mgr: SessionManager,
    config: EngineConfig,
    prime_paths: list[Path] | None = None,
    backend_name: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> None:
    """Create engine and run session, then write artifacts on completion."""
    backend = create_backend(backend=backend_name, model=model, max_tokens=max_tokens)
    engine = ConversationEngine(session=session, session_mgr=mgr, backend=backend, config=config)

    # Kindex: first-run indexing prompt (only for new sessions)
    if session.round == 0:
        _kindex_prompt_and_index(Path.cwd())

    # Document priming: kindex context + previous artifacts + user files + interactive
    if prime_paths or session.round == 0:
        if session.round == 0:
            _auto_prime_previous(engine, Path.cwd())

        has_paths = bool(prime_paths)
        if has_paths:
            engine.prime_interactive(initial_paths=prime_paths)
        elif session.round == 0:
            if click.confirm("Prime with documents before starting?", default=False):
                engine.prime_interactive()

    engine.run_session()

    if session.phase == Phase.complete and session.prompt_md:
        cwd = Path.cwd()
        # Archive existing artifacts into .constrain/archive/<slug>/
        subdir, archived = archive_artifacts(
            cwd,
            _ARTIFACT_NAMES,
            archive_base=_archive_dir(cwd),
            slug_source_priority=["prompt.md", "constraints.yaml"],
        )
        if archived:
            click.echo(f"\nArchived previous artifacts to {subdir.name}/:")
            for orig, dest in archived:
                click.echo(f"  {orig.name}")

        written = write_artifacts(
            session.prompt_md,
            session.constraints_yaml,
            cwd,
            overwrite=False,
            trust_policy_yaml=session.trust_policy_yaml,
            component_map_yaml=session.component_map_yaml,
            schema_hints_yaml=session.schema_hints_yaml,
        )
        click.echo(f"\nArtifacts written:")
        for p in written:
            click.echo(f"  {p}")

        # Publish artifacts to kindex (if available)
        _kindex_publish_artifacts(session)


@click.group(cls=SafeGroup, invoke_without_command=True)
@_round_options
@click.option(
    "--prime", "-p", multiple=True, type=click.Path(exists=True),
    help="Document(s) to ingest before the interview. Repeat for multiple files.",
)
@click.option(
    "--backend", "-b", default=None,
    help=(
        "LLM backend: anthropic (default), openai, codex, claude, "
        "opencode, cursor. Env: CONSTRAIN_BACKEND."
    ),
)
@click.option(
    "--model", "-m", default=None,
    help="Model name override. Env: CONSTRAIN_MODEL.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum output tokens per backend call. Env: CONSTRAIN_MAX_TOKENS.",
)
@click.pass_context
def cli(
    ctx,
    min_understand,
    max_understand,
    min_challenge,
    max_challenge,
    prime,
    backend,
    model,
    max_tokens,
):
    """Constrain: find the boundary between specification and intent."""
    if backend:
        os.environ["CONSTRAIN_BACKEND"] = backend
    if model:
        os.environ["CONSTRAIN_MODEL"] = model
    if max_tokens is not None:
        if max_tokens < 1:
            raise click.ClickException("max-tokens must be a positive integer.")
        os.environ["CONSTRAIN_MAX_TOKENS"] = str(max_tokens)

    if ctx.invoked_subcommand is not None:
        return

    ensure_api_key()
    config = resolve_config(min_understand, max_understand, min_challenge, max_challenge)
    mgr = SessionManager(Path.cwd())

    incomplete = mgr.find_latest_incomplete()
    if incomplete and not prime:
        click.echo(f"Incomplete session found: {incomplete.id[:8]}...")
        click.echo(f"  Phase: {incomplete.phase.value}")
        click.echo(
            f"  Rounds: {incomplete.understand_rounds} understand, "
            f"{incomplete.challenge_rounds} challenge"
        )
        choice = click.prompt(
            "Resume this session? [y/n/list]",
            type=click.Choice(["y", "n", "list"], case_sensitive=False),
            default="y",
        )
        if choice == "list":
            _do_list(mgr)
            return
        if choice == "y":
            _run_engine(
                incomplete,
                mgr,
                config,
                backend_name=backend,
                model=model,
                max_tokens=max_tokens,
            )
            return

    session = mgr.create()
    mgr.save(session)
    _run_engine(
        session,
        mgr,
        config,
        prime_paths=[Path(p) for p in prime],
        backend_name=backend,
        model=model,
        max_tokens=max_tokens,
    )


@cli.command("new")
@_round_options
@click.option(
    "--prime", "-p", multiple=True, type=click.Path(exists=True),
    help="Document(s) to ingest before the interview.",
)
def cmd_new(min_understand, max_understand, min_challenge, max_challenge, prime):
    """Start a new session (ignores incomplete sessions)."""
    ensure_api_key()
    config = resolve_config(min_understand, max_understand, min_challenge, max_challenge)
    backend_name = os.environ.get("CONSTRAIN_BACKEND")
    model = os.environ.get("CONSTRAIN_MODEL")
    max_tokens = _resolve_optional_int_env("CONSTRAIN_MAX_TOKENS")
    mgr = SessionManager(Path.cwd())
    session = mgr.create()
    mgr.save(session)
    _run_engine(
        session,
        mgr,
        config,
        prime_paths=[Path(p) for p in prime],
        backend_name=backend_name,
        model=model,
        max_tokens=max_tokens,
    )


@cli.command("resume")
@click.argument("session_id", required=False)
def cmd_resume(session_id):
    """Resume an incomplete session."""
    ensure_api_key()
    mgr = SessionManager(Path.cwd())

    if session_id:
        try:
            session = mgr.load(session_id)
        except FileNotFoundError:
            raise click.ClickException(f"Session '{session_id}' not found.")
        except ValueError as e:
            raise click.ClickException(f"Could not load session '{session_id}': {e}")
        if session.phase == Phase.complete:
            raise click.ClickException(
                f"Session '{session_id}' is already completed. "
                "Use 'constrain show' to view its artifacts."
            )
    else:
        session = mgr.find_latest_incomplete()
        if session is None:
            raise click.ClickException(
                "No incomplete sessions found. Use 'constrain new' to start one."
            )

    config = resolve_config(None, None, None, None)
    backend_name = os.environ.get("CONSTRAIN_BACKEND")
    model = os.environ.get("CONSTRAIN_MODEL")
    max_tokens = _resolve_optional_int_env("CONSTRAIN_MAX_TOKENS")
    _run_engine(session, mgr, config, backend_name=backend_name, model=model, max_tokens=max_tokens)


@cli.command("show")
def cmd_show():
    """Display artifacts from the most recent completed session."""
    mgr = SessionManager(Path.cwd())
    sessions = mgr.list_all()
    completed = [s for s in sessions if s["is_complete"]]
    if not completed:
        raise click.ClickException("No completed sessions found. Use 'constrain' to start one.")

    latest = completed[0]
    session = mgr.load(latest["id"])

    if not session.prompt_md:
        raise click.ClickException(
            f"Artifacts not found in session {session.id[:8]}. They may not have been generated."
        )

    click.echo("=== prompt.md ===\n")
    click.echo(session.prompt_md)
    click.echo("\n=== constraints.yaml ===\n")
    click.echo(session.constraints_yaml)
    if session.trust_policy_yaml:
        click.echo("\n=== trust_policy.yaml ===\n")
        click.echo(session.trust_policy_yaml)
    if session.component_map_yaml:
        click.echo("\n=== component_map.yaml ===\n")
        click.echo(session.component_map_yaml)
    if session.schema_hints_yaml:
        click.echo("\n=== schema_hints.yaml ===\n")
        click.echo(session.schema_hints_yaml)


@cli.command("list")
def cmd_list():
    """List all sessions."""
    mgr = SessionManager(Path.cwd())
    _do_list(mgr)


def _do_list(mgr: SessionManager) -> None:
    sessions = mgr.list_all()
    if not sessions:
        raise click.ClickException("No sessions found. Use 'constrain' to start one.")

    # Table header
    click.echo(f"{'ID':<10} {'PHASE':<12} {'ROUNDS':<10} {'POSTURE':<10} {'UPDATED'}")
    click.echo("-" * 65)
    for s in sessions:
        sid = s["id"][:8]
        phase = s["phase"]
        rounds = s["understand_rounds"] + s["challenge_rounds"]
        posture = "***"
        updated = s["updated_at"][:19]
        click.echo(f"{sid:<10} {phase:<12} {rounds:<10} {posture:<10} {updated}")


@cli.command("export")
@click.option(
    "--format", "-f", "fmt", required=True,
    type=click.Choice(["baton", "pact", "arbiter", "ledger"], case_sensitive=False),
    help="Target format: baton, pact, arbiter, or ledger.",
)
def cmd_export(fmt):
    """Export artifacts as skeletons for downstream tools."""
    import yaml as pyyaml

    mgr = SessionManager(Path.cwd())
    sessions = mgr.list_all()
    completed = [s for s in sessions if s["is_complete"]]
    if not completed:
        raise click.ClickException("No completed sessions found.")

    session = mgr.load(completed[0]["id"])

    # Map format → (output filename, ...)
    _EXPORT_FILES = {
        "baton": ["baton.yaml"],
        "pact": ["task.md"],
        "arbiter": ["arbiter.yaml"],
        "ledger": ["schema_hints.yaml"],
    }

    # Archive any existing export target before writing
    export_file = _EXPORT_FILES[fmt][0]
    cwd = Path.cwd()
    subdir, archived = archive_artifacts(
        cwd,
        [export_file],
        archive_base=_archive_dir(cwd),
        slug_source_priority=[export_file],
    )
    if archived:
        click.echo(f"Archived {export_file} to {subdir.name}/")

    if fmt == "baton":
        if not session.component_map_yaml:
            raise click.ClickException("No component_map.yaml in session. Cannot export baton skeleton.")
        try:
            cm = pyyaml.safe_load(session.component_map_yaml)
        except pyyaml.YAMLError as e:
            raise click.ClickException(f"Invalid component_map.yaml: {e}")

        baton = {
            "version": "1.0",
            "generated_from": "constrain",
            "nodes": [],
            "edges": [],
        }
        for comp in (cm or {}).get("components", []):
            baton["nodes"].append({
                "name": comp.get("name"),
                "type": comp.get("type"),
                "port": comp.get("port"),
                "protocol": comp.get("protocol"),
            })
        for edge in (cm or {}).get("edges", []):
            baton["edges"].append({
                "from": edge.get("from"),
                "to": edge.get("to"),
                "protocol": edge.get("protocol"),
            })

        out = cwd / "baton.yaml"
        out.write_text(pyyaml.dump(baton, default_flow_style=False, sort_keys=False), encoding="utf-8")
        click.echo(f"Written: {out}")

    elif fmt == "pact":
        if not session.prompt_md:
            raise click.ClickException("No prompt.md in session. Cannot export pact skeleton.")
        # Extract system description for the task summary
        pm = session.problem_model
        task_lines = [
            "# Task",
            "",
            f"## System: {pm.system_description or 'TBD'}",
            "",
            "## Goal",
            "",
            "(Fill in the specific implementation goal)",
            "",
            "## Context",
            "",
            "See prompt.md for full system briefing.",
            "",
        ]
        if pm.acceptance_criteria:
            task_lines.append("## Acceptance Criteria")
            task_lines.append("")
            for ac in pm.acceptance_criteria:
                task_lines.append(f"- {ac}")
            task_lines.append("")

        out = cwd / "task.md"
        out.write_text("\n".join(task_lines), encoding="utf-8")
        click.echo(f"Written: {out}")

    elif fmt == "arbiter":
        if not session.trust_policy_yaml:
            raise click.ClickException("No trust_policy.yaml in session. Cannot export arbiter skeleton.")
        try:
            tp = pyyaml.safe_load(session.trust_policy_yaml)
        except pyyaml.YAMLError as e:
            raise click.ClickException(f"Invalid trust_policy.yaml: {e}")

        arbiter = {
            "version": "1.0",
            "generated_from": "constrain",
            "trust": (tp or {}).get("trust", {}),
            "classifications": (tp or {}).get("classifications", []),
            "soak": (tp or {}).get("soak", {}),
            "authority_map": (tp or {}).get("authority_map", []),
            "human_gates": (tp or {}).get("human_gates", {}),
        }
        out = cwd / "arbiter.yaml"
        out.write_text(pyyaml.dump(arbiter, default_flow_style=False, sort_keys=False), encoding="utf-8")
        click.echo(f"Written: {out}")

    elif fmt == "ledger":
        if not session.schema_hints_yaml:
            raise click.ClickException("No schema_hints.yaml in session. Cannot export ledger skeleton.")
        try:
            sh = pyyaml.safe_load(session.schema_hints_yaml)
        except pyyaml.YAMLError as e:
            raise click.ClickException(f"Invalid schema_hints.yaml: {e}")

        # Re-emit the schema_hints as the ledger skeleton
        ledger = {
            "version": "1.0",
            "generated_from": "constrain",
            "storage_backends": (sh or {}).get("storage_backends", []),
            "field_hints": (sh or {}).get("field_hints", []),
        }
        out = cwd / "schema_hints.yaml"
        out.write_text(pyyaml.dump(ledger, default_flow_style=False, sort_keys=False), encoding="utf-8")
        click.echo(f"Written: {out}")


@cli.command("validate")
def cmd_validate():
    """Validate all artifacts for internal consistency."""
    import yaml as pyyaml

    mgr = SessionManager(Path.cwd())
    sessions = mgr.list_all()
    completed = [s for s in sessions if s["is_complete"]]
    if not completed:
        raise click.ClickException("No completed sessions found.")

    session = mgr.load(completed[0]["id"])
    artifacts = SynthesisArtifacts(
        prompt_md=session.prompt_md,
        constraints_yaml=session.constraints_yaml,
        trust_policy_yaml=session.trust_policy_yaml,
        component_map_yaml=session.component_map_yaml,
        schema_hints_yaml=session.schema_hints_yaml,
    )

    errors = []

    # Validate YAML
    for name, content in [
        ("constraints.yaml", artifacts.constraints_yaml),
        ("trust_policy.yaml", artifacts.trust_policy_yaml),
        ("component_map.yaml", artifacts.component_map_yaml),
        ("schema_hints.yaml", artifacts.schema_hints_yaml),
    ]:
        if content:
            try:
                validate_yaml_content(content, name)
                click.echo(f"  {name}: valid YAML")
            except ValueError as e:
                errors.append(str(e))
                click.echo(f"  {name}: INVALID - {e}")
        else:
            click.echo(f"  {name}: (empty)")

    # Cross-validate
    warnings = validate_artifacts(artifacts)
    if warnings:
        click.echo(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            click.echo(f"  - {w}")
    else:
        click.echo("\nNo cross-validation warnings.")

    # Check prompt.md sections
    if artifacts.prompt_md:
        missing_sections = []
        if "## Trust and Authority Model" not in artifacts.prompt_md:
            missing_sections.append("Trust and Authority Model")
        if "## Component Topology" not in artifacts.prompt_md:
            missing_sections.append("Component Topology")
        if missing_sections:
            click.echo(f"\nprompt.md missing sections: {', '.join(missing_sections)}")
        else:
            click.echo("\nprompt.md: all required sections present")

    if errors:
        raise click.ClickException(f"{len(errors)} validation error(s) found.")
    click.echo("\nValidation passed.")


@cli.command("archive")
@click.argument("subcommand", required=False, default="list",
                type=click.Choice(["list", "show"], case_sensitive=False))
@click.argument("slug", required=False, default=None)
def cmd_archive(subcommand, slug):
    """List or inspect archived artifact sessions.

    \b
    constrain archive              List all archived sessions
    constrain archive list         Same as above
    constrain archive show <slug>  Show contents of an archived session
    """
    archive_base = _archive_dir(Path.cwd())
    sessions = list_archived_sessions(archive_base)

    if subcommand == "list":
        if not sessions:
            click.echo("No archived sessions.")
            return
        click.echo(f"{'SLUG':<30} {'FILES':<6} {'PATH'}")
        click.echo("-" * 70)
        for s in sessions:
            click.echo(f"{s['slug']:<30} {len(s['files']):<6} {s['path']}")

    elif subcommand == "show":
        if not slug:
            raise click.ClickException("Specify a slug: constrain archive show <slug>")
        artifacts = load_archived_artifacts(archive_base, slug)
        if not artifacts:
            raise click.ClickException(f"No archived session with slug '{slug}'.")
        for name, content in sorted(artifacts.items()):
            click.echo(f"=== {name} ===\n")
            click.echo(content)
            click.echo()


@cli.command("mcp-server")
@click.option("--project-dir", type=click.Path(exists=True), default=None,
              help="Project directory (default: cwd).")
def cmd_mcp_server(project_dir):
    """Run the Constrain MCP server (stdio transport)."""
    if project_dir:
        os.environ["CONSTRAIN_PROJECT_DIR"] = str(project_dir)
    from .mcp_server import main as mcp_main
    mcp_main()


def main() -> None:
    """Entry point for the constrain CLI."""
    cli()
