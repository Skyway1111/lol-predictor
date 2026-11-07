"""The ``lol-predict`` command: dashboard, update, retrain, upcoming."""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import click

from league_predictor import config
from league_predictor.artifacts import TOKENS

# Root logging config for every CLI run: downstream domain modules log to the console.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _package_version() -> str:
    """The installed ``league-predictor`` version, or ``0.0.0+unknown`` outside an install."""
    try:
        return _pkg_version("league-predictor")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@click.group()
@click.version_option(version=_package_version(), prog_name="lol-predict")
def main():
    """League of Legends Professional Match Predictor."""


def _default_db() -> str:
    """Click default resolved per invocation, so a patched ``config.DB_PATH`` is honoured."""
    return str(config.DB_PATH)


_db_option = click.option(
    "--db",
    default=_default_db,
    type=str,
    show_default="configured DB_PATH",
    help="Path to the SQLite database file.",
)


@main.command()
@click.option(
    "--port",
    default=lambda: config.DASHBOARD_DEFAULT_PORT,
    type=int,
    show_default="configured dashboard port",
    help="Port to run the dashboard on.",
)
@_db_option
@click.option(
    "--public",
    is_flag=True,
    help="Bind the dashboard on 0.0.0.0 (all interfaces) instead of "
    "loopback only. Also enabled via LOL_PREDICT_PUBLIC.",
)
@click.option("--open", "open_browser", is_flag=True, help="Open it in the browser once it is up.")
def dashboard(port: int, db: str, *, public: bool, open_browser: bool):
    """Start the read-only web dashboard; binds loopback unless --public / LOL_PREDICT_PUBLIC."""
    from league_predictor.orchestrator import run_orchestrator

    run_orchestrator(port=port, public=public, db_path=Path(db), open_browser=open_browser)


_allow_dirty = click.option(
    "--allow-dirty-ingest",
    is_flag=True,
    help="Continue past ingestion errors or unresolved player-identity collisions.",
)


@main.command()
@_db_option
@click.option("--full", is_flag=True, help="Re-download every configured year, not just this one.")
@_allow_dirty
def update(db: str, *, full: bool, allow_dirty_ingest: bool):
    """Pull new games, score them with the current models, rebuild the dashboard. ~8 min."""
    from league_predictor.orchestrator.jobs import update as run_update

    run_update(Path(db), full=full, allow_dirty=allow_dirty_ingest, progress=click.echo)
    click.echo("Update complete.")


@main.command()
@click.argument("models", nargs=-1, type=click.Choice(TOKENS))
@_db_option
@_allow_dirty
def retrain(models: tuple[str, ...], db: str, *, allow_dirty_ingest: bool):
    """Pull new games and refit MODELS (default: all four), then rebuild the dashboard. ~40 min."""
    from league_predictor.orchestrator.jobs import retrain as run_retrain

    run_retrain(models or TOKENS, Path(db), allow_dirty=allow_dirty_ingest, progress=click.echo)
    click.echo("Retrain complete.")


@main.command()
@_db_option
def upcoming(db: str):
    """Sync upcoming matches from PandaScore and predict them. Needs a PANDA key in panda.env."""
    from league_predictor.orchestrator.jobs import sync_upcoming

    result = sync_upcoming(Path(db), progress=click.echo)
    click.echo("Upcoming sync complete.")
    for key in ("fetched", "upserted", "paired", "canceled", "predicted"):
        click.echo(f"  {key.capitalize():<10} {result[key]}")
    if result["unmatched_teams"]:
        click.echo("  Unmatched PandaScore teams (no prediction until mapped):")
        for name in result["unmatched_teams"]:
            click.echo(f"    - {name}")
        click.echo("  Map one with: lol-predict upcoming-map-team <ps_team_id> <team_id>")


@main.command("upcoming-map-team")
@click.argument("ps_team_id", type=int)
@click.argument("team_id")
@_db_option
def upcoming_map_team(ps_team_id: int, team_id: str, db: str):
    """Map an unmatched PandaScore team onto a teams.team_id, once."""
    from league_predictor.orchestrator.jobs import map_upcoming_team

    try:
        map_upcoming_team(ps_team_id, team_id, Path(db))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Mapped ps_team_id {ps_team_id} -> {team_id}.")


if __name__ == "__main__":
    main()
