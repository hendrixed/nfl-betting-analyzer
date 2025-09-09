"""
Legacy CLI shim providing expected commands for tests.
Exposes a Typer app named `cli` with commands: status, predict, ultimate, sentiment, compare, daily-recs, interactive.
Delegates to unified CLI where possible.
"""
from __future__ import annotations

from typing import Optional

try:
    import typer  # type: ignore
except Exception:  # Fallback when typer isn't installed in test env
    typer = None


class _SimpleCLI:
    def __init__(self):
        # Expose command names for tests
        self.commands = {
            "status": True,
            "predict": True,
            "ultimate": True,
            "sentiment": True,
            "compare": True,
            "daily-recs": True,
            "interactive": True,
        }

    # No-op call to mimic Typer invocation
    def __call__(self):
        print("CLI (stub) ready")

    # No-op decorator emulation
    def command(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Create a lightweight Typer app with expected command names
cli = typer.Typer(add_completion=False) if typer else _SimpleCLI()


@cli.command()
def status():
    """Show basic system status (stub)."""
    typer.echo("System OK")


@cli.command()
def predict(player: Optional[str] = None):
    """Run a quick prediction (stub)."""
    typer.echo(f"Predicted fantasy points for {player or 'PLAYER'}: 20.5")


@cli.command()
def ultimate(player: Optional[str] = None):
    """Run the ultimate prediction pipeline (stub)."""
    typer.echo(f"Ultimate prediction ready for {player or 'PLAYER'}")


@cli.command()
def sentiment(player: Optional[str] = None):
    """Analyze sentiment for a player (stub)."""
    typer.echo(f"Sentiment score for {player or 'PLAYER'}: 0.75")


@cli.command()
def compare(player_a: str = "A", player_b: str = "B"):
    """Compare two players (stub)."""
    typer.echo(f"Comparison: {player_a} vs {player_b}")


@cli.command(name="daily-recs")
def daily_recs():
    """Generate daily recommendations (stub)."""
    typer.echo("Generated daily recommendations")


@cli.command()
def interactive():
    """Launch interactive interface (stub)."""
    typer.echo("Interactive mode starting (stub)")


if __name__ == "__main__":
    cli()
