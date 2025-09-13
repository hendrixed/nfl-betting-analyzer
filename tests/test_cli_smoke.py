#!/usr/bin/env python3
from typer.testing import CliRunner
from nfl_cli import app


def test_cli_lists_new_ingest_helpers_in_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])  # do not execute network or DB ops
    assert result.exit_code == 0
    out = result.stdout
    assert "snapshot-and-ingest-week" in out
    assert "daily-ingest" in out
