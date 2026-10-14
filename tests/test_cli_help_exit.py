"""Subcommand --help is click's normal exit, not an error (SafeGroup must not catch it)."""

import pytest
from click.testing import CliRunner

from constrain.cli import cli


@pytest.mark.parametrize("args", [
    ["--help"],
    ["validate", "--help"],
    ["boundary", "--help"],
    ["boundary", "verify", "--help"],
    ["export", "--help"],
])
def test_help_exits_zero_without_error(args):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "Error" not in result.output
    assert "Usage:" in result.output
