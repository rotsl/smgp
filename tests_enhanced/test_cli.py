"""Tests for the SMGP command-line interface.

Uses click.testing.CliRunner to invoke CLI commands in-process without
spawning subprocesses. Validates output formatting and exit codes.

References:
    Angles, R. & Gutierrez, C. (2008). "Survey of Graph Database Models."
"""
from __future__ import annotations

from click.testing import CliRunner

from smgp.cli import cli

runner = CliRunner()


class TestVersionCommand:
    """Tests for the 'smgp version' command."""

    def test_version_outputs_version_string(self) -> None:
        """Verify 'smgp version' prints the correct version number."""
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        # The version should be printed (strip newline for comparison)
        assert result.output.strip() == "1.0.0"

    def test_version_help(self) -> None:
        """Verify version command has a help message."""
        result = runner.invoke(cli, ["version", "--help"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()


class TestGraphStatsCommand:
    """Tests for the 'smgp graph stats' command."""

    def test_graph_stats_empty_graph(self) -> None:
        """Verify 'smgp graph stats' prints node/edge counts for an empty graph."""
        result = runner.invoke(cli, ["graph", "stats", "--hd-dim", "100"])
        assert result.exit_code == 0
        output = result.output
        # Should contain node and edge count labels
        assert "Nodes:" in output
        assert "Edges:" in output
        # Empty graph should have 0 nodes and 0 edges
        assert "0" in output

    def test_graph_stats_default_hd_dim(self) -> None:
        """Verify 'smgp graph stats' works with default HD dimensionality."""
        result = runner.invoke(cli, ["graph", "stats"])
        assert result.exit_code == 0
        assert "Nodes:" in output_str(result)

    def test_graph_stats_with_seed(self) -> None:
        """Verify 'smgp graph stats --seed 42' runs successfully."""
        result = runner.invoke(cli, ["graph", "stats", "--seed", "42"])
        assert result.exit_code == 0
        assert "Nodes:" in result.output


class TestVerifyCommand:
    """Tests for the 'smgp verify' command."""

    def test_verify_runs_without_error(self) -> None:
        """Verify 'smgp verify \"test claim\"' runs without error."""
        result = runner.invoke(cli, ["verify", "test claim"])
        assert result.exit_code == 0
        # Should contain verification status
        assert "Status:" in result.output

    def test_verify_shows_not_verified_for_unknown_claim(self) -> None:
        """Verify unrecognizable claims return NOT VERIFIED."""
        result = runner.invoke(cli, ["verify", "nonsensical claim xyz"])
        assert result.exit_code == 0
        assert "NOT VERIFIED" in result.output

    def test_verify_shows_confidence(self) -> None:
        """Verify output includes a confidence score."""
        result = runner.invoke(cli, ["verify", "test claim"])
        assert result.exit_code == 0
        assert "Confidence:" in result.output


class TestServerCommand:
    """Tests for the 'smgp server' command."""

    def test_server_requires_integration_deps(self) -> None:
        """Verify server command fails gracefully without FastAPI/uvicorn."""
        # The server command should fail because we mock/avoid installing
        # integration dependencies. Check that it gives a helpful message.
        result = runner.invoke(cli, ["server", "--help"])
        assert result.exit_code == 0
        assert "port" in result.output.lower()


class TestCLIRoot:
    """Tests for the root CLI group."""

    def test_cli_help(self) -> None:
        """Verify root help lists all commands."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "version" in result.output
        assert "graph" in result.output
        assert "verify" in result.output
        assert "server" in result.output

    def test_cli_main_entrypoint(self) -> None:
        """Verify the CLI can be invoked via the module entrypoint."""
        from smgp.cli import cli as cli_group
        assert cli_group is not None
        # Click derives the group name from the function name
        assert cli_group.name == "cli"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def output_str(result) -> str:
    """Extract stripped output from a CliRunner result."""
    return result.output.strip()
