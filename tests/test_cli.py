"""Tests for the CLI interface."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cli_env(tmp_dir, sample_docs):
    """Set up CLI test environment."""
    return {
        "docs_dir": str(sample_docs),
        "index_dir": str(tmp_dir / ".ts"),
        "tmp_dir": tmp_dir,
    }


def run_cli(*args, cwd=None):
    """Run tokenshrink CLI and return result."""
    cmd = [sys.executable, "-m", "tokenshrink.cli"] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=120,
    )
    return result


class TestCLIIndex:
    """Test CLI index command."""

    def test_index_basic(self, cli_env):
        r = run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        assert r.returncode == 0
        assert "Indexed" in r.stdout
        assert "Chunks" in r.stdout

    def test_index_json_output(self, cli_env):
        r = run_cli("--index-dir", cli_env["index_dir"], "--json", "index", cli_env["docs_dir"])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "files_indexed" in data
        assert "chunks_added" in data
        assert data["files_indexed"] > 0

    def test_index_with_extensions(self, cli_env):
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "--json",
            "index", cli_env["docs_dir"],
            "-e", ".md",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["files_indexed"] == 4  # Only .md files

    def test_index_force(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "--json",
            "index", cli_env["docs_dir"],
            "-f",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["files_indexed"] > 0  # Re-indexed with force


class TestCLIQuery:
    """Test CLI query command."""

    def test_query_basic(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "query", "authentication tokens",
            "--no-compress",
        )
        assert r.returncode == 0
        assert "Sources:" in r.stdout or "No relevant" in r.stdout

    def test_query_json(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "--json",
            "query", "authentication",
            "--no-compress",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "context" in data
        assert "sources" in data
        assert "original_tokens" in data

    def test_query_with_scores(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "query", "rate limits",
            "--no-compress",
            "--scores",
        )
        assert r.returncode == 0
        assert "Chunk Importance Scores" in r.stdout
        assert "sim=" in r.stdout
        assert "density=" in r.stdout

    def test_query_json_with_scores(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "--json",
            "query", "deployment kubernetes",
            "--no-compress",
            "--scores",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "chunk_scores" in data
        for cs in data["chunk_scores"]:
            assert "similarity" in cs
            assert "density" in cs
            assert "importance" in cs

    def test_query_no_dedup(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "--json",
            "query", "authentication",
            "--no-compress",
            "--no-dedup",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["dedup_removed"] == 0

    def test_query_k_param(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "--json",
            "query", "authentication",
            "--no-compress",
            "-k", "2",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert len(data.get("sources", [])) <= 2


class TestCLISearch:
    """Test CLI search command."""

    def test_search_basic(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "search", "rate limits",
        )
        assert r.returncode == 0
        assert "score:" in r.stdout

    def test_search_json(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "--json",
            "search", "rate limits",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_search_empty_index(self, cli_env):
        r = run_cli(
            "--index-dir", cli_env["index_dir"],
            "search", "anything",
        )
        assert r.returncode == 0
        assert "No results" in r.stdout


class TestCLIStats:
    """Test CLI stats command."""

    def test_stats_empty(self, cli_env):
        r = run_cli("--index-dir", cli_env["index_dir"], "stats")
        assert r.returncode == 0
        assert "Chunks: 0" in r.stdout

    def test_stats_after_index(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli("--index-dir", cli_env["index_dir"], "stats")
        assert r.returncode == 0
        assert "Chunks:" in r.stdout
        assert "Files:" in r.stdout

    def test_stats_json(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli("--index-dir", cli_env["index_dir"], "--json", "stats")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "total_chunks" in data
        assert data["total_chunks"] > 0


class TestCLIClear:
    """Test CLI clear command."""

    def test_clear(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli("--index-dir", cli_env["index_dir"], "clear")
        assert r.returncode == 0
        assert "cleared" in r.stdout.lower()

    def test_clear_json(self, cli_env):
        run_cli("--index-dir", cli_env["index_dir"], "index", cli_env["docs_dir"])
        r = run_cli("--index-dir", cli_env["index_dir"], "--json", "clear")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["status"] == "cleared"


class TestCLIMisc:
    """Test miscellaneous CLI behavior."""

    def test_version(self):
        r = run_cli("--version")
        assert r.returncode == 0
        assert "0.2.0" in r.stdout

    def test_no_command(self):
        r = run_cli()
        assert r.returncode == 0  # Just prints help

    def test_help(self):
        r = run_cli("--help")
        assert r.returncode == 0
        assert "tokenshrink" in r.stdout.lower()
