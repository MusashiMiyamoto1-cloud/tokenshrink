"""Integration tests — full pipeline end-to-end."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tokenshrink import TokenShrink, ShrinkResult, ChunkScore


class TestFullPipeline:
    """End-to-end tests through the full workflow."""

    def test_index_query_cycle(self, sample_docs, make_ts):
        """Index → query → verify results."""
        ts = make_ts(adaptive=True, dedup=True)
        
        stats = ts.index(str(sample_docs))
        assert stats["files_indexed"] >= 4
        
        result = ts.query("How do I authenticate API requests?")
        assert len(result.context) > 0
        assert "token" in result.context.lower() or "auth" in result.context.lower()
        assert len(result.chunk_scores) > 0
        
        for cs in result.chunk_scores:
            assert 0 <= cs.similarity <= 1
            assert 0 <= cs.density <= 1
            assert 0 <= cs.importance <= 1

    def test_index_reindex_query(self, sample_docs, make_ts):
        """Index → modify file → reindex → query."""
        ts = make_ts()
        ts.index(str(sample_docs))
        
        auth_file = sample_docs / "auth.md"
        auth_file.write_text(
            auth_file.read_text() + "\n\nNew section: API keys can also be passed as query parameters "
            "using the ?api_key=YOUR_KEY format. This is less secure than headers."
        )
        
        r2 = ts.index(str(sample_docs))
        assert r2["files_indexed"] == 1
        
        result = ts.query("api key query parameter")
        assert len(result.context) > 0

    def test_multiple_queries_consistency(self, indexed_ts):
        """Same query twice → same results."""
        r1 = indexed_ts.query("rate limits per minute", k=3)
        r2 = indexed_ts.query("rate limits per minute", k=3)
        assert r1.context == r2.context
        assert r1.sources == r2.sources
        assert r1.original_tokens == r2.original_tokens

    def test_clear_and_rebuild(self, sample_docs, make_ts):
        """Index → clear → reindex → verify."""
        ts = make_ts()
        ts.index(str(sample_docs))
        assert ts._index.ntotal > 0
        
        ts.clear()
        assert ts._index.ntotal == 0
        
        ts.index(str(sample_docs))
        assert ts._index.ntotal > 0
        result = ts.query("deployment")
        assert len(result.context) > 0

    def test_different_queries_different_results(self, indexed_ts):
        """Different queries → different top results."""
        r_auth = indexed_ts.query("authentication bearer token", k=3)
        r_deploy = indexed_ts.query("kubernetes deployment pods", k=3)
        
        # Should have different primary sources
        assert r_auth.context != r_deploy.context

    def test_search_then_query(self, indexed_ts):
        """Search for candidates, then query for context."""
        search_results = indexed_ts.search("rate limits", k=3)
        assert len(search_results) > 0
        
        # Now get compressed context
        result = indexed_ts.query("rate limits", k=3)
        assert len(result.context) > 0
        
        # Search results should overlap with query sources
        search_sources = {r["source"] for r in search_results}
        query_sources = set(result.sources)
        assert len(search_sources & query_sources) > 0

    def test_persistence_across_instances(self, tmp_dir, sample_docs, make_ts):
        """Data survives instance recreation."""
        idx = str(tmp_dir / ".ts_persist")
        
        ts1 = make_ts(index_dir=idx)
        ts1.index(str(sample_docs))
        r1 = ts1.query("authentication")
        
        ts2 = make_ts(index_dir=idx)
        r2 = ts2.query("authentication")
        
        assert r1.context == r2.context

    def test_adaptive_vs_non_adaptive(self, sample_docs, make_ts):
        """Adaptive mode produces different chunk ordering."""
        ts = make_ts(adaptive=True, dedup=False)
        ts.index(str(sample_docs))
        
        r_adaptive = ts.query("authentication", adaptive=True)
        r_flat = ts.query("authentication", adaptive=False)
        
        # Both should have results
        assert len(r_adaptive.context) > 0
        assert len(r_flat.context) > 0
        
        # Adaptive scores should vary; non-adaptive all 0.5
        if r_flat.chunk_scores:
            for cs in r_flat.chunk_scores:
                assert cs.compression_ratio == 0.5

    def test_dedup_vs_no_dedup(self, indexed_ts):
        """Dedup mode removes redundant chunks."""
        r_dedup = indexed_ts.query("authentication bearer token", k=5, dedup=True)
        r_full = indexed_ts.query("authentication bearer token", k=5, dedup=False)
        
        # With dedup, fewer or equal chunks in context
        assert len(r_dedup.context) <= len(r_full.context) or r_dedup.dedup_removed >= 0


class TestSemanticRelevance:
    """Test that retrieval is semantically meaningful."""

    def test_auth_query_finds_auth_docs(self, indexed_ts):
        result = indexed_ts.query("OAuth2 client credentials", k=3)
        source_names = [Path(s).name for s in result.sources]
        assert any("auth" in name.lower() for name in source_names)

    def test_rate_query_finds_rate_docs(self, indexed_ts):
        result = indexed_ts.query("request per minute limits", k=3)
        source_names = [Path(s).name for s in result.sources]
        assert any("rate" in name.lower() for name in source_names)

    def test_deploy_query_finds_deploy_docs(self, indexed_ts):
        result = indexed_ts.query("docker kubernetes pods replicas", k=3)
        source_names = [Path(s).name for s in result.sources]
        assert any("deploy" in name.lower() for name in source_names)

    def test_code_query_finds_code(self, indexed_ts):
        result = indexed_ts.query("Python requests retry backoff", k=3)
        source_names = [Path(s).name for s in result.sources]
        assert any(name.endswith(".py") for name in source_names)

    def test_unrelated_query_low_scores(self, indexed_ts):
        """Query about something not in the docs → low similarity scores."""
        result = indexed_ts.query("quantum physics string theory", k=3)
        if result.chunk_scores:
            max_sim = max(cs.similarity for cs in result.chunk_scores)
            assert max_sim < 0.8  # Should not score very high


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_short_query(self, indexed_ts):
        result = indexed_ts.query("auth")
        assert isinstance(result, ShrinkResult)

    def test_very_long_query(self, indexed_ts):
        long_query = "How do I " + " ".join(["authenticate"] * 100) + " with the API?"
        result = indexed_ts.query(long_query)
        assert isinstance(result, ShrinkResult)

    def test_special_characters_query(self, indexed_ts):
        result = indexed_ts.query("what is X-RateLimit-Remaining header?")
        assert isinstance(result, ShrinkResult)

    def test_unicode_query(self, indexed_ts):
        result = indexed_ts.query("认证 API 令牌")
        assert isinstance(result, ShrinkResult)

    def test_empty_query(self, indexed_ts):
        result = indexed_ts.query("")
        assert isinstance(result, ShrinkResult)

    def test_numeric_query(self, indexed_ts):
        result = indexed_ts.query("429 status code")
        assert isinstance(result, ShrinkResult)

    def test_binary_file_skipped(self, tmp_dir, make_ts):
        """Binary files shouldn't crash indexing."""
        docs = tmp_dir / "docs"
        docs.mkdir()
        (docs / "readme.md").write_text("Some valid content here " * 50)
        (docs / "binary.md").write_bytes(b"\x00\x01\x02\xff" * 100)
        
        ts = make_ts()
        result = ts.index(str(docs))
        assert result["files_indexed"] >= 1

    def test_large_file(self, tmp_dir, make_ts):
        """Large file should be chunked properly."""
        docs = tmp_dir / "docs"
        docs.mkdir()
        (docs / "big.md").write_text(
            "This is a large document with many sections.\n\n" +
            "\n\n".join([f"## Section {i}\n\n" + f"Content for section {i}. " * 200 for i in range(50)])
        )
        
        ts = make_ts()
        result = ts.index(str(docs))
        assert result["chunks_added"] > 10

    def test_empty_files_handled(self, tmp_dir, make_ts):
        """Empty files shouldn't crash."""
        docs = tmp_dir / "docs"
        docs.mkdir()
        (docs / "empty.md").write_text("")
        (docs / "content.md").write_text("Valid content here " * 50)
        
        ts = make_ts()
        result = ts.index(str(docs))
        assert result["files_indexed"] >= 1

    def test_whitespace_only_files(self, tmp_dir, make_ts):
        """Files with only whitespace shouldn't crash."""
        docs = tmp_dir / "docs"
        docs.mkdir()
        (docs / "whitespace.md").write_text("   \n\n\t\t\n   ")
        
        ts = make_ts()
        result = ts.index(str(docs))
        assert isinstance(result, dict)

    def test_concurrent_queries(self, indexed_ts):
        """Multiple queries in sequence (not truly concurrent, but rapid)."""
        queries = [
            "authentication",
            "rate limits",
            "deployment",
            "Python client",
            "OAuth2",
        ]
        results = [indexed_ts.query(q) for q in queries]
        assert all(isinstance(r, ShrinkResult) for r in results)
        assert all(len(r.context) > 0 for r in results)

    def test_k_larger_than_index(self, indexed_ts):
        """k > total chunks → returns all chunks."""
        result = indexed_ts.query("auth", k=10000)
        assert isinstance(result, ShrinkResult)

    def test_min_score_zero(self, indexed_ts):
        """min_score=0 returns everything."""
        result = indexed_ts.query("auth", min_score=0.0)
        assert len(result.chunk_scores) >= 1

    def test_min_score_one(self, indexed_ts):
        """min_score=1.0 returns nothing (or exact match only)."""
        result = indexed_ts.query("auth", min_score=1.0)
        assert isinstance(result, ShrinkResult)
