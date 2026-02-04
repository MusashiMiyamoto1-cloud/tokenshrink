"""Tests for the TokenShrink pipeline class."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tokenshrink import TokenShrink, ShrinkResult, ChunkScore


class TestTokenShrinkInit:
    """Test initialization and configuration."""

    def test_default_init(self, make_ts):
        ts = make_ts()
        assert ts.chunk_size == 512
        assert ts.chunk_overlap == 50
        assert ts._adaptive is True
        assert ts._dedup is True
        assert ts._dedup_threshold == 0.85

    def test_custom_params(self, make_ts):
        ts = make_ts(
            chunk_size=256,
            chunk_overlap=25,
            adaptive=False,
            dedup=False,
            dedup_threshold=0.9,
        )
        assert ts.chunk_size == 256
        assert ts.chunk_overlap == 25
        assert ts._adaptive is False
        assert ts._dedup is False
        assert ts._dedup_threshold == 0.9

    def test_index_dir_created_on_save(self, tmp_dir, sample_docs, make_ts):
        idx_dir = tmp_dir / ".ts2"
        ts = make_ts(index_dir=str(idx_dir))
        assert not idx_dir.exists()
        ts.index(str(sample_docs))
        assert idx_dir.exists()
        assert (idx_dir / "index.faiss").exists()
        assert (idx_dir / "meta.json").exists()

    def test_device_auto(self, make_ts):
        ts = make_ts(device="auto")
        assert ts._device in ("mps", "cuda", "cpu")

    def test_device_explicit(self, make_ts):
        ts = make_ts(device="cpu")
        assert ts._device == "cpu"


class TestIndexing:
    """Test file indexing."""

    def test_index_directory(self, sample_docs, make_ts):
        ts = make_ts()
        result = ts.index(str(sample_docs))
        assert result["files_indexed"] > 0
        assert result["chunks_added"] > 0
        assert result["total_chunks"] > 0

    def test_index_single_file(self, sample_docs, make_ts):
        ts = make_ts()
        result = ts.index(str(sample_docs / "auth.md"))
        assert result["files_indexed"] == 1
        assert result["chunks_added"] >= 1

    def test_index_extensions_filter(self, sample_docs, make_ts):
        ts = make_ts()
        result = ts.index(str(sample_docs), extensions=(".md",))
        assert result["files_indexed"] == 4  # auth, auth2, rate-limits, deployment

    def test_index_py_files(self, sample_docs, make_ts):
        ts = make_ts()
        result = ts.index(str(sample_docs), extensions=(".py",))
        assert result["files_indexed"] == 1  # client.py

    def test_reindex_unchanged_files_skipped(self, sample_docs, make_ts):
        ts = make_ts()
        ts.index(str(sample_docs))
        r2 = ts.index(str(sample_docs))
        assert r2["files_indexed"] == 0

    def test_reindex_with_force(self, sample_docs, make_ts):
        ts = make_ts()
        ts.index(str(sample_docs))
        r2 = ts.index(str(sample_docs), force=True)
        assert r2["files_indexed"] > 0

    def test_index_empty_dir(self, tmp_dir, make_ts):
        empty = tmp_dir / "empty"
        empty.mkdir()
        ts = make_ts()
        result = ts.index(str(empty))
        assert result["files_indexed"] == 0
        assert result["chunks_added"] == 0

    def test_index_nonexistent_extension(self, sample_docs, make_ts):
        ts = make_ts()
        result = ts.index(str(sample_docs), extensions=(".xyz",))
        assert result["files_indexed"] == 0

    def test_index_persists(self, tmp_dir, sample_docs, make_ts):
        idx_dir = str(tmp_dir / ".ts_persist")
        ts1 = make_ts(index_dir=idx_dir)
        ts1.index(str(sample_docs))
        total = ts1._index.ntotal

        # New instance loads from disk (needs its own model load for persistence test)
        ts2 = make_ts(index_dir=idx_dir)
        assert ts2._index.ntotal == total
        assert len(ts2._chunks) == len(ts1._chunks)

    def test_skip_hidden_files(self, tmp_dir, make_ts):
        docs = tmp_dir / "docs"
        docs.mkdir()
        (docs / ".hidden.md").write_text("Hidden content " * 100)
        (docs / "visible.md").write_text("Visible content " * 100)

        ts = make_ts()
        result = ts.index(str(docs))
        assert result["files_indexed"] == 1

    def test_skip_venv_dirs(self, tmp_dir, make_ts):
        docs = tmp_dir / "project"
        docs.mkdir()
        (docs / "readme.md").write_text("Project readme " * 100)
        venv = docs / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "stuff.py").write_text("venv code " * 100)

        ts = make_ts()
        result = ts.index(str(docs))
        assert result["files_indexed"] == 1


class TestChunking:
    """Test text chunking logic."""

    def test_small_text_no_chunks(self, make_ts):
        ts = make_ts()
        chunks = ts._chunk_text("too short", "test.md")
        assert len(chunks) == 0

    def test_minimum_chunk_size(self, make_ts):
        ts = make_ts()
        text = " ".join(["word"] * 25)
        chunks = ts._chunk_text(text, "test.md")
        assert len(chunks) >= 1

    def test_overlap(self, make_ts):
        ts = make_ts(chunk_size=100, chunk_overlap=20)
        text = " ".join(["word"] * 250)
        chunks = ts._chunk_text(text, "test.md")
        assert len(chunks) >= 3

    def test_source_preserved(self, make_ts):
        ts = make_ts()
        text = " ".join(["word"] * 100)
        chunks = ts._chunk_text(text, "/path/to/test.md")
        for c in chunks:
            assert c["source"] == "/path/to/test.md"
            assert "offset" in c

    def test_chunk_has_text(self, make_ts):
        ts = make_ts()
        text = "The quick brown fox " * 30
        chunks = ts._chunk_text(text, "test.md")
        for c in chunks:
            assert len(c["text"]) > 0


class TestQuery:
    """Test query and retrieval."""

    def test_basic_query(self, indexed_ts):
        result = indexed_ts.query("How does authentication work?")
        assert isinstance(result, ShrinkResult)
        assert len(result.context) > 0
        assert len(result.sources) > 0

    def test_query_returns_relevant(self, indexed_ts):
        result = indexed_ts.query("rate limits per minute")
        assert "rate" in result.context.lower() or "limit" in result.context.lower()

    def test_empty_index_query(self, tmp_dir):
        ts = TokenShrink(index_dir=str(tmp_dir / ".ts"), compression=False)
        result = ts.query("anything")
        assert result.context == ""
        assert result.sources == []
        assert result.ratio == 1.0

    def test_query_k_param(self, indexed_ts):
        r1 = indexed_ts.query("authentication", k=1)
        r5 = indexed_ts.query("authentication", k=5)
        # More chunks → more context (usually)
        assert len(r5.context) >= len(r1.context)

    def test_query_min_score(self, indexed_ts):
        # Very high min_score → fewer results
        result = indexed_ts.query("authentication", min_score=0.99)
        # May return empty if nothing scores that high
        assert isinstance(result, ShrinkResult)

    def test_query_scores_populated(self, indexed_ts):
        result = indexed_ts.query("authentication tokens expire")
        assert len(result.chunk_scores) > 0
        for cs in result.chunk_scores:
            assert isinstance(cs, ChunkScore)
            assert 0.0 <= cs.similarity <= 1.0
            assert 0.0 <= cs.density <= 1.0
            assert 0.0 <= cs.importance <= 1.0
            assert 0.2 <= cs.compression_ratio <= 0.9

    def test_query_no_compression(self, indexed_ts):
        result = indexed_ts.query("authentication", compress=False)
        assert result.ratio == 1.0
        assert result.original_tokens == result.compressed_tokens

    def test_query_dedup_disabled(self, indexed_ts):
        result = indexed_ts.query("authentication", dedup=False)
        assert result.dedup_removed == 0

    def test_query_adaptive_disabled(self, indexed_ts):
        result = indexed_ts.query("authentication", adaptive=False)
        # All chunk scores should have default ratio
        for cs in result.chunk_scores:
            assert cs.compression_ratio == 0.5

    def test_query_sources_are_paths(self, indexed_ts):
        result = indexed_ts.query("deployment kubernetes")
        for src in result.sources:
            assert Path(src).name.endswith((".md", ".py", ".txt"))

    def test_savings_string(self, indexed_ts):
        result = indexed_ts.query("authentication")
        savings = result.savings
        assert "Saved" in savings
        assert "tokens" in savings


class TestSearch:
    """Test search (no compression)."""

    def test_search_returns_list(self, indexed_ts):
        results = indexed_ts.search("rate limits")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_has_scores(self, indexed_ts):
        results = indexed_ts.search("authentication")
        for r in results:
            assert "score" in r
            assert "text" in r
            assert "source" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_search_ordered_by_score(self, indexed_ts):
        results = indexed_ts.search("authentication", k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_index(self, tmp_dir):
        ts = TokenShrink(index_dir=str(tmp_dir / ".ts"), compression=False)
        assert ts.search("anything") == []

    def test_search_min_score(self, indexed_ts):
        all_results = indexed_ts.search("authentication", min_score=0.0)
        filtered = indexed_ts.search("authentication", min_score=0.5)
        assert len(filtered) <= len(all_results)


class TestDeduplication:
    """Test cross-passage deduplication with real embeddings."""

    def test_dedup_with_similar_docs(self, indexed_ts):
        result = indexed_ts.query("Bearer token authentication API", k=5, dedup=True)
        assert isinstance(result.dedup_removed, int)

    def test_dedup_removes_less_with_high_threshold(self, sample_docs, make_ts):
        ts = make_ts(dedup_threshold=0.99)
        ts.index(str(sample_docs))
        result = ts.query("authentication", k=5, dedup=True)
        assert result.dedup_removed == 0 or result.dedup_removed < 2

    def test_dedup_marks_scores(self, indexed_ts):
        result = indexed_ts.query("authentication", k=5, dedup=True)
        deduped_scores = [cs for cs in result.chunk_scores if cs.deduplicated]
        assert len(deduped_scores) == result.dedup_removed


class TestStats:
    """Test stats reporting."""

    def test_stats_empty(self, make_ts):
        ts = make_ts()
        stats = ts.stats()
        assert stats["total_chunks"] == 0
        assert stats["total_files"] == 0

    def test_stats_after_index(self, indexed_ts):
        stats = indexed_ts.stats()
        assert stats["total_chunks"] > 0
        assert stats["total_files"] > 0
        assert stats["compression_available"] in (True, False)
        assert "device" in stats

    def test_stats_index_dir(self, indexed_ts):
        stats = indexed_ts.stats()
        assert "index_dir" in stats


class TestClear:
    """Test index clearing."""

    def test_clear_removes_data(self, indexed_ts):
        assert indexed_ts._index.ntotal > 0
        indexed_ts.clear()
        assert indexed_ts._index.ntotal == 0
        assert indexed_ts._chunks == []
        assert indexed_ts._file_hashes == {}

    def test_clear_removes_dir(self, tmp_dir, sample_docs, make_ts):
        idx_dir = tmp_dir / ".ts_clear"
        ts = make_ts(index_dir=str(idx_dir))
        ts.index(str(sample_docs))
        assert idx_dir.exists()
        ts.clear()
        assert not idx_dir.exists()


class TestShrinkResult:
    """Test ShrinkResult dataclass."""

    def test_savings_property(self):
        r = ShrinkResult(
            context="test",
            sources=[],
            original_tokens=1000,
            compressed_tokens=300,
            ratio=0.3,
        )
        assert "70%" in r.savings
        assert "1000" in r.savings
        assert "300" in r.savings

    def test_savings_with_dedup(self):
        r = ShrinkResult(
            context="test",
            sources=[],
            original_tokens=1000,
            compressed_tokens=300,
            ratio=0.3,
            dedup_removed=3,
        )
        assert "3 redundant chunks removed" in r.savings

    def test_savings_pct(self):
        r = ShrinkResult(
            context="test",
            sources=[],
            original_tokens=1000,
            compressed_tokens=400,
            ratio=0.4,
        )
        assert abs(r.savings_pct - 60.0) < 0.1

    def test_zero_ratio(self):
        r = ShrinkResult(
            context="",
            sources=[],
            original_tokens=0,
            compressed_tokens=0,
            ratio=1.0,
        )
        assert r.savings_pct == 0.0


class TestChunkScore:
    """Test ChunkScore dataclass."""

    def test_fields(self):
        cs = ChunkScore(
            index=0,
            text="test",
            source="file.md",
            similarity=0.85,
            density=0.6,
            importance=0.78,
            compression_ratio=0.65,
        )
        assert cs.index == 0
        assert cs.similarity == 0.85
        assert cs.deduplicated is False

    def test_dedup_flag(self):
        cs = ChunkScore(
            index=0,
            text="test",
            source="file.md",
            similarity=0.85,
            density=0.6,
            importance=0.78,
            compression_ratio=0.65,
            deduplicated=True,
        )
        assert cs.deduplicated is True
