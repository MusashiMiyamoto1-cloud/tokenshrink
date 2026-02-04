"""Stress tests — performance, scale, and resource usage."""

import time
import os
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tokenshrink import TokenShrink
from tokenshrink.pipeline import _information_density, _compute_importance, _adaptive_ratio


class TestIndexingPerformance:
    """Benchmark indexing speed and memory."""

    def test_index_50_files(self, tmp_dir, large_docs, make_ts):
        """Index 50 files and measure time."""
        ts = make_ts()
        
        start = time.perf_counter()
        result = ts.index(str(large_docs))
        elapsed = time.perf_counter() - start
        
        assert result["files_indexed"] == 50
        assert result["chunks_added"] > 100
        assert elapsed < 120, f"Indexing 50 files took {elapsed:.1f}s (max 120s)"
        print(f"\n  Indexed 50 files: {result['chunks_added']} chunks in {elapsed:.1f}s")

    def test_reindex_performance(self, tmp_dir, large_docs, make_ts):
        """Reindexing unchanged files should be near-instant."""
        ts = make_ts()
        ts.index(str(large_docs))
        
        start = time.perf_counter()
        result = ts.index(str(large_docs))
        elapsed = time.perf_counter() - start
        
        assert result["files_indexed"] == 0
        assert elapsed < 2.0, f"Reindex (no changes) took {elapsed:.1f}s (max 2s)"
        print(f"\n  Reindex (no-op): {elapsed:.3f}s")

    def test_force_reindex_performance(self, tmp_dir, large_docs, make_ts):
        """Force reindex performance."""
        ts = make_ts()
        ts.index(str(large_docs))
        
        start = time.perf_counter()
        result = ts.index(str(large_docs), force=True)
        elapsed = time.perf_counter() - start
        
        assert result["files_indexed"] == 50
        assert elapsed < 120, f"Force reindex took {elapsed:.1f}s (max 120s)"
        print(f"\n  Force reindex 50 files: {elapsed:.1f}s")


class TestQueryPerformance:
    """Benchmark query speed."""

    @pytest.fixture(autouse=True)
    def setup_large_index(self, tmp_dir, large_docs, make_ts):
        """Set up a large index for query tests."""
        self.ts = make_ts(adaptive=True, dedup=True)
        self.ts.index(str(large_docs))

    def test_single_query_speed(self):
        """Single query should be fast."""
        start = time.perf_counter()
        result = self.ts.query("machine learning gradient descent")
        elapsed = time.perf_counter() - start
        
        assert len(result.context) > 0
        assert elapsed < 5.0, f"Single query took {elapsed:.1f}s (max 5s)"
        print(f"\n  Single query: {elapsed:.3f}s")

    def test_batch_queries(self):
        """Run 20 queries and measure throughput."""
        queries = [
            "machine learning neural networks",
            "database postgresql indexes",
            "TCP networking DNS resolution",
            "TLS encryption CORS security",
            "CI/CD pipeline deployment",
            "React component rendering",
            "REST API design patterns",
            "unit testing integration",
            "prometheus monitoring grafana",
            "redis caching CDN",
            "gradient descent optimization",
            "SQL query performance",
            "HTTP protocol headers",
            "authentication authorization",
            "docker kubernetes containers",
            "JavaScript frontend framework",
            "GraphQL schema resolver",
            "pytest mock fixtures",
            "alerting observability traces",
            "cache invalidation strategy",
        ]
        
        start = time.perf_counter()
        results = [self.ts.query(q) for q in queries]
        elapsed = time.perf_counter() - start
        
        assert all(len(r.context) > 0 for r in results)
        qps = len(queries) / elapsed
        print(f"\n  20 queries: {elapsed:.1f}s ({qps:.1f} queries/sec)")
        assert elapsed < 60, f"20 queries took {elapsed:.1f}s (max 60s)"

    def test_query_with_dedup(self):
        """Query with dedup shouldn't be much slower."""
        # Warm up
        self.ts.query("test", k=5, dedup=True)
        
        start = time.perf_counter()
        for _ in range(10):
            self.ts.query("database security performance", k=5, dedup=True)
        elapsed = time.perf_counter() - start
        
        per_query = elapsed / 10
        print(f"\n  Query + dedup avg: {per_query:.3f}s")
        assert per_query < 5.0

    def test_query_k_scaling(self):
        """Query time vs k value."""
        times = {}
        for k in [1, 3, 5, 10]:
            actual_k = min(k, self.ts._index.ntotal)
            start = time.perf_counter()
            for _ in range(5):
                self.ts.query("machine learning", k=actual_k)
            times[k] = (time.perf_counter() - start) / 5
        
        print(f"\n  Query time vs k: {json.dumps({k: f'{t:.3f}s' for k, t in times.items()})}")
        # k=10 shouldn't be orders of magnitude slower than k=1
        if times.get(10) and times.get(1):
            assert times[10] < times[1] * 10


class TestSearchPerformance:
    """Benchmark search speed (no compression overhead)."""

    @pytest.fixture(autouse=True)
    def setup_large_index(self, tmp_dir, large_docs, make_ts):
        self.ts = make_ts()
        self.ts.index(str(large_docs))

    def test_search_speed(self):
        start = time.perf_counter()
        for _ in range(50):
            self.ts.search("database query optimization", k=5)
        elapsed = time.perf_counter() - start
        
        per_search = elapsed / 50
        print(f"\n  Search avg: {per_search:.3f}s ({50/elapsed:.0f} searches/sec)")
        assert per_search < 2.0


class TestUtilityPerformance:
    """Benchmark utility functions."""

    def test_information_density_speed(self):
        """Information density should be O(n) and fast."""
        text = "The quick brown fox jumps over the lazy dog. " * 1000
        
        start = time.perf_counter()
        for _ in range(1000):
            _information_density(text)
        elapsed = time.perf_counter() - start
        
        per_call = elapsed / 1000
        print(f"\n  _information_density (45k chars): {per_call*1000:.2f}ms/call")
        assert per_call < 0.01  # < 10ms per call

    def test_compute_importance_speed(self):
        start = time.perf_counter()
        for _ in range(100000):
            _compute_importance(0.85, 0.6)
        elapsed = time.perf_counter() - start
        
        per_call = elapsed / 100000
        print(f"\n  _compute_importance: {per_call*1e6:.1f}µs/call")
        assert per_call < 0.001

    def test_adaptive_ratio_speed(self):
        start = time.perf_counter()
        for _ in range(100000):
            _adaptive_ratio(0.75)
        elapsed = time.perf_counter() - start
        
        per_call = elapsed / 100000
        print(f"\n  _adaptive_ratio: {per_call*1e6:.1f}µs/call")
        assert per_call < 0.001


class TestIndexSize:
    """Test index file sizes and memory usage."""

    def test_index_file_size(self, tmp_dir, large_docs, make_ts):
        """Index files shouldn't be unreasonably large."""
        ts = make_ts()
        ts.index(str(large_docs))
        
        idx_dir = tmp_dir / ".ts"
        faiss_size = (idx_dir / "index.faiss").stat().st_size
        meta_size = (idx_dir / "meta.json").stat().st_size
        
        # 50 files, ~100+ chunks, 384-dim vectors
        print(f"\n  FAISS index: {faiss_size/1024:.1f}KB")
        print(f"  Metadata: {meta_size/1024:.1f}KB")
        
        # FAISS index: each vector is 384 * 4 bytes = 1.5KB
        # 200 chunks → ~300KB is reasonable
        assert faiss_size < 5 * 1024 * 1024  # < 5MB
        assert meta_size < 50 * 1024 * 1024   # < 50MB (text chunks stored)

    def test_chunk_count_scaling(self, tmp_dir, make_ts):
        """More files → proportionally more chunks."""
        docs = tmp_dir / "docs"
        docs.mkdir()
        
        ts = make_ts(chunk_size=100)
        
        counts = {}
        for n in [5, 10, 20]:
            # Create n files
            for i in range(n):
                (docs / f"doc-{i:03d}.md").write_text(f"Document {i} content. " * 200)
            
            ts.clear()
            result = ts.index(str(docs))
            counts[n] = result["chunks_added"]
            
            # Clean up for next iteration
            for f in docs.glob("*.md"):
                f.unlink()
        
        print(f"\n  Chunks per file count: {json.dumps(counts)}")
        # Should scale roughly linearly
        assert counts[20] > counts[5]


class TestMemoryStability:
    """Test for memory leaks and stability."""

    def test_repeated_index_clear(self, tmp_dir, sample_docs, make_ts):
        """Repeated index/clear cycles shouldn't leak."""
        ts = make_ts()
        
        for i in range(10):
            ts.index(str(sample_docs), force=True)
            ts.clear()
        
        # Final state should be clean
        assert ts._index.ntotal == 0
        assert ts._chunks == []

    def test_repeated_queries(self, indexed_ts):
        """Many repeated queries shouldn't degrade."""
        for _ in range(50):
            result = indexed_ts.query("authentication")
            assert len(result.context) > 0
