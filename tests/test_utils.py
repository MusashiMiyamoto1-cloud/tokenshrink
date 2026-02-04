"""Tests for REFRAG-inspired utility functions."""

import math
import numpy as np
import pytest

from tokenshrink.pipeline import (
    _information_density,
    _compute_importance,
    _adaptive_ratio,
    _deduplicate_chunks,
)


class TestInformationDensity:
    """Tests for _information_density."""

    def test_empty_string(self):
        assert _information_density("") == 0.0

    def test_single_char(self):
        # Single repeated char = zero entropy
        result = _information_density("aaaa")
        assert result == 0.0 or result < 0.01

    def test_uniform_distribution(self):
        # All unique chars = high entropy
        result = _information_density("abcdefghijklmnopqrstuvwxyz0123456789")
        assert result > 0.5

    def test_returns_float(self):
        result = _information_density("hello world")
        assert isinstance(result, float)

    def test_range_0_to_1(self):
        texts = [
            "aaaa",
            "hello world",
            "The quick brown fox jumps over the lazy dog",
            "import sys; x = {k: v for k, v in enumerate(range(100))}",
            "a" * 1000,
            "".join(chr(i) for i in range(32, 127)),
        ]
        for text in texts:
            result = _information_density(text)
            assert 0.0 <= result <= 1.0, f"Out of range for: {text[:30]}"

    def test_code_higher_than_prose(self):
        prose = "the the the the the the the the the"
        code = "def f(x): return {k: v for k, v in zip(range(10), map(str, x))}"
        assert _information_density(code) > _information_density(prose)

    def test_case_insensitive(self):
        # Function lowercases internally
        result1 = _information_density("Hello World")
        result2 = _information_density("hello world")
        assert result1 == result2

    def test_technical_content_higher(self):
        boilerplate = "This is a very simple and basic test document with common words repeated."
        technical = "PostgreSQL JSONB GIN indexes support @> containment, ? existence, #> path operators."
        assert _information_density(technical) > _information_density(boilerplate)

    def test_repetitive_low_density(self):
        repetitive = "the " * 200
        result = _information_density(repetitive)
        assert result < 0.3


class TestComputeImportance:
    """Tests for _compute_importance."""

    def test_default_weights(self):
        # 0.7 * similarity + 0.3 * density
        result = _compute_importance(1.0, 1.0)
        assert abs(result - 1.0) < 0.001

    def test_zero_inputs(self):
        assert _compute_importance(0.0, 0.0) == 0.0

    def test_similarity_dominates(self):
        # With default weights (0.7/0.3), similarity matters more
        high_sim = _compute_importance(0.9, 0.1)
        high_den = _compute_importance(0.1, 0.9)
        assert high_sim > high_den

    def test_custom_weights(self):
        result = _compute_importance(0.8, 0.6, sim_weight=0.5, density_weight=0.5)
        assert abs(result - 0.7) < 0.001

    def test_returns_float(self):
        assert isinstance(_compute_importance(0.5, 0.5), float)

    def test_linear_combination(self):
        s, d = 0.6, 0.4
        expected = 0.7 * s + 0.3 * d
        assert abs(_compute_importance(s, d) - expected) < 0.001

    def test_boundary_values(self):
        assert _compute_importance(0.0, 1.0) == pytest.approx(0.3)
        assert _compute_importance(1.0, 0.0) == pytest.approx(0.7)


class TestAdaptiveRatio:
    """Tests for _adaptive_ratio."""

    def test_high_importance_high_ratio(self):
        result = _adaptive_ratio(1.0)
        assert result >= 0.8

    def test_low_importance_low_ratio(self):
        result = _adaptive_ratio(0.0)
        assert result <= 0.3

    def test_mid_importance(self):
        result = _adaptive_ratio(0.5)
        assert 0.4 <= result <= 0.7

    def test_range_clamped(self):
        # Even with extreme inputs, stays in [min_ratio, max_ratio]
        assert _adaptive_ratio(100.0) <= 0.9
        assert _adaptive_ratio(-100.0) >= 0.2

    def test_custom_bounds(self):
        result = _adaptive_ratio(0.5, min_ratio=0.1, max_ratio=1.0)
        assert 0.1 <= result <= 1.0

    def test_monotonic(self):
        # Higher importance → higher ratio
        prev = _adaptive_ratio(0.0)
        for imp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            curr = _adaptive_ratio(imp)
            assert curr >= prev, f"Not monotonic at importance={imp}"
            prev = curr

    def test_default_params(self):
        # importance=0 → min_ratio=0.2, importance=1 → max_ratio=0.9
        assert _adaptive_ratio(0.0) == pytest.approx(0.2)
        assert _adaptive_ratio(1.0) == pytest.approx(0.9)

    def test_returns_float(self):
        assert isinstance(_adaptive_ratio(0.5), float)


class TestDeduplicateChunks:
    """Tests for _deduplicate_chunks."""

    def test_empty_list(self):
        chunks, removed = _deduplicate_chunks([], np.array([]).reshape(0, 0), 0.85)
        assert chunks == []
        assert removed == []

    def test_single_chunk(self):
        chunks = [{"text": "hello", "score": 0.9}]
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        result, removed = _deduplicate_chunks(chunks, emb, 0.85)
        assert len(result) == 1
        assert removed == []

    def test_identical_chunks_deduplicated(self):
        chunks = [
            {"text": "hello world", "score": 0.9},
            {"text": "hello world", "score": 0.8},
        ]
        # Identical embeddings → similarity = 1.0 > 0.85 threshold
        emb = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        result, removed = _deduplicate_chunks(chunks, emb, 0.85)
        assert len(result) == 1
        assert len(removed) == 1

    def test_different_chunks_kept(self):
        chunks = [
            {"text": "hello world", "score": 0.9},
            {"text": "goodbye moon", "score": 0.8},
        ]
        # Orthogonal embeddings → similarity = 0.0 < 0.85
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        result, removed = _deduplicate_chunks(chunks, emb, 0.85)
        assert len(result) == 2
        assert removed == []

    def test_higher_score_kept(self):
        """When deduplicating, the higher-scored chunk is kept."""
        chunks = [
            {"text": "low score", "score": 0.3},
            {"text": "high score", "score": 0.9},
        ]
        emb = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        result, removed = _deduplicate_chunks(chunks, emb, 0.85)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_threshold_boundary(self):
        """Similarity exactly at threshold should trigger dedup."""
        chunks = [
            {"text": "a", "score": 0.9},
            {"text": "b", "score": 0.8},
        ]
        # Create embeddings with known similarity > threshold
        emb = np.array([
            [1.0, 0.0],
            [0.86, 0.51],  # similarity with [1,0] = 0.86
        ], dtype=np.float32)
        # Normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sim = float(emb[0] @ emb[1])
        
        result, removed = _deduplicate_chunks(chunks, emb, threshold=sim - 0.01)
        assert len(result) == 1  # Should dedup since sim > threshold-epsilon

    def test_multiple_duplicates(self):
        """Three similar chunks → keep only one."""
        chunks = [
            {"text": "a", "score": 0.7},
            {"text": "b", "score": 0.9},
            {"text": "c", "score": 0.8},
        ]
        emb = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        result, removed = _deduplicate_chunks(chunks, emb, 0.85)
        assert len(result) == 1
        assert len(removed) == 2
        assert result[0]["score"] == 0.9  # Highest score kept

    def test_mixed_duplicates(self):
        """Some chunks are duplicates, others are unique."""
        chunks = [
            {"text": "a", "score": 0.9},
            {"text": "b", "score": 0.8},
            {"text": "c", "score": 0.7},
            {"text": "d", "score": 0.6},
        ]
        # a and b are similar, c and d are different
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],  # Very similar to a
            [0.0, 1.0, 0.0],   # Different
            [0.0, 0.0, 1.0],   # Different
        ], dtype=np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        result, removed = _deduplicate_chunks(chunks, emb, 0.85)
        assert len(result) == 3  # a kept, b removed, c and d kept
        assert len(removed) == 1
