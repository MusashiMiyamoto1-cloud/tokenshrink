"""
TokenShrink core: FAISS retrieval + LLMLingua compression.

v0.2.0: REFRAG-inspired adaptive compression, deduplication, importance scoring.
"""

import os
import json
import hashlib
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional compression
try:
    from llmlingua import PromptCompressor
    HAS_COMPRESSION = True
except ImportError:
    HAS_COMPRESSION = False


@dataclass
class ChunkScore:
    """Per-chunk scoring metadata (REFRAG-inspired)."""
    index: int
    text: str
    source: str
    similarity: float        # Cosine similarity to query
    density: float           # Information density (entropy proxy)
    importance: float        # Combined importance score
    compression_ratio: float # Adaptive ratio assigned to this chunk
    deduplicated: bool = False  # Flagged as redundant


@dataclass
class ShrinkResult:
    """Result from a query."""
    context: str
    sources: list[str]
    original_tokens: int
    compressed_tokens: int
    ratio: float
    chunk_scores: list[ChunkScore] = field(default_factory=list)
    dedup_removed: int = 0
    
    @property
    def savings(self) -> str:
        pct = (1 - self.ratio) * 100
        extra = ""
        if self.dedup_removed > 0:
            extra = f", {self.dedup_removed} redundant chunks removed"
        return f"Saved {pct:.0f}% ({self.original_tokens} → {self.compressed_tokens} tokens{extra})"
    
    @property
    def savings_pct(self) -> float:
        return (1 - self.ratio) * 100


# ---------------------------------------------------------------------------
#  REFRAG-inspired utilities
# ---------------------------------------------------------------------------

def _information_density(text: str) -> float:
    """
    Estimate information density of text via character-level entropy.
    Higher entropy ≈ more information-dense (code, data, technical content).
    Lower entropy ≈ more redundant (boilerplate, filler).
    Returns 0.0-1.0 normalized score.
    """
    if not text:
        return 0.0
    
    freq = {}
    for ch in text.lower():
        freq[ch] = freq.get(ch, 0) + 1
    
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize: English text entropy is ~4.0-4.5 bits/char
    # Code/data is ~5.0-6.0, very repetitive text is ~2.0-3.0
    # Map to 0-1 range with midpoint at ~4.5
    normalized = min(1.0, max(0.0, (entropy - 2.0) / 4.0))
    return normalized


def _compute_importance(similarity: float, density: float, 
                        sim_weight: float = 0.7, density_weight: float = 0.3) -> float:
    """
    Combined importance score from similarity and density.
    REFRAG insight: not all retrieved chunks contribute equally.
    High similarity + high density = most important (compress less).
    Low similarity + low density = least important (compress more or drop).
    """
    return sim_weight * similarity + density_weight * density


def _adaptive_ratio(importance: float, base_ratio: float = 0.5,
                    min_ratio: float = 0.2, max_ratio: float = 0.9) -> float:
    """
    Map importance score to compression ratio.
    High importance → keep more (higher ratio, less compression).
    Low importance → compress harder (lower ratio).
    
    ratio=1.0 means keep everything, ratio=0.2 means keep 20%.
    """
    # Linear interpolation: low importance → min_ratio, high → max_ratio
    ratio = min_ratio + importance * (max_ratio - min_ratio)
    return min(max_ratio, max(min_ratio, ratio))


def _deduplicate_chunks(chunks: list[dict], embeddings: np.ndarray,
                         threshold: float = 0.85) -> tuple[list[dict], list[int]]:
    """
    Remove near-duplicate chunks using embedding cosine similarity.
    REFRAG insight: block-diagonal attention means redundant passages waste compute.
    
    Returns: (deduplicated_chunks, removed_indices)
    """
    if len(chunks) <= 1:
        return chunks, []
    
    # Compute pairwise similarities
    # embeddings should already be normalized (from SentenceTransformer with normalize_embeddings=True)
    sim_matrix = embeddings @ embeddings.T
    
    keep = []
    removed = []
    kept_indices = set()
    
    # Greedy: keep highest-scored chunks, remove near-duplicates
    # Sort by score descending
    scored = sorted(enumerate(chunks), key=lambda x: x[1].get("score", 0), reverse=True)
    
    for idx, chunk in scored:
        if idx in removed:
            continue
        
        # Check if this chunk is too similar to any already-kept chunk
        is_dup = False
        for kept_idx in kept_indices:
            if sim_matrix[idx, kept_idx] > threshold:
                is_dup = True
                break
        
        if is_dup:
            removed.append(idx)
        else:
            keep.append(chunk)
            kept_indices.add(idx)
    
    return keep, removed


class TokenShrink:
    """
    Token-efficient context loading.
    
    Usage:
        ts = TokenShrink()
        ts.index("./docs")
        result = ts.query("What are the constraints?")
        print(result.context)
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        device: str = "auto",
        compression: bool = True,
        adaptive: bool = True,
        dedup: bool = True,
        dedup_threshold: float = 0.85,
    ):
        """
        Initialize TokenShrink.
        
        Args:
            index_dir: Where to store the FAISS index. Default: ./.tokenshrink
            model: Sentence transformer model for embeddings.
            chunk_size: Words per chunk.
            chunk_overlap: Overlap between chunks.
            device: Device for compression (auto, mps, cuda, cpu).
            compression: Enable LLMLingua compression.
            adaptive: Enable REFRAG-inspired adaptive compression (v0.2).
            dedup: Enable cross-passage deduplication (v0.2).
            dedup_threshold: Cosine similarity threshold for dedup (0-1).
        """
        self.index_dir = Path(index_dir or ".tokenshrink")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._compression_enabled = compression and HAS_COMPRESSION
        self._adaptive = adaptive
        self._dedup = dedup
        self._dedup_threshold = dedup_threshold
        
        # Auto-detect device
        if device == "auto":
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device
        
        # Load embedding model
        self._model = SentenceTransformer(model)
        self._dim = self._model.get_sentence_embedding_dimension()
        
        # FAISS index
        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks: list[dict] = []
        self._file_hashes: dict[str, str] = {}
        
        # Load existing index
        if self.index_dir.exists():
            self._load()
        
        # Lazy-load compressor
        self._compressor: Optional[PromptCompressor] = None

    def _get_compressor(self) -> PromptCompressor:
        """Lazy-load the compressor."""
        if self._compressor is None:
            if not HAS_COMPRESSION:
                raise ImportError(
                    "Compression requires llmlingua. "
                    "Install with: pip install tokenshrink[compression]"
                )
            self._compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map=self._device,
            )
        return self._compressor

    def _chunk_text(self, text: str, source: str) -> list[dict]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 20:
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "source": source,
                "offset": i,
            })
        return chunks

    def _hash_file(self, path: Path) -> str:
        """Get file content hash."""
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def index(
        self,
        path: str,
        extensions: tuple[str, ...] = (".md", ".txt", ".py", ".json", ".yaml", ".yml"),
        force: bool = False,
    ) -> dict:
        """
        Index files for retrieval.
        
        Args:
            path: File or directory to index.
            extensions: File extensions to include (for directories).
            force: Re-index even if unchanged.
            
        Returns:
            Stats dict with files_indexed, chunks_added, total_chunks.
        """
        path = Path(path)
        skip_dirs = {"node_modules", "__pycache__", ".venv", "venv", ".git", ".tokenshrink"}
        
        files_indexed = 0
        chunks_added = 0
        
        if path.is_file():
            files = [path]
        else:
            files = [
                f for f in path.rglob("*")
                if f.is_file()
                and f.suffix.lower() in extensions
                and not f.name.startswith(".")
                and not any(d in f.parts for d in skip_dirs)
            ]
        
        for file_path in files:
            try:
                file_str = str(file_path.resolve())
                current_hash = self._hash_file(file_path)
                
                if not force and self._file_hashes.get(file_str) == current_hash:
                    continue
                
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                chunks = self._chunk_text(text, file_str)
                if not chunks:
                    continue
                
                embeddings = self._model.encode(
                    [c["text"] for c in chunks],
                    normalize_embeddings=True
                )
                
                self._index.add(np.array(embeddings, dtype=np.float32))
                self._chunks.extend(chunks)
                self._file_hashes[file_str] = current_hash
                
                files_indexed += 1
                chunks_added += len(chunks)
                
            except Exception as e:
                print(f"Warning: {file_path}: {e}")
        
        self._save()
        
        return {
            "files_indexed": files_indexed,
            "chunks_added": chunks_added,
            "total_chunks": self._index.ntotal,
            "total_files": len(self._file_hashes),
        }

    def query(
        self,
        question: str,
        k: int = 5,
        min_score: float = 0.3,
        max_tokens: int = 2000,
        compress: Optional[bool] = None,
        adaptive: Optional[bool] = None,
        dedup: Optional[bool] = None,
    ) -> ShrinkResult:
        """
        Get relevant, compressed context for a question.
        
        Args:
            question: The query.
            k: Number of chunks to retrieve.
            min_score: Minimum similarity score (0-1).
            max_tokens: Target token limit for compression.
            compress: Override compression setting.
            adaptive: Override adaptive compression (REFRAG-inspired).
            dedup: Override deduplication setting.
            
        Returns:
            ShrinkResult with context, sources, token stats, and chunk scores.
        """
        if self._index.ntotal == 0:
            return ShrinkResult(
                context="",
                sources=[],
                original_tokens=0,
                compressed_tokens=0,
                ratio=1.0,
            )
        
        use_adaptive = adaptive if adaptive is not None else self._adaptive
        use_dedup = dedup if dedup is not None else self._dedup
        
        # Retrieve
        embedding = self._model.encode([question], normalize_embeddings=True)
        scores, indices = self._index.search(
            np.array(embedding, dtype=np.float32),
            min(k, self._index.ntotal)
        )
        
        results = []
        result_embeddings = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_score:
                chunk = self._chunks[idx].copy()
                chunk["score"] = float(score)
                chunk["_idx"] = int(idx)
                results.append(chunk)
        
        if not results:
            return ShrinkResult(
                context="",
                sources=[],
                original_tokens=0,
                compressed_tokens=0,
                ratio=1.0,
            )
        
        # ── REFRAG Step 1: Importance scoring ──
        chunk_scores = []
        for i, chunk in enumerate(results):
            density = _information_density(chunk["text"])
            importance = _compute_importance(chunk["score"], density)
            comp_ratio = _adaptive_ratio(importance) if use_adaptive else 0.5
            
            chunk_scores.append(ChunkScore(
                index=i,
                text=chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
                source=chunk["source"],
                similarity=chunk["score"],
                density=density,
                importance=importance,
                compression_ratio=comp_ratio,
            ))
        
        # ── REFRAG Step 2: Cross-passage deduplication ──
        dedup_removed = 0
        if use_dedup and len(results) > 1:
            # Get embeddings for dedup
            chunk_texts = [c["text"] for c in results]
            chunk_embs = self._model.encode(chunk_texts, normalize_embeddings=True)
            
            deduped, removed_indices = _deduplicate_chunks(
                results, np.array(chunk_embs, dtype=np.float32),
                threshold=self._dedup_threshold
            )
            
            dedup_removed = len(removed_indices)
            
            # Mark removed chunks in scores
            for idx in removed_indices:
                if idx < len(chunk_scores):
                    chunk_scores[idx].deduplicated = True
            
            results = deduped
        
        # Sort remaining by importance (highest first)
        if use_adaptive:
            # Pair results with their scores for sorting
            result_score_pairs = []
            for chunk in results:
                # Find matching score
                for cs in chunk_scores:
                    if not cs.deduplicated and cs.source == chunk["source"] and cs.similarity == chunk["score"]:
                        result_score_pairs.append((chunk, cs))
                        break
                else:
                    result_score_pairs.append((chunk, None))
            
            result_score_pairs.sort(key=lambda x: x[1].importance if x[1] else 0, reverse=True)
            results = [pair[0] for pair in result_score_pairs]
        
        # Combine chunks
        combined = "\n\n---\n\n".join(
            f"[{Path(c['source']).name}]\n{c['text']}" for c in results
        )
        sources = list(set(c["source"] for c in results))
        
        # Estimate tokens
        original_tokens = len(combined.split())
        
        # ── REFRAG Step 3: Adaptive compression ──
        should_compress = compress if compress is not None else self._compression_enabled
        
        if should_compress and original_tokens > 100:
            if use_adaptive:
                compressed, stats = self._compress_adaptive(results, chunk_scores, max_tokens)
            else:
                compressed, stats = self._compress(combined, max_tokens)
            
            return ShrinkResult(
                context=compressed,
                sources=sources,
                original_tokens=stats["original"],
                compressed_tokens=stats["compressed"],
                ratio=stats["ratio"],
                chunk_scores=chunk_scores,
                dedup_removed=dedup_removed,
            )
        
        return ShrinkResult(
            context=combined,
            sources=sources,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            ratio=1.0,
            chunk_scores=chunk_scores,
            dedup_removed=dedup_removed,
        )

    def _compress_adaptive(self, chunks: list[dict], scores: list[ChunkScore],
                           max_tokens: int) -> tuple[str, dict]:
        """
        REFRAG-inspired adaptive compression: each chunk gets a different
        compression ratio based on its importance score.
        
        High-importance chunks (high similarity + high density) are kept 
        nearly intact. Low-importance chunks are compressed aggressively.
        """
        compressor = self._get_compressor()
        
        # Build a map from chunk source+score to its ChunkScore
        score_map = {}
        for cs in scores:
            if not cs.deduplicated:
                score_map[(cs.source, cs.similarity)] = cs
        
        compressed_parts = []
        total_original = 0
        total_compressed = 0
        
        for chunk in chunks:
            text = f"[{Path(chunk['source']).name}]\n{chunk['text']}"
            cs = score_map.get((chunk["source"], chunk.get("score", 0)))
            
            # Determine per-chunk ratio
            if cs:
                target_ratio = cs.compression_ratio
            else:
                target_ratio = 0.5  # Default fallback
            
            est_tokens = len(text.split())
            
            if est_tokens < 20:
                # Too short to compress meaningfully
                compressed_parts.append(text)
                total_original += est_tokens
                total_compressed += est_tokens
                continue
            
            try:
                # Compress with chunk-specific ratio
                max_chars = 1500
                if len(text) <= max_chars:
                    result = compressor.compress_prompt(
                        text,
                        rate=target_ratio,
                        force_tokens=["\n", ".", "!", "?"],
                    )
                    compressed_parts.append(result["compressed_prompt"])
                    total_original += result["origin_tokens"]
                    total_compressed += result["compressed_tokens"]
                else:
                    # Sub-chunk large texts
                    parts = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
                    for part in parts:
                        if not part.strip():
                            continue
                        r = compressor.compress_prompt(part, rate=target_ratio)
                        compressed_parts.append(r["compressed_prompt"])
                        total_original += r["origin_tokens"]
                        total_compressed += r["compressed_tokens"]
            except Exception:
                # Fallback: use uncompressed
                compressed_parts.append(text)
                total_original += est_tokens
                total_compressed += est_tokens
        
        combined = "\n\n---\n\n".join(compressed_parts)
        
        return combined, {
            "original": total_original,
            "compressed": total_compressed,
            "ratio": total_compressed / total_original if total_original else 1.0,
        }

    def _compress(self, text: str, max_tokens: int) -> tuple[str, dict]:
        """Compress text using LLMLingua-2."""
        compressor = self._get_compressor()
        
        # LLMLingua-2 works best with smaller chunks
        max_chars = 1500
        est_tokens = len(text.split())
        target_ratio = min(0.9, max_tokens / est_tokens) if est_tokens else 0.5
        
        if len(text) <= max_chars:
            result = compressor.compress_prompt(
                text,
                rate=target_ratio,
                force_tokens=["\n", ".", "!", "?"],
            )
            return result["compressed_prompt"], {
                "original": result["origin_tokens"],
                "compressed": result["compressed_tokens"],
                "ratio": result["compressed_tokens"] / result["origin_tokens"],
            }
        
        # Chunk large texts
        parts = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        compressed_parts = []
        total_original = 0
        total_compressed = 0
        
        for part in parts:
            if not part.strip():
                continue
            r = compressor.compress_prompt(part, rate=target_ratio)
            compressed_parts.append(r["compressed_prompt"])
            total_original += r["origin_tokens"]
            total_compressed += r["compressed_tokens"]
        
        return " ".join(compressed_parts), {
            "original": total_original,
            "compressed": total_compressed,
            "ratio": total_compressed / total_original if total_original else 1.0,
        }

    def search(self, question: str, k: int = 5, min_score: float = 0.3) -> list[dict]:
        """Search without compression. Returns raw chunks with scores."""
        if self._index.ntotal == 0:
            return []
        
        embedding = self._model.encode([question], normalize_embeddings=True)
        scores, indices = self._index.search(
            np.array(embedding, dtype=np.float32),
            min(k, self._index.ntotal)
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_score:
                chunk = self._chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)
        
        return results

    def stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_chunks": self._index.ntotal,
            "total_files": len(self._file_hashes),
            "index_dir": str(self.index_dir),
            "compression_available": HAS_COMPRESSION,
            "compression_enabled": self._compression_enabled,
            "device": self._device,
        }

    def clear(self):
        """Clear the index."""
        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks = []
        self._file_hashes = {}
        if self.index_dir.exists():
            import shutil
            shutil.rmtree(self.index_dir)

    def _save(self):
        """Save index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "meta.json", "w") as f:
            json.dump({
                "chunks": self._chunks,
                "hashes": self._file_hashes,
            }, f)

    def _load(self):
        """Load index from disk."""
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "meta.json"
        
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
                self._chunks = data.get("chunks", [])
                self._file_hashes = data.get("hashes", {})
