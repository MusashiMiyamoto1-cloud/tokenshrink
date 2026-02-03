"""
TokenShrink core: FAISS retrieval + LLMLingua compression.
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
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
class ShrinkResult:
    """Result from a query."""
    context: str
    sources: list[str]
    original_tokens: int
    compressed_tokens: int
    ratio: float
    
    @property
    def savings(self) -> str:
        pct = (1 - self.ratio) * 100
        return f"Saved {pct:.0f}% ({self.original_tokens} → {self.compressed_tokens} tokens)"
    
    @property
    def savings_pct(self) -> float:
        return (1 - self.ratio) * 100


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
        """
        self.index_dir = Path(index_dir or ".tokenshrink")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._compression_enabled = compression and HAS_COMPRESSION
        
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
    ) -> ShrinkResult:
        """
        Get relevant, compressed context for a question.
        
        Args:
            question: The query.
            k: Number of chunks to retrieve.
            min_score: Minimum similarity score (0-1).
            max_tokens: Target token limit for compression.
            compress: Override compression setting.
            
        Returns:
            ShrinkResult with context, sources, and token stats.
        """
        if self._index.ntotal == 0:
            return ShrinkResult(
                context="",
                sources=[],
                original_tokens=0,
                compressed_tokens=0,
                ratio=1.0,
            )
        
        # Retrieve
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
        
        if not results:
            return ShrinkResult(
                context="",
                sources=[],
                original_tokens=0,
                compressed_tokens=0,
                ratio=1.0,
            )
        
        # Combine chunks
        combined = "\n\n---\n\n".join(
            f"[{Path(c['source']).name}]\n{c['text']}" for c in results
        )
        sources = list(set(c["source"] for c in results))
        
        # Estimate tokens
        original_tokens = len(combined.split())
        
        # Compress if enabled
        should_compress = compress if compress is not None else self._compression_enabled
        
        if should_compress and original_tokens > 100:
            compressed, stats = self._compress(combined, max_tokens)
            return ShrinkResult(
                context=compressed,
                sources=sources,
                original_tokens=stats["original"],
                compressed_tokens=stats["compressed"],
                ratio=stats["ratio"],
            )
        
        return ShrinkResult(
            context=combined,
            sources=sources,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            ratio=1.0,
        )

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
