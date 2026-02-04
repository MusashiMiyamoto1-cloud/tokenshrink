"""
TokenShrink: Cut your AI costs 50-80%.

FAISS semantic retrieval + LLMLingua compression for token-efficient context loading.

v0.2.0: REFRAG-inspired adaptive compression, cross-passage deduplication,
        importance scoring. See README for details.

Usage:
    from tokenshrink import TokenShrink
    
    ts = TokenShrink()
    ts.index("./docs")
    
    result = ts.query("What are the API limits?")
    print(result.context)       # Compressed, relevant context
    print(result.savings)       # "Saved 72% (1200 → 336 tokens, 2 redundant chunks removed)"
    print(result.chunk_scores)  # Per-chunk importance scores

CLI:
    tokenshrink index ./docs
    tokenshrink query "your question"
    tokenshrink stats
"""

from tokenshrink.pipeline import TokenShrink, ShrinkResult, ChunkScore

__version__ = "0.2.0"
__all__ = ["TokenShrink", "ShrinkResult", "ChunkScore"]
