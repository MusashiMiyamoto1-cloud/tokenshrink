"""
TokenShrink: Cut your AI costs 50-80%.

FAISS semantic retrieval + LLMLingua compression for token-efficient context loading.

Usage:
    from tokenshrink import TokenShrink
    
    ts = TokenShrink()
    ts.index("./docs")
    
    result = ts.query("What are the API limits?")
    print(result.context)      # Compressed, relevant context
    print(result.savings)      # "Saved 65% (1200 → 420 tokens)"

CLI:
    tokenshrink index ./docs
    tokenshrink query "your question"
    tokenshrink stats
"""

from tokenshrink.pipeline import TokenShrink, ShrinkResult

__version__ = "0.1.0"
__all__ = ["TokenShrink", "ShrinkResult"]
