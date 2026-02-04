"""
TokenShrink CLI.

Usage:
    tokenshrink index ./docs
    tokenshrink query "your question"
    tokenshrink stats
    tokenshrink clear
"""

import argparse
import sys
import json
import os
from pathlib import Path

# Early suppression: check for --quiet or --json BEFORE heavy imports
if "--quiet" in sys.argv or "--json" in sys.argv:
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TQDM_DISABLE"] = "1"
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.WARNING)

from tokenshrink import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="tokenshrink",
        description="Cut your AI costs 50-80%. FAISS retrieval + LLMLingua compression.",
    )
    parser.add_argument("--version", action="version", version=f"tokenshrink {__version__}")
    parser.add_argument(
        "--index-dir",
        default=".tokenshrink",
        help="Directory to store the index (default: .tokenshrink)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress model loading messages",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # index
    index_parser = subparsers.add_parser("index", help="Index files for retrieval")
    index_parser.add_argument("path", help="File or directory to index")
    index_parser.add_argument(
        "-e", "--extensions",
        default=".md,.txt,.py,.json,.yaml,.yml",
        help="File extensions to include (comma-separated)",
    )
    index_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Re-index even if files unchanged",
    )
    
    # query
    query_parser = subparsers.add_parser("query", help="Get relevant context for a question")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    query_parser.add_argument(
        "-c", "--compress",
        action="store_true",
        help="Enable compression (requires llmlingua)",
    )
    query_parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable compression",
    )
    query_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Target token limit (default: 2000)",
    )
    query_parser.add_argument(
        "--adaptive",
        action="store_true",
        default=None,
        help="Enable REFRAG-inspired adaptive compression (default: on)",
    )
    query_parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive compression",
    )
    query_parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable cross-passage deduplication",
    )
    query_parser.add_argument(
        "--scores",
        action="store_true",
        help="Show per-chunk importance scores",
    )
    
    # search (alias for query without compression)
    search_parser = subparsers.add_parser("search", help="Search without compression")
    search_parser.add_argument("question", help="Your question")
    search_parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    
    # stats
    subparsers.add_parser("stats", help="Show index statistics")
    
    # clear
    subparsers.add_parser("clear", help="Clear the index")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Lazy import to avoid loading ML models for --help/--version
    from tokenshrink import TokenShrink
    
    # Determine compression setting
    compression = True
    if hasattr(args, 'no_compress') and args.no_compress:
        compression = False
    if hasattr(args, 'compress') and args.compress:
        compression = True
    
    ts = TokenShrink(
        index_dir=args.index_dir,
        compression=compression,
    )
    
    if args.command == "index":
        extensions = tuple(e.strip() if e.startswith(".") else f".{e.strip()}" 
                          for e in args.extensions.split(","))
        result = ts.index(args.path, extensions=extensions, force=args.force)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"✓ Indexed {result['files_indexed']} files")
            print(f"  Chunks: {result['chunks_added']} added, {result['total_chunks']} total")
            print(f"  Files: {result['total_files']} tracked")
    
    elif args.command == "query":
        compress = None
        if args.compress:
            compress = True
        elif args.no_compress:
            compress = False
        
        adaptive_flag = None
        if getattr(args, 'adaptive', None):
            adaptive_flag = True
        elif getattr(args, 'no_adaptive', False):
            adaptive_flag = False
        
        dedup_flag = None
        if getattr(args, 'no_dedup', False):
            dedup_flag = False
        
        result = ts.query(
            args.question,
            k=args.k,
            max_tokens=args.max_tokens,
            compress=compress,
            adaptive=adaptive_flag,
            dedup=dedup_flag,
        )
        
        if args.json:
            output = {
                "context": result.context,
                "sources": result.sources,
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "savings_pct": result.savings_pct,
                "dedup_removed": result.dedup_removed,
            }
            if getattr(args, 'scores', False) and result.chunk_scores:
                output["chunk_scores"] = [
                    {
                        "source": Path(cs.source).name,
                        "similarity": round(cs.similarity, 3),
                        "density": round(cs.density, 3),
                        "importance": round(cs.importance, 3),
                        "compression_ratio": round(cs.compression_ratio, 3),
                        "deduplicated": cs.deduplicated,
                    }
                    for cs in result.chunk_scores
                ]
            print(json.dumps(output, indent=2))
        else:
            if result.sources:
                print(f"Sources: {', '.join(Path(s).name for s in result.sources)}")
                print(f"Stats: {result.savings}")
                
                if result.savings_pct == 0.0:
                    print("  Tip: Install llmlingua for compression: pip install llmlingua")
                
                if getattr(args, 'scores', False) and result.chunk_scores:
                    print("\nChunk Importance Scores:")
                    for cs in result.chunk_scores:
                        status = " [DEDUP]" if cs.deduplicated else ""
                        print(f"  {Path(cs.source).name}: "
                              f"sim={cs.similarity:.2f} density={cs.density:.2f} "
                              f"importance={cs.importance:.2f} ratio={cs.compression_ratio:.2f}"
                              f"{status}")
                
                print()
                print(result.context)
            else:
                print("No relevant content found.")
    
    elif args.command == "search":
        results = ts.search(args.question, k=args.k)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if not results:
                print("No results found.")
            else:
                for i, r in enumerate(results, 1):
                    print(f"\n[{i}] {Path(r['source']).name} (score: {r['score']:.3f})")
                    print("-" * 40)
                    print(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))
    
    elif args.command == "stats":
        result = ts.stats()
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Index: {result['index_dir']}")
            print(f"Chunks: {result['total_chunks']}")
            print(f"Files: {result['total_files']}")
            print(f"Compression: {'available' if result['compression_available'] else 'not installed'}")
            print(f"Device: {result['device']}")
    
    elif args.command == "clear":
        ts.clear()
        if args.json:
            print(json.dumps({"status": "cleared"}))
        else:
            print("✓ Index cleared")


if __name__ == "__main__":
    main()
