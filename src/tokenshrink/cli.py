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
from pathlib import Path

from tokenshrink import TokenShrink, __version__


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
        
        result = ts.query(
            args.question,
            k=args.k,
            max_tokens=args.max_tokens,
            compress=compress,
        )
        
        if args.json:
            print(json.dumps({
                "context": result.context,
                "sources": result.sources,
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "savings_pct": result.savings_pct,
            }, indent=2))
        else:
            if result.sources:
                print(f"Sources: {', '.join(Path(s).name for s in result.sources)}")
                print(f"Stats: {result.savings}")
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
