# TokenShrink

**Cut your AI costs 50-80%.** FAISS semantic retrieval + LLMLingua compression.

Stop loading entire files into your prompts. Load only what's relevant, compressed.

## Quick Start

```bash
pip install tokenshrink

# Index your docs
tokenshrink index ./docs

# Get compressed context
tokenshrink query "What are the API limits?" --compress
```

## Why TokenShrink?

| Without | With TokenShrink |
|---------|------------------|
| Load entire file (5000 tokens) | Load relevant chunks (200 tokens) |
| $0.15 per query | $0.03 per query |
| Slow responses | Fast responses |
| Hit context limits | Stay under limits |

**Real numbers:** 50-80% token reduction on typical RAG workloads.

## Installation

```bash
# Basic (retrieval only)
pip install tokenshrink

# With compression (recommended)
pip install tokenshrink[compression]
```

## Usage

### CLI

```bash
# Index files
tokenshrink index ./docs
tokenshrink index ./src --extensions .py,.md

# Query (retrieval only)
tokenshrink query "How do I authenticate?"

# Query with compression
tokenshrink query "How do I authenticate?" --compress

# View stats
tokenshrink stats

# JSON output (for scripts)
tokenshrink query "question" --json
```

### Python API

```python
from tokenshrink import TokenShrink

# Initialize
ts = TokenShrink()

# Index your files
ts.index("./docs")

# Get compressed context
result = ts.query("What are the rate limits?")

print(result.context)      # Ready for your LLM
print(result.savings)      # "Saved 65% (1200 → 420 tokens)"
print(result.sources)      # ["api.md", "limits.md"]
```

### Integration Examples

**With OpenAI:**

```python
from tokenshrink import TokenShrink
from openai import OpenAI

ts = TokenShrink()
ts.index("./knowledge")

client = OpenAI()

def ask(question: str) -> str:
    # Get relevant, compressed context
    ctx = ts.query(question)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Context:\n{ctx.context}"},
            {"role": "user", "content": question}
        ]
    )
    
    print(f"Token savings: {ctx.savings}")
    return response.choices[0].message.content

answer = ask("What's the refund policy?")
```

**With LangChain:**

```python
from tokenshrink import TokenShrink
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

ts = TokenShrink()
ts.index("./docs")

def get_context(query: str) -> str:
    result = ts.query(query)
    return result.context

# Use in your chain
template = PromptTemplate(
    input_variables=["context", "question"],
    template="Context:\n{context}\n\nQuestion: {question}"
)
```

## How It Works

```
┌──────────┐     ┌───────────┐     ┌────────────┐
│  Files   │ ──► │  Indexer  │ ──► │ FAISS Index│
└──────────┘     │ (MiniLM)  │     └────────────┘
                 └───────────┘            │
                                          ▼
┌──────────┐     ┌───────────┐     ┌────────────┐
│ Question │ ──► │  Search   │ ──► │  Relevant  │
└──────────┘     │           │     │  Chunks    │
                 └───────────┘     └────────────┘
                                          │
                                          ▼
                               ┌────────────────┐
                               │  Compressor    │
                               │ (LLMLingua-2)  │
                               └────────────────┘
                                          │
                                          ▼
                               ┌────────────────┐
                               │ Optimized      │
                               │ Context        │
                               └────────────────┘
```

1. **Index**: Chunks your files, creates embeddings with MiniLM
2. **Search**: Finds relevant chunks via semantic similarity
3. **Compress**: Removes redundancy while preserving meaning

## Works With REFRAG

[REFRAG](https://arxiv.org/abs/2509.01092) (Meta, 2025) — [paper](https://arxiv.org/abs/2509.01092) · [github](https://github.com/Shaivpidadi/refrag) — demonstrated that RAG contexts have sparse, block-diagonal attention patterns — most retrieved passages barely interact during decoding. Their compress→sense→expand pipeline achieves 30x TTFT speedup at the **decoding** stage.

TokenShrink is the **upstream** complement: we reduce what goes into the context window *before* decoding starts. Stack them:

```
Your files → TokenShrink (retrieval + compression) → LLM → REFRAG (decode-time optimization)
              ↓ 50-80% fewer tokens                        ↓ 30x faster first token
```

Together, you get end-to-end savings across both retrieval and inference.

### Roadmap: REFRAG-Inspired Features

- **Adaptive compression** — Vary compression ratio per chunk based on information density (REFRAG's "sense" concept applied upstream)
- **Block-diagonal deduplication** — Detect and remove cross-passage redundancy exploiting attention sparsity patterns
- **Chunk importance scoring** — Score retrieved chunks by estimated attention weight, compress low-importance chunks more aggressively

## Configuration

```python
ts = TokenShrink(
    index_dir=".tokenshrink",    # Where to store the index
    model="all-MiniLM-L6-v2",    # Embedding model
    chunk_size=512,              # Words per chunk
    chunk_overlap=50,            # Overlap between chunks
    device="auto",               # auto, mps, cuda, cpu
    compression=True,            # Enable LLMLingua
)
```

## Supported File Types

Default: `.md`, `.txt`, `.py`, `.json`, `.yaml`, `.yml`

Custom:
```bash
tokenshrink index ./src --extensions .py,.ts,.js,.md
```

## Performance

| Metric | Value |
|--------|-------|
| Index 1000 files | ~30 seconds |
| Search latency | <50ms |
| Compression | ~200ms |
| Token reduction | 50-80% |

## Requirements

- Python 3.10+
- 4GB RAM (8GB for compression)
- Apple Silicon: MPS acceleration
- NVIDIA: CUDA acceleration

## FAQ

**Q: Do I need LLMLingua?**  
A: No. Retrieval works without it (still saves 60-70% by loading only relevant chunks). Add compression for extra 20-30% savings.

**Q: Does it work with non-English?**  
A: Retrieval works well with multilingual content. Compression is English-optimized.

**Q: How do I update the index?**  
A: Just run `tokenshrink index` again. It detects changed files automatically.

## Uninstall

```bash
pip uninstall tokenshrink
rm -rf .tokenshrink  # Remove local index
```

## Feedback — Humans & Agents Welcome

We want to hear from you. Whether you're a developer or an AI agent using TokenShrink in a pipeline, your feedback shapes what we build next.

**Quick feedback (30 seconds):**
- 👍 Working great? [Tell us what you're using it for](https://github.com/MusashiMiyamoto1-cloud/tokenshrink/discussions/categories/show-and-tell)
- 💡 Feature idea? [Share it](https://github.com/MusashiMiyamoto1-cloud/tokenshrink/discussions/categories/ideas)
- 🐛 Bug? [Open an issue](https://github.com/MusashiMiyamoto1-cloud/tokenshrink/issues/new?template=bug.md)
- ❓ Question? [Ask here](https://github.com/MusashiMiyamoto1-cloud/tokenshrink/discussions/categories/q-a)

**For agents:**
```bash
# Report feedback programmatically
tokenshrink feedback "compression ratio lower than expected on code files"

# Or post directly via GitHub API
curl -X POST https://api.github.com/repos/MusashiMiyamoto1-cloud/tokenshrink/issues \
  -H "Authorization: token YOUR_TOKEN" \
  -d '{"title":"Feedback: ...","labels":["feedback"]}'
```

Every piece of feedback gets read. We're building this in the open.

---

Built by [Musashi](https://github.com/MusashiMiyamoto1-cloud) · Part of [Agent Guard](https://agentguard.co)
