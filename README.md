# Debate Agent Memory

`debate-agent-memory` is a small, decoupled Python memory module for debate-style LLM
agents. It is intentionally framework-free, so an existing agent can add memory without
binding itself to LangChain, LlamaIndex, a specific vector database, or a specific prompt
layout.

The module currently provides:

- structured memory records for user messages, agent replies, claims, decisions, and notes
- pluggable storage backends, with JSONL, SQLite, and in-memory implementations included
- local relevance retrieval using lexical overlap, recency, and importance scoring
- prompt-friendly formatting for injecting retrieved memories into an agent context
- a lightweight CLI for adding, searching, listing, and exporting memories

The repository is ready for the next step: once the old agent's memory logic is described in
a Markdown document, the module can be adapted to match that behavior while staying
independent.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Quick Start

```python
from debate_agent_memory import AgentMemory, JsonlMemoryStore

memory = AgentMemory(JsonlMemoryStore("memory.jsonl"))

memory.add(
    content="The user prefers concise rebuttals with explicit evidence.",
    session_id="debate-001",
    agent_id="assistant-a",
    role="user",
    kind="preference",
    tags=["style", "evidence"],
    importance=0.8,
)

hits = memory.search(
    "How should I write the next rebuttal?",
    session_id="debate-001",
    top_k=3,
)

memory_block = memory.format_for_prompt(hits)
print(memory_block)
```

Example prompt block:

```text
Relevant memories:
1. [preference | importance=0.80 | tags=style,evidence] The user prefers concise rebuttals with explicit evidence.
```

## CLI

```bash
debate-memory add "The judge values direct clash on definitions." --session debate-001 --kind note --tag judging --importance 0.7
debate-memory search "definition clash" --session debate-001
debate-memory list --session debate-001
debate-memory export
```

By default the CLI stores data in `.memory/memories.jsonl`. Use `--store path/to/file.jsonl`
to choose another JSONL file, or pass a `.db`, `.sqlite`, or `.sqlite3` path to use SQLite.

## Design Notes

The module separates memory into four layers:

1. `MemoryRecord`: a structured event or fact.
2. `MemoryStore`: persistence and filtering.
3. `MemoryRetriever`: scoring and ranking.
4. `AgentMemory`: the high-level facade used by an agent.

This keeps the old agent integration simple: it can call `memory.add(...)` when important
events happen, call `memory.search(...)` before generating a response, and inject
`memory.format_for_prompt(...)` into the prompt.

## When To Use This Instead Of Raw Context Concatenation

Use this module when conversations are long, repeated across sessions, or contain facts that
should outlive a single context window. Direct context concatenation is simpler for short
single-session conversations, but it becomes expensive and noisy as history grows.

The final decision should be made after the old memory logic is documented and tested against
real debate-agent traces.
