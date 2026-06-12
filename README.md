# Debate Agent Memory

`debate-agent-memory` is a decoupled Python memory module for debate-style LLM agents.
It is intentionally framework-free, so an existing agent can add memory without binding
itself to LangChain, LlamaIndex, a specific vector database, or a specific prompt layout.

The module currently provides:

- layered context memory: archive, working turns, summaries, ledger, unfinished state, and focus
- structured records for turns, summaries, claims, obligations, decisions, and notes
- compaction candidate selection and accepted-summary absorption
- context budget pressure estimation
- active context and full export rendering
- pluggable storage backends, with JSONL, SQLite, and in-memory implementations included
- local relevance retrieval using lexical overlap, recency, and importance scoring
- prompt-friendly formatting for injecting retrieved memories into an agent context
- a lightweight CLI for adding, searching, listing, and exporting memories

The main API is `ContextMemory`, which implements the layered design from
`CONTEXT_MEMORY_LOGIC.md`. The older `AgentMemory` API remains available as a lightweight
record/search helper.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Layered Context Memory

Use `ContextMemory` when building a long-running debate agent that needs recent exact
conversation plus compressed older state.

```python
from debate_agent_memory import (
    ContextMemory,
    LedgerRecord,
    StaticBrief,
    SummaryRecord,
)

memory = ContextMemory(
    session_id="debate-001",
    static_brief=StaticBrief(
        topic="Public transit funding",
        agent_role="negative debater",
        goals=["Answer unresolved claims", "Preserve definitions and burdens"],
        style_requirements=["Concise", "Evidence-first"],
    ),
)

user_turn = memory.add_external_turn(
    "Transit funding reduces congestion because more riders leave cars.",
    speaker="opponent",
    speaker_kind="opponent",
    phase="constructive",
)

memory.add_ledger_entry(
    LedgerRecord(
        speaker="opponent",
        claim="Transit funding reduces congestion.",
        reasoning="More riders leave cars.",
        source_turn_ids=[user_turn.id],
        phase="constructive",
    )
)

agent_turn = memory.add_agent_turn(
    "That link needs evidence: mode shift is not automatic.",
    responds_to=user_turn.id,
    phase="constructive",
)

payload = memory.build_context(latest_query="How should I answer congestion?")
```

The returned `payload` keeps the context pieces separate:

- `payload.static_brief`
- `payload.summaries`
- `payload.ledger`
- `payload.current_focus`
- `payload.working_turns`
- `payload.budget`

That separation lets the caller choose how to render prompts while keeping the memory state
debuggable.

## Compaction Flow

This module does not invent summaries by itself. It selects safe candidates and lets the
caller provide an accepted summary from an LLM, rule-based summarizer, or human process.

```python
from debate_agent_memory import PressureLevel

candidates = memory.select_compaction_candidates(PressureLevel.HARD)

summary = SummaryRecord(
    source_turn_ids=[turn.id for turn in candidates],
    start_turn_id=candidates[0].id,
    end_turn_id=candidates[-1].id,
    title="Opening congestion exchange",
    summary="Opponent claimed transit funding reduces congestion; agent challenged the mode-shift link.",
    open_obligations=["Answer whether funding is sufficient to cause mode shift."],
    challenge_targets=["Causality from funding to congestion reduction."],
)

memory.absorb_summary(summary)
```

After `absorb_summary(...)`:

- source turns remain in `archive`
- source turns are marked `compressed=True`
- source turns are removed from active `working_turns`
- the summary becomes active historical memory

## Unfinished Output

Interrupted agent output is stored both as an exposed agent turn and as unfinished state:

```python
memory.add_agent_turn(
    "I would answer fairness by distinguishing access",
    interrupted=True,
    incomplete=True,
)

focus = memory.build_current_focus()
assert focus.unfinished_state is not None
```

A later complete agent turn clears unfinished state automatically.

## Export And Snapshots

```python
memory.save_snapshot("memory-state.json")
loaded = ContextMemory.load_snapshot("memory-state.json")

print(memory.export_markdown())
```

The export includes static brief, current focus, summaries, ledger, working turns, full raw
archive, compression markers, unfinished state, and budget history.

## Lightweight Memory API

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

1. `ContextMemory`: layered session state for archive, working turns, summaries, ledger, focus.
2. `context_models`: structured records for turns, summaries, ledger entries, budgets, payloads.
3. `AgentMemory`: a lightweight event/fact record and search facade.
4. `MemoryStore`: optional persistence for the lightweight API.
5. `MemoryRetriever`: scoring and ranking for lightweight memories.

This keeps the old agent integration simple: it can call `memory.add(...)` when important
events happen, call `memory.search(...)` before generating a response, and inject
`memory.format_for_prompt(...)` into the prompt. For the full design, use
`ContextMemory.add_external_turn(...)`, `ContextMemory.add_agent_turn(...)`,
`ContextMemory.build_context(...)`, and `ContextMemory.absorb_summary(...)`.

## When To Use This Instead Of Raw Context Concatenation

Use this module when conversations are long, repeated across sessions, or contain facts that
should outlive a single context window. Direct context concatenation is simpler for short
single-session conversations, but it becomes expensive and noisy as history grows.

The final decision should be made after the old memory logic is documented and tested against
real debate-agent traces.
