from debate_agent_memory import AgentMemory, InMemoryStore, JsonlMemoryStore, SQLiteMemoryStore


def test_add_and_search_memory() -> None:
    memory = AgentMemory(InMemoryStore())
    memory.add(
        "The user wants concise rebuttals with strong evidence.",
        session_id="s1",
        kind="preference",
        tags=["style"],
        importance=0.9,
    )
    memory.add(
        "The debate topic is public transportation funding.",
        session_id="s2",
        kind="topic",
        tags=["topic"],
        importance=0.4,
    )

    results = memory.search("How should I write evidence rebuttal?", session_id="s1")

    assert len(results) == 1
    assert results[0].record.kind == "preference"
    assert "evidence" in results[0].record.content


def test_prompt_formatter() -> None:
    memory = AgentMemory(InMemoryStore())
    memory.add(
        "The judge dislikes vague impact comparison.",
        session_id="round-1",
        kind="judge_note",
        tags=["judge"],
        importance=0.7,
    )

    block = memory.format_for_prompt(memory.search("impact comparison", session_id="round-1"))

    assert block.startswith("Relevant memories:")
    assert "judge_note" in block
    assert "impact comparison" in block


def test_jsonl_store_persists_records(tmp_path) -> None:
    store_path = tmp_path / "memories.jsonl"
    memory = AgentMemory(JsonlMemoryStore(store_path))
    created = memory.add("Prefer direct clash on definitions.", session_id="s1", tags=["debate"])

    reloaded = AgentMemory(JsonlMemoryStore(store_path))
    records = reloaded.list(session_id="s1")

    assert len(records) == 1
    assert records[0].id == created.id
    assert records[0].tags == ["debate"]


def test_update_and_delete_memory() -> None:
    memory = AgentMemory(InMemoryStore())
    created = memory.add("Old note", importance=0.1)

    updated = memory.update(created.id, content="New note", importance=0.8)

    assert updated.content == "New note"
    assert updated.importance == 0.8
    assert memory.delete(created.id) is True
    assert memory.get(created.id) is None


def test_sqlite_store_persists_records(tmp_path) -> None:
    store_path = tmp_path / "memories.sqlite3"
    memory = AgentMemory(SQLiteMemoryStore(store_path))
    created = memory.add("Opponent dropped solvency evidence.", session_id="s1", tags=["flow"])

    reloaded = AgentMemory(SQLiteMemoryStore(store_path))
    records = reloaded.list(session_id="s1")

    assert len(records) == 1
    assert records[0].id == created.id
    assert records[0].content == "Opponent dropped solvency evidence."
