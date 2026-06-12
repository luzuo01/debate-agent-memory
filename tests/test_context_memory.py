from debate_agent_memory import (
    ContextMemory,
    LedgerRecord,
    LedgerStatus,
    PressureLevel,
    StaticBrief,
    SummaryRecord,
)


def test_context_memory_keeps_archive_and_working_turns() -> None:
    memory = ContextMemory(
        session_id="round-1",
        static_brief=StaticBrief(topic="Public transit funding", agent_role="negative"),
    )

    user_turn = memory.add_external_turn("We should fund transit because it reduces traffic.")
    agent_turn = memory.add_agent_turn("That assumes induced demand disappears.")

    assert user_turn is not None
    assert len(memory.archive) == 2
    assert [turn.id for turn in memory.working_turns] == [user_turn.id, agent_turn.id]
    assert memory.export_state()["static_brief"]["topic"] == "Public transit funding"


def test_self_echo_and_duplicate_turns_are_skipped() -> None:
    memory = ContextMemory(session_id="round-1")
    memory.add_agent_turn("I will answer the solvency claim.")

    echo = memory.add_external_turn("I will answer the solvency claim.")
    first = memory.add_external_turn("What about solvency?", speaker="opponent")
    duplicate = memory.add_external_turn("What about solvency?", speaker="opponent")

    assert echo is None
    assert first is not None
    assert duplicate is None
    assert len(memory.archive) == 2


def test_ledger_entries_are_cautiously_deduplicated() -> None:
    memory = ContextMemory(session_id="round-1")
    turn = memory.add_external_turn("Transit funding reduces congestion.", speaker="opponent")
    assert turn is not None

    first = memory.add_ledger_entry(
        LedgerRecord(
            speaker="opponent",
            claim="Transit funding reduces congestion.",
            source_turn_ids=[turn.id],
        )
    )
    merged = memory.add_ledger_entry(
        LedgerRecord(
            speaker="opponent",
            claim="Transit funding reduces congestion.",
            evidence="They cite commute-time reductions.",
            status=LedgerStatus.NEEDS_EVIDENCE,
            source_turn_ids=[turn.id],
        )
    )

    assert first.id == merged.id
    assert len(memory.ledger) == 1
    assert memory.ledger[0].evidence == "They cite commute-time reductions."
    assert memory.ledger[0].status == LedgerStatus.NEEDS_EVIDENCE


def test_compaction_candidates_and_absorb_summary() -> None:
    memory = ContextMemory(session_id="round-1", preserve_recent_turns=2)
    turns = [
        memory.add_external_turn(f"Opponent claim {index}", speaker="opponent")
        for index in range(5)
    ]
    assert all(turn is not None for turn in turns)

    candidates = memory.select_compaction_candidates(PressureLevel.HARD)
    candidate_ids = [turn.id for turn in candidates]

    assert candidate_ids == [turn.id for turn in turns[:4]]

    summary = SummaryRecord(
        source_turn_ids=candidate_ids,
        start_turn_id=candidate_ids[0],
        end_turn_id=candidate_ids[-1],
        title="Opening claims",
        summary="Opponent made several opening claims.",
        open_obligations=["Answer the congestion warrant."],
        unanswered_points=["Whether funding is sufficient."],
        challenge_targets=["Causality from funding to congestion reduction."],
    )
    memory.absorb_summary(summary)

    assert len(memory.summaries) == 1
    assert [turn.id for turn in memory.working_turns] == [turns[-1].id]
    compressed = [turn for turn in memory.archive if turn.id in candidate_ids]
    assert all(turn.compressed for turn in compressed)


def test_unfinished_state_and_context_payload() -> None:
    memory = ContextMemory(session_id="round-1")
    user_turn = memory.add_external_turn("Answer their definition of fairness.")
    agent_turn = memory.add_agent_turn(
        "I would answer fairness by distinguishing access",
        interrupted=True,
        incomplete=True,
        responds_to=user_turn.id if user_turn else None,
    )

    payload = memory.build_context(latest_query="fairness definition")

    assert agent_turn.interrupted is True
    assert payload.current_focus.unfinished_state is not None
    assert "distinguishing access" in payload.current_focus.unfinished_state.text

    memory.add_agent_turn("Completed answer.", interrupted=False, incomplete=False)
    assert memory.build_context().current_focus.unfinished_state is None


def test_snapshot_roundtrip(tmp_path) -> None:
    path = tmp_path / "memory.json"
    memory = ContextMemory(session_id="round-1")
    turn = memory.add_external_turn("Preserve this turn.", speaker="user")
    assert turn is not None
    memory.add_ledger_entry(LedgerRecord(claim="A durable claim.", source_turn_ids=[turn.id]))
    memory.save_snapshot(path)

    loaded = ContextMemory.load_snapshot(path)

    assert loaded.session_id == "round-1"
    assert loaded.archive[0].text == "Preserve this turn."
    assert loaded.ledger[0].claim == "A durable claim."
