"""Layered context memory implementation."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional

from debate_agent_memory.context_models import (
    BudgetSnapshot,
    ContextPayload,
    CurrentFocus,
    LedgerRecord,
    LedgerStatus,
    PressureLevel,
    StaticBrief,
    SummaryRecord,
    TurnRecord,
    UnfinishedState,
)


class ContextMemory:
    """A layered scratchpad for long-running interactive agents.

    The class keeps raw archive turns as the source of truth while exposing bounded working
    memory, structured summaries, a durable ledger, unfinished state, and context rendering.
    It does not call an LLM internally; callers provide accepted summaries or ledger entries.
    """

    def __init__(
        self,
        session_id: str,
        static_brief: Optional[StaticBrief] = None,
        max_working_turns: int = 12,
        preserve_recent_turns: int = 6,
        max_context_size: int = 12000,
        reserved_output_size: int = 2000,
        soft_threshold: float = 0.65,
        hard_threshold: float = 0.82,
        emergency_threshold: float = 0.95,
    ) -> None:
        self.session_id = session_id
        self.static_brief = static_brief or StaticBrief()
        self.max_working_turns = max_working_turns
        self.preserve_recent_turns = preserve_recent_turns
        self.max_context_size = max_context_size
        self.reserved_output_size = reserved_output_size
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.emergency_threshold = emergency_threshold

        self.archive: List[TurnRecord] = []
        self.working_turn_ids: List[str] = []
        self.summaries: List[SummaryRecord] = []
        self.ledger: List[LedgerRecord] = []
        self.unfinished_state: Optional[UnfinishedState] = None
        self.usage_or_budget_history: List[BudgetSnapshot] = []
        self._seen_turn_fingerprints: set[str] = set()
        self._lock = RLock()

    def add_external_turn(
        self,
        text: str,
        speaker: str = "user",
        speaker_role: Optional[str] = None,
        speaker_side: Optional[str] = None,
        speaker_kind: str = "user",
        phase: Optional[str] = None,
        interrupted: bool = False,
        incomplete: bool = False,
        responds_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_if_duplicate: bool = True,
        skip_if_self_echo: bool = True,
    ) -> Optional[TurnRecord]:
        return self._add_turn(
            text=text,
            speaker=speaker,
            speaker_role=speaker_role,
            speaker_side=speaker_side,
            speaker_kind=speaker_kind,
            phase=phase,
            interrupted=interrupted,
            incomplete=incomplete,
            responds_to=responds_to,
            metadata=metadata,
            skip_if_duplicate=skip_if_duplicate,
            skip_if_self_echo=skip_if_self_echo,
        )

    def add_agent_turn(
        self,
        text: str,
        speaker: str = "agent",
        speaker_role: Optional[str] = "assistant",
        speaker_side: Optional[str] = None,
        phase: Optional[str] = None,
        interrupted: bool = False,
        incomplete: bool = False,
        responds_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnRecord:
        turn = self._add_turn(
            text=text,
            speaker=speaker,
            speaker_role=speaker_role,
            speaker_side=speaker_side,
            speaker_kind="agent",
            phase=phase,
            interrupted=interrupted,
            incomplete=incomplete,
            responds_to=responds_to,
            metadata=metadata,
            skip_if_duplicate=False,
            skip_if_self_echo=False,
        )
        if turn is None:
            raise RuntimeError("Agent turns are not skipped as duplicates.")
        if interrupted or incomplete:
            self.save_unfinished_state(text, source_turn_id=turn.id, reason="interrupted")
        else:
            self.clear_unfinished_state()
        return turn

    def skip_self_echo(self, text: str, speaker: str = "agent") -> bool:
        """Return True when text looks like the latest agent output re-ingested as input."""

        normalized = self._fingerprint(speaker, text)
        with self._lock:
            return normalized in self._seen_turn_fingerprints or self._is_agent_echo(text)

    def add_ledger_entry(self, entry: LedgerRecord) -> LedgerRecord:
        with self._lock:
            for index, existing in enumerate(self.ledger):
                same_key = existing.dedupe_key == entry.dedupe_key
                overlapping_sources = bool(set(existing.source_turn_ids) & set(entry.source_turn_ids))
                if same_key or overlapping_sources:
                    merged = existing.merge(entry)
                    self.ledger[index] = merged
                    return merged
            self.ledger.append(entry)
            return entry

    def update_ledger_status(self, ledger_id: str, status: LedgerStatus | str) -> LedgerRecord:
        status = LedgerStatus(status)
        with self._lock:
            for index, existing in enumerate(self.ledger):
                if existing.id == ledger_id:
                    updated = LedgerRecord(
                        id=existing.id,
                        speaker=existing.speaker,
                        speaker_role=existing.speaker_role,
                        speaker_side=existing.speaker_side,
                        claim=existing.claim,
                        evidence=existing.evidence,
                        reasoning=existing.reasoning,
                        weakness=existing.weakness,
                        status=status,
                        source_turn_ids=existing.source_turn_ids,
                        phase=existing.phase,
                        category=existing.category,
                        created_at=existing.created_at,
                        metadata=existing.metadata,
                    )
                    self.ledger[index] = updated
                    return updated
        raise KeyError(f"Ledger record not found: {ledger_id}")

    def save_unfinished_state(
        self,
        text: str,
        source_turn_id: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnfinishedState:
        with self._lock:
            self.unfinished_state = UnfinishedState(
                text=text,
                source_turn_id=source_turn_id,
                reason=reason,
                metadata=dict(metadata or {}),
            )
            return self.unfinished_state

    def clear_unfinished_state(self) -> None:
        with self._lock:
            self.unfinished_state = None

    def compute_pressure(self) -> BudgetSnapshot:
        with self._lock:
            estimated = self._estimate_active_size()
            total = estimated + self.reserved_output_size
            ratio = total / max(self.max_context_size, 1)
            if ratio >= self.emergency_threshold:
                level = PressureLevel.EMERGENCY
            elif ratio >= self.hard_threshold:
                level = PressureLevel.HARD
            elif ratio >= self.soft_threshold:
                level = PressureLevel.SOFT
            else:
                level = PressureLevel.NORMAL
            snapshot = BudgetSnapshot(
                estimated_active_size=estimated,
                reserved_output_size=self.reserved_output_size,
                maximum_context_size=self.max_context_size,
                soft_threshold=self.soft_threshold,
                hard_threshold=self.hard_threshold,
                emergency_threshold=self.emergency_threshold,
                pressure_level=level,
                reason=f"active+reserved ratio={ratio:.2f}",
            )
            self.usage_or_budget_history.append(snapshot)
            return snapshot

    def select_compaction_candidates(
        self,
        pressure_level: Optional[PressureLevel | str] = None,
        min_batch_size: int = 2,
    ) -> List[TurnRecord]:
        with self._lock:
            level = (
                PressureLevel(pressure_level)
                if pressure_level
                else self.compute_pressure().pressure_level
            )
            preserve_count = self._preserve_count_for(level)
            working = self.working_turns
            candidates = working[: max(0, len(working) - preserve_count)]
            if len(candidates) < min_batch_size and level != PressureLevel.EMERGENCY:
                return []
            return candidates

    def absorb_summary(
        self,
        summary: SummaryRecord,
        ledger_entries: Optional[Iterable[LedgerRecord]] = None,
    ) -> SummaryRecord:
        with self._lock:
            source_ids = set(summary.source_turn_ids)
            self.summaries.append(summary)
            self.archive = [
                turn.mark_compressed() if turn.id in source_ids else turn
                for turn in self.archive
            ]
            self.working_turn_ids = [
                turn_id for turn_id in self.working_turn_ids if turn_id not in source_ids
            ]
            for entry in ledger_entries or []:
                self.add_ledger_entry(entry)
            return summary

    def consolidate_summaries(
        self,
        source_summary_ids: Iterable[str],
        consolidated: SummaryRecord,
        ledger_entries: Optional[Iterable[LedgerRecord]] = None,
    ) -> SummaryRecord:
        with self._lock:
            source_ids = set(source_summary_ids)
            existing_ids = {summary.id for summary in self.summaries}
            missing = source_ids - existing_ids
            if missing:
                raise KeyError(f"Summary record(s) not found: {sorted(missing)}")
            self.summaries = [summary for summary in self.summaries if summary.id not in source_ids]
            self.summaries.append(consolidated)
            for entry in ledger_entries or []:
                self.add_ledger_entry(entry)
            return consolidated

    def build_context(
        self,
        latest_query: Optional[str] = None,
        max_ledger_items: int = 12,
        max_summaries: int = 8,
    ) -> ContextPayload:
        with self._lock:
            budget = self.compute_pressure()
            ledger = self._rank_ledger(latest_query, max_ledger_items)
            summaries = self.summaries[-max_summaries:]
            focus = self.build_current_focus(ledger_items=ledger[:6])
            return ContextPayload(
                static_brief=self.static_brief,
                summaries=summaries,
                ledger=ledger,
                current_focus=focus,
                working_turns=self.working_turns,
                budget=budget,
            )

    def build_short_context(self, latest_query: Optional[str] = None) -> ContextPayload:
        return self.build_context(latest_query=latest_query, max_ledger_items=5, max_summaries=3)

    def build_current_focus(
        self,
        ledger_items: Optional[List[LedgerRecord]] = None,
    ) -> CurrentFocus:
        with self._lock:
            working = self.working_turns
            recent_agent_turns = [turn for turn in working if turn.speaker_kind == "agent"][-2:]
            latest_external_turns = [turn for turn in working if turn.speaker_kind != "agent"][-3:]
            summaries = self.summaries[-3:]
            open_obligations = self._take_unique(
                item
                for summary in summaries
                for item in summary.open_obligations
            )
            unanswered = self._take_unique(
                item
                for summary in summaries
                for item in summary.unanswered_points
            )
            challenge_targets = self._take_unique(
                item
                for summary in summaries
                for item in summary.challenge_targets
            )
            support_targets = self._take_unique(
                item
                for summary in summaries
                for item in summary.support_targets
            )
            unresolved_ledger = [
                item
                for item in (ledger_items or self.ledger)
                if item.status
                in {
                    LedgerStatus.OPEN,
                    LedgerStatus.NEEDS_EVIDENCE,
                    LedgerStatus.NEEDS_CLARIFICATION,
                }
            ][:6]
            return CurrentFocus(
                recent_agent_turns=recent_agent_turns,
                latest_external_turns=latest_external_turns,
                unfinished_state=self.unfinished_state,
                open_obligations=open_obligations,
                unanswered_challenges=unanswered,
                challenge_targets=challenge_targets,
                support_targets=support_targets,
                ledger_items=unresolved_ledger,
            )

    def export_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "static_brief": self.static_brief.to_dict(),
                "current_focus": self.build_current_focus().to_dict(),
                "summaries": [summary.to_dict() for summary in self.summaries],
                "ledger": [entry.to_dict() for entry in self.ledger],
                "working_turns": [turn.to_dict() for turn in self.working_turns],
                "archive": [turn.to_dict() for turn in self.archive],
                "unfinished_state": (
                    self.unfinished_state.to_dict() if self.unfinished_state else None
                ),
                "usage_or_budget_history": [
                    snapshot.to_dict() for snapshot in self.usage_or_budget_history
                ],
            }

    def export_markdown(self) -> str:
        state = self.export_state()
        lines = [f"# Memory Export: {self.session_id}", ""]
        brief = state["static_brief"]
        if brief.get("topic"):
            lines.extend(["## Static Brief", "", f"- Topic: {brief['topic']}"])
        lines.extend(["", "## Current Focus", ""])
        focus = state["current_focus"]
        for item in focus["open_obligations"]:
            lines.append(f"- Open obligation: {item}")
        for item in focus["unanswered_challenges"]:
            lines.append(f"- Unanswered: {item}")
        if focus["unfinished_state"]:
            lines.append(f"- Unfinished: {focus['unfinished_state']['text']}")
        lines.extend(["", "## Summaries", ""])
        for summary in self.summaries:
            lines.append(f"### {summary.title or summary.id}")
            lines.append(summary.summary)
            lines.append("")
        lines.extend(["## Ledger", ""])
        for entry in self.ledger:
            lines.append(f"- [{entry.status.value}] {entry.claim}")
        lines.extend(["", "## Working Turns", ""])
        for turn in self.working_turns:
            lines.append(f"- {turn.speaker} ({turn.speaker_kind}): {turn.text}")
        lines.extend(["", "## Archive", ""])
        for turn in self.archive:
            marker = " compressed" if turn.compressed else ""
            lines.append(f"- #{turn.sequence}{marker} {turn.speaker}: {turn.text}")
        return "\n".join(lines).strip() + "\n"

    def save_snapshot(self, path: str | Path) -> None:
        data = json.dumps(self.export_state(), ensure_ascii=False, indent=2)
        Path(path).write_text(data, encoding="utf-8")

    @classmethod
    def load_snapshot(cls, path: str | Path) -> "ContextMemory":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        memory = cls(
            session_id=payload["session_id"],
            static_brief=StaticBrief.from_dict(payload.get("static_brief") or {}),
        )
        memory.archive = [TurnRecord.from_dict(item) for item in payload.get("archive", [])]
        memory.working_turn_ids = [
            item["id"] for item in payload.get("working_turns", [])
        ]
        memory.summaries = [
            SummaryRecord.from_dict(item) for item in payload.get("summaries", [])
        ]
        memory.ledger = [LedgerRecord.from_dict(item) for item in payload.get("ledger", [])]
        memory.unfinished_state = UnfinishedState.from_dict(payload.get("unfinished_state"))
        memory.usage_or_budget_history = [
            BudgetSnapshot.from_dict(item)
            for item in payload.get("usage_or_budget_history", [])
        ]
        memory._seen_turn_fingerprints = {
            memory._fingerprint(turn.speaker, turn.text) for turn in memory.archive
        }
        return memory

    @property
    def working_turns(self) -> List[TurnRecord]:
        lookup = {turn.id: turn for turn in self.archive}
        return [lookup[turn_id] for turn_id in self.working_turn_ids if turn_id in lookup]

    def _add_turn(
        self,
        text: str,
        speaker: str,
        speaker_role: Optional[str],
        speaker_side: Optional[str],
        speaker_kind: str,
        phase: Optional[str],
        interrupted: bool,
        incomplete: bool,
        responds_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        skip_if_duplicate: bool,
        skip_if_self_echo: bool,
    ) -> Optional[TurnRecord]:
        normalized_text = " ".join(text.split())
        fingerprint = self._fingerprint(speaker, normalized_text)
        with self._lock:
            if skip_if_self_echo and self._is_agent_echo(normalized_text):
                return None
            if skip_if_duplicate and fingerprint in self._seen_turn_fingerprints:
                return None
            turn = TurnRecord(
                sequence=len(self.archive) + 1,
                speaker=speaker,
                speaker_role=speaker_role,
                speaker_side=speaker_side,
                speaker_kind=speaker_kind,
                phase=phase,
                text=normalized_text,
                interrupted=interrupted,
                incomplete=incomplete,
                responds_to=responds_to,
                metadata=dict(metadata or {}),
            )
            self.archive.append(turn)
            self.working_turn_ids.append(turn.id)
            self._seen_turn_fingerprints.add(fingerprint)
            return turn

    def _preserve_count_for(self, level: PressureLevel) -> int:
        if level == PressureLevel.EMERGENCY:
            return max(1, self.preserve_recent_turns // 3)
        if level == PressureLevel.HARD:
            return max(1, self.preserve_recent_turns // 2)
        return self.preserve_recent_turns

    def _estimate_active_size(self) -> int:
        payload = {
            "static_brief": self.static_brief.to_dict(),
            "summaries": [summary.to_dict() for summary in self.summaries],
            "ledger": [entry.to_dict() for entry in self.ledger],
            "unfinished_state": (
                self.unfinished_state.to_dict() if self.unfinished_state else None
            ),
            "working_turns": [turn.to_dict() for turn in self.working_turns],
        }
        return len(json.dumps(payload, ensure_ascii=False))

    def _rank_ledger(self, latest_query: Optional[str], max_items: int) -> List[LedgerRecord]:
        query_terms = set((latest_query or "").lower().split())

        def score(entry: LedgerRecord) -> tuple[int, int, Any]:
            unresolved = entry.status in {
                LedgerStatus.OPEN,
                LedgerStatus.NEEDS_EVIDENCE,
                LedgerStatus.NEEDS_CLARIFICATION,
            }
            text = " ".join(
                value or ""
                for value in [entry.claim, entry.evidence, entry.reasoning, entry.weakness]
            ).lower()
            overlap = len(query_terms & set(text.split())) if query_terms else 0
            return (1 if unresolved else 0, overlap, entry.updated_at)

        return sorted(self.ledger, key=score, reverse=True)[:max_items]

    def _fingerprint(self, speaker: str, text: str) -> str:
        return f"{speaker.strip().lower()}:{' '.join(text.lower().split())}"

    def _is_agent_echo(self, text: str) -> bool:
        normalized = " ".join(text.lower().split())
        if not normalized:
            return False
        for turn in reversed(self.archive[-5:]):
            if turn.speaker_kind != "agent":
                continue
            agent_text = " ".join(turn.text.lower().split())
            if normalized == agent_text:
                return True
        return False

    def _take_unique(self, values: Iterable[str], limit: int = 8) -> List[str]:
        seen = set()
        result = []
        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
            if len(result) >= limit:
                break
        return result
