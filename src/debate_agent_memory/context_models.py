"""Layered context-memory records described by the design document."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from debate_agent_memory.models import parse_datetime, utc_now


class PressureLevel(str, Enum):
    NORMAL = "normal"
    SOFT = "soft"
    HARD = "hard"
    EMERGENCY = "emergency"


class LedgerStatus(str, Enum):
    OPEN = "open"
    ANSWERED = "answered"
    REFUTED = "refuted"
    ACCEPTED = "accepted"
    SUPERSEDED = "superseded"
    ABANDONED = "abandoned"
    NEEDS_EVIDENCE = "needs_evidence"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True)
class StaticBrief:
    topic: Optional[str] = None
    agent_role: Optional[str] = None
    user_role: Optional[str] = None
    rules: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    style_requirements: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "agent_role": self.agent_role,
            "user_role": self.user_role,
            "rules": list(self.rules),
            "goals": list(self.goals),
            "style_requirements": list(self.style_requirements),
            "procedures": list(self.procedures),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StaticBrief":
        return cls(
            topic=payload.get("topic"),
            agent_role=payload.get("agent_role"),
            user_role=payload.get("user_role"),
            rules=list(payload.get("rules") or []),
            goals=list(payload.get("goals") or []),
            style_requirements=list(payload.get("style_requirements") or []),
            procedures=list(payload.get("procedures") or []),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TurnRecord:
    text: str
    speaker: str
    speaker_role: Optional[str] = None
    speaker_side: Optional[str] = None
    speaker_kind: str = "unknown"
    phase: Optional[str] = None
    interrupted: bool = False
    incomplete: bool = False
    responds_to: Optional[str] = None
    compressed: bool = False
    id: str = field(default_factory=lambda: f"turn-{uuid4()}")
    sequence: int = 0
    created_at: Any = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = self.text.strip()
        if not text:
            raise ValueError("Turn text cannot be empty.")
        object.__setattr__(self, "text", text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "speaker": self.speaker,
            "speaker_role": self.speaker_role,
            "speaker_side": self.speaker_side,
            "speaker_kind": self.speaker_kind,
            "phase": self.phase,
            "text": self.text,
            "interrupted": self.interrupted,
            "incomplete": self.incomplete,
            "responds_to": self.responds_to,
            "compressed": self.compressed,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TurnRecord":
        return cls(
            id=payload["id"],
            sequence=int(payload.get("sequence", 0)),
            speaker=payload["speaker"],
            speaker_role=payload.get("speaker_role"),
            speaker_side=payload.get("speaker_side"),
            speaker_kind=payload.get("speaker_kind", "unknown"),
            phase=payload.get("phase"),
            text=payload["text"],
            interrupted=bool(payload.get("interrupted", False)),
            incomplete=bool(payload.get("incomplete", False)),
            responds_to=payload.get("responds_to"),
            compressed=bool(payload.get("compressed", False)),
            created_at=parse_datetime(payload["created_at"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    def mark_compressed(self) -> "TurnRecord":
        data = self.to_dict()
        data["compressed"] = True
        return TurnRecord.from_dict(data)


@dataclass(frozen=True)
class SummaryRecord:
    source_turn_ids: List[str]
    title: str
    summary: str
    start_turn_id: Optional[str] = None
    end_turn_id: Optional[str] = None
    phase: Optional[str] = None
    open_obligations: List[str] = field(default_factory=list)
    supported_points: List[str] = field(default_factory=list)
    unanswered_points: List[str] = field(default_factory=list)
    challenge_targets: List[str] = field(default_factory=list)
    support_targets: List[str] = field(default_factory=list)
    definitions_and_commitments: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"summary-{uuid4()}")
    created_at: Any = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_turn_ids:
            raise ValueError("Summary must reference at least one source turn.")
        if not self.summary.strip():
            raise ValueError("Summary text cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_turn_ids": list(self.source_turn_ids),
            "start_turn_id": self.start_turn_id,
            "end_turn_id": self.end_turn_id,
            "title": self.title,
            "summary": self.summary,
            "phase": self.phase,
            "open_obligations": list(self.open_obligations),
            "supported_points": list(self.supported_points),
            "unanswered_points": list(self.unanswered_points),
            "challenge_targets": list(self.challenge_targets),
            "support_targets": list(self.support_targets),
            "definitions_and_commitments": list(self.definitions_and_commitments),
            "evidence": list(self.evidence),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SummaryRecord":
        return cls(
            id=payload["id"],
            source_turn_ids=list(payload.get("source_turn_ids") or []),
            start_turn_id=payload.get("start_turn_id"),
            end_turn_id=payload.get("end_turn_id"),
            title=payload.get("title", ""),
            summary=payload["summary"],
            phase=payload.get("phase"),
            open_obligations=list(payload.get("open_obligations") or []),
            supported_points=list(payload.get("supported_points") or []),
            unanswered_points=list(payload.get("unanswered_points") or []),
            challenge_targets=list(payload.get("challenge_targets") or []),
            support_targets=list(payload.get("support_targets") or []),
            definitions_and_commitments=list(payload.get("definitions_and_commitments") or []),
            evidence=list(payload.get("evidence") or []),
            created_at=parse_datetime(payload["created_at"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LedgerRecord:
    claim: str
    speaker: Optional[str] = None
    speaker_role: Optional[str] = None
    speaker_side: Optional[str] = None
    evidence: Optional[str] = None
    reasoning: Optional[str] = None
    weakness: Optional[str] = None
    status: LedgerStatus = LedgerStatus.OPEN
    source_turn_ids: List[str] = field(default_factory=list)
    phase: Optional[str] = None
    category: Optional[str] = None
    id: str = field(default_factory=lambda: f"ledger-{uuid4()}")
    created_at: Any = field(default_factory=utc_now)
    updated_at: Any = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim = self.claim.strip()
        if not claim:
            raise ValueError("Ledger claim cannot be empty.")
        object.__setattr__(self, "claim", claim)
        if isinstance(self.status, str):
            object.__setattr__(self, "status", LedgerStatus(self.status))

    @property
    def dedupe_key(self) -> str:
        speaker = (self.speaker or "").strip().lower()
        claim = " ".join(self.claim.lower().split())
        return f"{speaker}:{claim}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "speaker": self.speaker,
            "speaker_role": self.speaker_role,
            "speaker_side": self.speaker_side,
            "claim": self.claim,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "weakness": self.weakness,
            "status": self.status.value,
            "source_turn_ids": list(self.source_turn_ids),
            "phase": self.phase,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LedgerRecord":
        return cls(
            id=payload["id"],
            speaker=payload.get("speaker"),
            speaker_role=payload.get("speaker_role"),
            speaker_side=payload.get("speaker_side"),
            claim=payload["claim"],
            evidence=payload.get("evidence"),
            reasoning=payload.get("reasoning"),
            weakness=payload.get("weakness"),
            status=LedgerStatus(payload.get("status", LedgerStatus.OPEN.value)),
            source_turn_ids=list(payload.get("source_turn_ids") or []),
            phase=payload.get("phase"),
            category=payload.get("category"),
            created_at=parse_datetime(payload["created_at"]),
            updated_at=parse_datetime(payload.get("updated_at", payload["created_at"])),
            metadata=dict(payload.get("metadata") or {}),
        )

    def merge(self, other: "LedgerRecord") -> "LedgerRecord":
        status = other.status if other.status != LedgerStatus.OPEN else self.status
        return LedgerRecord(
            id=self.id,
            speaker=self.speaker or other.speaker,
            speaker_role=self.speaker_role or other.speaker_role,
            speaker_side=self.speaker_side or other.speaker_side,
            claim=self.claim,
            evidence=self.evidence or other.evidence,
            reasoning=self.reasoning or other.reasoning,
            weakness=self.weakness or other.weakness,
            status=status,
            source_turn_ids=sorted(set(self.source_turn_ids) | set(other.source_turn_ids)),
            phase=self.phase or other.phase,
            category=self.category or other.category,
            created_at=self.created_at,
            updated_at=utc_now(),
            metadata={**other.metadata, **self.metadata},
        )


@dataclass(frozen=True)
class UnfinishedState:
    text: str
    source_turn_id: Optional[str] = None
    reason: Optional[str] = None
    created_at: Any = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_turn_id": self.source_turn_id,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["UnfinishedState"]:
        if payload is None:
            return None
        return cls(
            text=payload["text"],
            source_turn_id=payload.get("source_turn_id"),
            reason=payload.get("reason"),
            created_at=parse_datetime(payload["created_at"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class BudgetSnapshot:
    estimated_active_size: int
    reserved_output_size: int
    maximum_context_size: int
    soft_threshold: float
    hard_threshold: float
    emergency_threshold: float
    pressure_level: PressureLevel
    reason: str
    created_at: Any = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_active_size": self.estimated_active_size,
            "reserved_output_size": self.reserved_output_size,
            "maximum_context_size": self.maximum_context_size,
            "soft_threshold": self.soft_threshold,
            "hard_threshold": self.hard_threshold,
            "emergency_threshold": self.emergency_threshold,
            "pressure_level": self.pressure_level.value,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BudgetSnapshot":
        return cls(
            estimated_active_size=int(payload["estimated_active_size"]),
            reserved_output_size=int(payload["reserved_output_size"]),
            maximum_context_size=int(payload["maximum_context_size"]),
            soft_threshold=float(payload["soft_threshold"]),
            hard_threshold=float(payload["hard_threshold"]),
            emergency_threshold=float(payload["emergency_threshold"]),
            pressure_level=PressureLevel(payload["pressure_level"]),
            reason=payload["reason"],
            created_at=parse_datetime(payload["created_at"]),
        )


@dataclass(frozen=True)
class CurrentFocus:
    recent_agent_turns: List[TurnRecord] = field(default_factory=list)
    latest_external_turns: List[TurnRecord] = field(default_factory=list)
    unfinished_state: Optional[UnfinishedState] = None
    open_obligations: List[str] = field(default_factory=list)
    unanswered_challenges: List[str] = field(default_factory=list)
    challenge_targets: List[str] = field(default_factory=list)
    support_targets: List[str] = field(default_factory=list)
    ledger_items: List[LedgerRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recent_agent_turns": [turn.to_dict() for turn in self.recent_agent_turns],
            "latest_external_turns": [turn.to_dict() for turn in self.latest_external_turns],
            "unfinished_state": (
                self.unfinished_state.to_dict() if self.unfinished_state else None
            ),
            "open_obligations": list(self.open_obligations),
            "unanswered_challenges": list(self.unanswered_challenges),
            "challenge_targets": list(self.challenge_targets),
            "support_targets": list(self.support_targets),
            "ledger_items": [item.to_dict() for item in self.ledger_items],
        }


@dataclass(frozen=True)
class ContextPayload:
    static_brief: StaticBrief
    summaries: List[SummaryRecord]
    ledger: List[LedgerRecord]
    current_focus: CurrentFocus
    working_turns: List[TurnRecord]
    budget: BudgetSnapshot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "static_brief": self.static_brief.to_dict(),
            "summaries": [summary.to_dict() for summary in self.summaries],
            "ledger": [item.to_dict() for item in self.ledger],
            "current_focus": self.current_focus.to_dict(),
            "working_turns": [turn.to_dict() for turn in self.working_turns],
            "budget": self.budget.to_dict(),
        }

