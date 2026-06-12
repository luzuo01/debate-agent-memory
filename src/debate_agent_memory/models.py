"""Data models used by the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(frozen=True)
class MemoryRecord:
    """A single memory item captured from an agent interaction or derived note."""

    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    role: Optional[str] = None
    kind: str = "note"
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        content = self.content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")

        importance = max(0.0, min(1.0, float(self.importance)))
        normalized_tags = sorted({tag.strip() for tag in self.tags if tag.strip()})

        object.__setattr__(self, "content", content)
        object.__setattr__(self, "importance", importance)
        object.__setattr__(self, "tags", normalized_tags)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "role": self.role,
            "kind": self.kind,
            "content": self.content,
            "tags": list(self.tags),
            "importance": self.importance,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=payload["id"],
            session_id=payload.get("session_id"),
            agent_id=payload.get("agent_id"),
            role=payload.get("role"),
            kind=payload.get("kind", "note"),
            content=payload["content"],
            tags=list(payload.get("tags") or []),
            importance=float(payload.get("importance", 0.5)),
            metadata=dict(payload.get("metadata") or {}),
            created_at=parse_datetime(payload["created_at"]),
            updated_at=parse_datetime(payload.get("updated_at", payload["created_at"])),
        )

    def with_updates(self, **updates: Any) -> "MemoryRecord":
        data = self.to_dict()
        data.update(updates)
        data["updated_at"] = utc_now().isoformat()
        return MemoryRecord.from_dict(data)


@dataclass(frozen=True)
class MemoryFilter:
    """Optional constraints applied before ranking memories."""

    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    role: Optional[str] = None
    kinds: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_importance: Optional[float] = None

    def matches(self, record: MemoryRecord) -> bool:
        if self.session_id is not None and record.session_id != self.session_id:
            return False
        if self.agent_id is not None and record.agent_id != self.agent_id:
            return False
        if self.role is not None and record.role != self.role:
            return False
        if self.kinds is not None and record.kind not in self.kinds:
            return False
        if self.tags is not None and not set(self.tags).issubset(record.tags):
            return False
        if self.min_importance is not None and record.importance < self.min_importance:
            return False
        return True

    @classmethod
    def from_values(
        cls,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        role: Optional[str] = None,
        kinds: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        min_importance: Optional[float] = None,
    ) -> "MemoryFilter":
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            role=role,
            kinds=list(kinds) if kinds is not None else None,
            tags=list(tags) if tags is not None else None,
            min_importance=min_importance,
        )


@dataclass(frozen=True)
class MemorySearchResult:
    """A ranked memory search hit."""

    record: MemoryRecord
    score: float
    reasons: List[str] = field(default_factory=list)
