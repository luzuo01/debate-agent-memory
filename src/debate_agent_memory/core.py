"""High-level facade for adding, retrieving, and formatting memories."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from debate_agent_memory.formatters import PromptMemoryFormatter
from debate_agent_memory.models import MemoryFilter, MemoryRecord, MemorySearchResult
from debate_agent_memory.retrievers import LexicalMemoryRetriever, summarize_memories
from debate_agent_memory.stores import InMemoryStore, MemoryStore


class AgentMemory:
    """Main integration point for an agent."""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        retriever: Optional[LexicalMemoryRetriever] = None,
        formatter: Optional[PromptMemoryFormatter] = None,
    ) -> None:
        self.store = store or InMemoryStore()
        self.retriever = retriever or LexicalMemoryRetriever()
        self.formatter = formatter or PromptMemoryFormatter()

    def add(
        self,
        content: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        role: Optional[str] = None,
        kind: str = "note",
        tags: Optional[Iterable[str]] = None,
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            content=content,
            session_id=session_id,
            agent_id=agent_id,
            role=role,
            kind=kind,
            tags=list(tags or []),
            importance=importance,
            metadata=dict(metadata or {}),
        )
        return self.store.add(record)

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        importance: float = 0.5,
    ) -> List[MemoryRecord]:
        return [
            self.add(
                user_message,
                session_id=session_id,
                agent_id=agent_id,
                role="user",
                kind="turn",
                tags=tags,
                importance=importance,
            ),
            self.add(
                assistant_message,
                session_id=session_id,
                agent_id=agent_id,
                role="assistant",
                kind="turn",
                tags=tags,
                importance=importance,
            ),
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        role: Optional[str] = None,
        kinds: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        min_importance: Optional[float] = None,
        include_zero_score: bool = False,
    ) -> List[MemorySearchResult]:
        filters = MemoryFilter.from_values(
            session_id=session_id,
            agent_id=agent_id,
            role=role,
            kinds=kinds,
            tags=tags,
            min_importance=min_importance,
        )
        records = self.store.list(filters)
        return self.retriever.search(
            query=query,
            records=records,
            top_k=top_k,
            include_zero_score=include_zero_score,
        )

    def list(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        role: Optional[str] = None,
        kinds: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        min_importance: Optional[float] = None,
    ) -> List[MemoryRecord]:
        filters = MemoryFilter.from_values(
            session_id=session_id,
            agent_id=agent_id,
            role=role,
            kinds=kinds,
            tags=tags,
            min_importance=min_importance,
        )
        return self.store.list(filters)

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        return self.store.get(record_id)

    def update(self, record_id: str, **updates: Any) -> MemoryRecord:
        record = self.store.get(record_id)
        if record is None:
            raise KeyError(f"Memory record not found: {record_id}")
        updated = record.with_updates(**updates)
        return self.store.update(updated)

    def delete(self, record_id: str) -> bool:
        return self.store.delete(record_id)

    def format_for_prompt(self, results: Iterable[MemorySearchResult]) -> str:
        return self.formatter.format_results(results)

    def summarize(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        max_items: int = 8,
    ) -> str:
        return summarize_memories(
            self.list(session_id=session_id, agent_id=agent_id),
            max_items=max_items,
        )
