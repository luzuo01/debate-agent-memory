"""Prompt formatting helpers for retrieved memories."""

from __future__ import annotations

from typing import Iterable

from debate_agent_memory.models import MemoryRecord, MemorySearchResult


class PromptMemoryFormatter:
    """Render memories in a compact, prompt-friendly block."""

    def __init__(
        self,
        title: str = "Relevant memories",
        empty_text: str = "No relevant memories.",
        max_chars_per_memory: int = 500,
    ) -> None:
        self.title = title
        self.empty_text = empty_text
        self.max_chars_per_memory = max_chars_per_memory

    def format_results(self, results: Iterable[MemorySearchResult]) -> str:
        records = [result.record for result in results]
        return self.format_records(records)

    def format_records(self, records: Iterable[MemoryRecord]) -> str:
        records = list(records)
        if not records:
            return self.empty_text

        lines = [f"{self.title}:"]
        for index, record in enumerate(records, start=1):
            tags = f" | tags={','.join(record.tags)}" if record.tags else ""
            role = f" | role={record.role}" if record.role else ""
            content = self._truncate(record.content)
            lines.append(
                f"{index}. [{record.kind}{role} | importance={record.importance:.2f}{tags}] "
                f"{content}"
            )
        return "\n".join(lines)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars_per_memory:
            return text
        return text[: self.max_chars_per_memory - 3].rstrip() + "..."
