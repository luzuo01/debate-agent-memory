"""Memory retrieval and ranking strategies."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

from debate_agent_memory.models import MemoryRecord, MemorySearchResult

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class LexicalMemoryRetriever:
    """Rank memories using local lexical similarity, recency, and importance."""

    def __init__(
        self,
        lexical_weight: float = 0.72,
        importance_weight: float = 0.2,
        recency_weight: float = 0.08,
        recency_half_life_days: float = 14.0,
    ) -> None:
        self.lexical_weight = lexical_weight
        self.importance_weight = importance_weight
        self.recency_weight = recency_weight
        self.recency_half_life_days = recency_half_life_days

    def search(
        self,
        query: str,
        records: Sequence[MemoryRecord],
        top_k: int = 5,
        include_zero_score: bool = False,
    ) -> List[MemorySearchResult]:
        query_tokens = Counter(tokenize(query))
        results = [
            self.score(query_tokens=query_tokens, record=record)
            for record in records
        ]
        if not include_zero_score:
            results = [result for result in results if result.score > 0.0]
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def score(self, query_tokens: Counter[str], record: MemoryRecord) -> MemorySearchResult:
        haystack = " ".join(
            [
                record.content,
                record.kind,
                " ".join(record.tags),
                str(record.metadata.get("topic", "")),
            ]
        )
        lexical_score = cosine_similarity(query_tokens, Counter(tokenize(haystack)))
        recency_score = self._recency_score(record.created_at)
        importance_score = record.importance
        score = (
            self.lexical_weight * lexical_score
            + self.importance_weight * importance_score
            + self.recency_weight * recency_score
        )

        reasons = []
        if lexical_score > 0.0:
            reasons.append(f"lexical={lexical_score:.3f}")
        if importance_score > 0.0:
            reasons.append(f"importance={importance_score:.3f}")
        if recency_score > 0.0:
            reasons.append(f"recency={recency_score:.3f}")

        return MemorySearchResult(record=record, score=round(score, 6), reasons=reasons)

    def _recency_score(self, created_at: datetime) -> float:
        now = datetime.now(timezone.utc)
        age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)
        if self.recency_half_life_days <= 0:
            return 0.0
        return math.exp(-math.log(2) * age_days / self.recency_half_life_days)


def summarize_memories(records: Iterable[MemoryRecord], max_items: int = 8) -> str:
    """Create a compact extractive summary for memory compaction workflows."""

    selected = sorted(records, key=lambda item: (item.importance, item.created_at), reverse=True)
    lines = []
    for record in selected[:max_items]:
        tag_text = f" [{','.join(record.tags)}]" if record.tags else ""
        lines.append(f"- ({record.kind}){tag_text} {record.content}")
    return "\n".join(lines)
