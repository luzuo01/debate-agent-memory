"""Storage backends for memory records."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from debate_agent_memory.models import MemoryFilter, MemoryRecord


class MemoryStore(ABC):
    """Persistence boundary used by the high-level memory facade."""

    @abstractmethod
    def add(self, record: MemoryRecord) -> MemoryRecord:
        raise NotImplementedError

    @abstractmethod
    def get(self, record_id: str) -> Optional[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def list(self, filters: Optional[MemoryFilter] = None) -> List[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def update(self, record: MemoryRecord) -> MemoryRecord:
        raise NotImplementedError

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """Volatile store useful for tests and short-lived experiments."""

    def __init__(self, records: Optional[Iterable[MemoryRecord]] = None) -> None:
        self._records: Dict[str, MemoryRecord] = {}
        for record in records or []:
            self._records[record.id] = record

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.id] = record
        return record

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        return self._records.get(record_id)

    def list(self, filters: Optional[MemoryFilter] = None) -> List[MemoryRecord]:
        records = sorted(self._records.values(), key=lambda item: item.created_at)
        if filters is None:
            return records
        return [record for record in records if filters.matches(record)]

    def update(self, record: MemoryRecord) -> MemoryRecord:
        if record.id not in self._records:
            raise KeyError(f"Memory record not found: {record.id}")
        self._records[record.id] = record
        return record

    def delete(self, record_id: str) -> bool:
        return self._records.pop(record_id, None) is not None


class JsonlMemoryStore(MemoryStore):
    """Append-friendly JSONL store with atomic full-file rewrites on update/delete."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def add(self, record: MemoryRecord) -> MemoryRecord:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        for record in self.list():
            if record.id == record_id:
                return record
        return None

    def list(self, filters: Optional[MemoryFilter] = None) -> List[MemoryRecord]:
        records: List[MemoryRecord] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = MemoryRecord.from_dict(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid memory record in {self.path} at line {line_number}"
                    ) from exc
                if filters is None or filters.matches(record):
                    records.append(record)
        return sorted(records, key=lambda item: item.created_at)

    def update(self, record: MemoryRecord) -> MemoryRecord:
        records = self.list()
        found = False
        updated_records = []
        for existing in records:
            if existing.id == record.id:
                updated_records.append(record)
                found = True
            else:
                updated_records.append(existing)
        if not found:
            raise KeyError(f"Memory record not found: {record.id}")
        self._rewrite(updated_records)
        return record

    def delete(self, record_id: str) -> bool:
        records = self.list()
        updated_records = [record for record in records if record.id != record_id]
        if len(updated_records) == len(records):
            return False
        self._rewrite(updated_records)
        return True

    def _rewrite(self, records: Iterable[MemoryRecord]) -> None:
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        temp_path.replace(self.path)


class SQLiteMemoryStore(MemoryStore):
    """SQLite-backed store for larger local memory files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def add(self, record: MemoryRecord) -> MemoryRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.id,
                    json.dumps(record.to_dict(), ensure_ascii=False),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
        return record

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM memories WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return MemoryRecord.from_dict(json.loads(row["payload"]))

    def list(self, filters: Optional[MemoryFilter] = None) -> List[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM memories ORDER BY created_at ASC"
            ).fetchall()
        records = [MemoryRecord.from_dict(json.loads(row["payload"])) for row in rows]
        if filters is None:
            return records
        return [record for record in records if filters.matches(record)]

    def update(self, record: MemoryRecord) -> MemoryRecord:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE memories
                SET payload = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(record.to_dict(), ensure_ascii=False),
                    record.updated_at.isoformat(),
                    record.id,
                ),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"Memory record not found: {record.id}")
        return record

    def delete(self, record_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (record_id,))
        return cursor.rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
