"""Command line interface for the memory module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from debate_agent_memory.core import AgentMemory
from debate_agent_memory.stores import JsonlMemoryStore, SQLiteMemoryStore


def build_memory(store_path: str) -> AgentMemory:
    if store_path.endswith((".db", ".sqlite", ".sqlite3")):
        return AgentMemory(SQLiteMemoryStore(store_path))
    return AgentMemory(JsonlMemoryStore(store_path))


def add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", dest="session_id")
    parser.add_argument("--agent", dest="agent_id")
    parser.add_argument("--role")
    parser.add_argument("--kind", action="append", dest="kinds")
    parser.add_argument("--tag", action="append", dest="tags")
    parser.add_argument("--min-importance", type=float)


def print_records(records: Iterable[object]) -> None:
    for item in records:
        if hasattr(item, "record"):
            record = item.record
            score = f" score={item.score:.4f}"
        else:
            record = item
            score = ""
        tags = ",".join(record.tags)
        print(
            f"{record.id}{score} session={record.session_id or '-'} "
            f"kind={record.kind} role={record.role or '-'} importance={record.importance:.2f} "
            f"tags={tags or '-'} :: {record.content}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="debate-memory")
    parser.add_argument("--store", default=str(Path(".memory") / "memories.jsonl"))

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a memory record.")
    add_parser.add_argument("content")
    add_parser.add_argument("--session", dest="session_id")
    add_parser.add_argument("--agent", dest="agent_id")
    add_parser.add_argument("--role")
    add_parser.add_argument("--kind", default="note")
    add_parser.add_argument("--tag", action="append", dest="tags")
    add_parser.add_argument("--importance", type=float, default=0.5)
    add_parser.add_argument("--metadata", default="{}")

    search_parser = subparsers.add_parser("search", help="Search memory records.")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--include-zero-score", action="store_true")
    add_common_filters(search_parser)

    list_parser = subparsers.add_parser("list", help="List memory records.")
    add_common_filters(list_parser)

    export_parser = subparsers.add_parser("export", help="Export records as JSON.")
    add_common_filters(export_parser)

    summarize_parser = subparsers.add_parser("summarize", help="Create a compact summary.")
    summarize_parser.add_argument("--session", dest="session_id")
    summarize_parser.add_argument("--agent", dest="agent_id")
    summarize_parser.add_argument("--max-items", type=int, default=8)

    args = parser.parse_args(argv)
    memory = build_memory(args.store)

    if args.command == "add":
        metadata = json.loads(args.metadata)
        record = memory.add(
            args.content,
            session_id=args.session_id,
            agent_id=args.agent_id,
            role=args.role,
            kind=args.kind,
            tags=args.tags,
            importance=args.importance,
            metadata=metadata,
        )
        print(record.id)
        return 0

    if args.command == "search":
        results = memory.search(
            args.query,
            top_k=args.top_k,
            session_id=args.session_id,
            agent_id=args.agent_id,
            role=args.role,
            kinds=args.kinds,
            tags=args.tags,
            min_importance=args.min_importance,
            include_zero_score=args.include_zero_score,
        )
        print_records(results)
        return 0

    if args.command == "list":
        print_records(
            memory.list(
                session_id=args.session_id,
                agent_id=args.agent_id,
                role=args.role,
                kinds=args.kinds,
                tags=args.tags,
                min_importance=args.min_importance,
            )
        )
        return 0

    if args.command == "export":
        records = memory.list(
            session_id=args.session_id,
            agent_id=args.agent_id,
            role=args.role,
            kinds=args.kinds,
            tags=args.tags,
            min_importance=args.min_importance,
        )
        print(json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2))
        return 0

    if args.command == "summarize":
        print(memory.summarize(args.session_id, args.agent_id, args.max_items))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
