import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from .config import Config


_QUERY_TOKEN = re.compile(r"\w+", re.UNICODE)


def _database_path(db_path: Optional[Path] = None) -> Path:
    return Path(db_path if db_path is not None else Config.BM25_DB_PATH)


def _chunk_record(chunk: Dict) -> tuple[str, str, str]:
    chunk_id = chunk.get("id")
    content = chunk.get("content")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("Every chunk must have a non-empty stable string id.")
    if not isinstance(content, str):
        raise ValueError(f"Chunk {chunk_id!r} must have string content.")
    metadata = {"title": chunk.get("title"), **chunk.get("metadata", {})}
    metadata_json = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return chunk_id, content, metadata_json


def build_bm25_index(
    chunks: Sequence[Dict], db_path: Optional[Path] = None
) -> Set[str]:
    """Synchronize a persistent SQLite FTS5 index with the provided chunks."""
    records = [_chunk_record(chunk) for chunk in chunks]
    record_ids = [record[0] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Processed chunks contain duplicate stable IDs.")

    path = _database_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS bm25_documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS bm25_fts
            USING fts5(id UNINDEXED, content)
            """
        )

        current = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                "SELECT id, content, metadata_json FROM bm25_documents"
            )
        }
        desired = {record[0]: (record[1], record[2]) for record in records}

        stale_ids = current.keys() - desired.keys()
        for stale_id in stale_ids:
            connection.execute("DELETE FROM bm25_fts WHERE id = ?", (stale_id,))
            connection.execute("DELETE FROM bm25_documents WHERE id = ?", (stale_id,))

        for chunk_id, content, metadata_json in records:
            if current.get(chunk_id) == (content, metadata_json):
                continue
            connection.execute("DELETE FROM bm25_fts WHERE id = ?", (chunk_id,))
            connection.execute(
                """
                INSERT INTO bm25_documents (id, content, metadata_json)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    metadata_json = excluded.metadata_json
                """,
                (chunk_id, content, metadata_json),
            )
            connection.execute(
                "INSERT INTO bm25_fts (id, content) VALUES (?, ?)",
                (chunk_id, content),
            )

        synchronized_ids = {
            row[0]
            for row in connection.execute("SELECT id FROM bm25_documents")
        }
        if synchronized_ids != set(record_ids):
            raise RuntimeError(
                "BM25 synchronization produced an unexpected ID set "
                f"(expected {len(record_ids)}, found {len(synchronized_ids)})."
            )

    print(f"🎉 BM25 index synchronized with {len(synchronized_ids)} chunks at {path}.")
    return synchronized_ids


def search_bm25(
    query: str, limit: int = 10, db_path: Optional[Path] = None
) -> List[Dict]:
    """Return matching chunks ordered by their SQLite FTS5 BM25 rank."""
    path = _database_path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"BM25 index database does not exist: {path}")
    tokens = _QUERY_TOKEN.findall(query)
    if not tokens or limit <= 0:
        return []
    match_query = " OR ".join(f'"{token}"' for token in tokens)

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT d.id, d.content, d.metadata_json, bm25(bm25_fts) AS rank
            FROM bm25_fts
            JOIN bm25_documents AS d ON d.id = bm25_fts.id
            WHERE bm25_fts MATCH ?
            ORDER BY rank ASC, d.id ASC
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
    return [
        {
            "id": row[0],
            "content": row[1],
            "metadata": json.loads(row[2]),
            "score": row[3],
        }
        for row in rows
    ]


def get_bm25_ids(db_path: Optional[Path] = None) -> Set[str]:
    """Return the IDs currently stored in the persistent BM25 document table."""
    path = _database_path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"BM25 index database does not exist: {path}")
    with sqlite3.connect(path) as connection:
        return {row[0] for row in connection.execute("SELECT id FROM bm25_documents")}
