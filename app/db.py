from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_id    TEXT PRIMARY KEY,
    date       TEXT NOT NULL,
    path       TEXT NOT NULL,
    rows       INTEGER NOT NULL,
    status     TEXT NOT NULL CHECK (status IN ('active', 'obsolete', 'deleted')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_date ON files(date, status);

-- every (user, date) pair ever ingested; used to know which dates a user touches
CREATE TABLE IF NOT EXISTS user_dates (
    user_id TEXT NOT NULL,
    date    TEXT NOT NULL,
    PRIMARY KEY (user_id, date)
);

CREATE TABLE IF NOT EXISTS erasure_tasks (
    task_id    TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL UNIQUE,          -- one task per user, forever
    status     TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'failed', 'completed')),
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- files a task still has to inspect/rewrite
CREATE TABLE IF NOT EXISTS erasure_files (
    task_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    status  TEXT NOT NULL CHECK (status IN ('pending', 'done')),
    PRIMARY KEY (task_id, file_id)
);

-- a date is restricted as long as at least one row references it
CREATE TABLE IF NOT EXISTS restricted_dates (
    date    TEXT NOT NULL,
    task_id TEXT NOT NULL,
    PRIMARY KEY (date, task_id)
);

-- append-only feed of date changes consumed by downstream mirrors.
-- AUTOINCREMENT guarantees a sequence number is never reused, so a cursor
-- handed out before a restart keeps its meaning after it.
CREATE TABLE IF NOT EXISTS change_events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT NOT NULL,
    action     TEXT NOT NULL CHECK (action IN ('refetch', 'withdraw', 'delete')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_change_events_seq ON change_events(seq);
"""


def append_event(conn: sqlite3.Connection, date: str, action: str, now: str) -> None:
    """Append one change event inside the caller's transaction.

    Every event is written in the same transaction as the state change it
    describes, so a change can never become visible without its notification
    (nor a notification without the change).
    """
    conn.execute(
        "INSERT INTO change_events(date, action, created_at) VALUES (?,?,?)",
        (date, action, now),
    )


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    """SQLite-backed state.

    A single connection guarded by an RLock; every multi-step write happens
    inside ``transaction()`` (BEGIN IMMEDIATE) so concurrent API requests and
    the background worker never observe a half-applied change.
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=15000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---------------------------------------------------------------- files

    def record_upload(self, file_id: str, date: str, path: str, rows: int,
                      user_ids: Iterable[str]) -> None:
        """Register a new part file and, in the same transaction, notify the
        change feed.

        A ``refetch`` is emitted only when the date is downloadable at commit
        time: while a date is restricted (an erasure covering it is running)
        the upstream upload must not tell mirrors to pull, because the date is
        not served and its content is still being rewritten. The erasure's
        own finalize step emits the notification once every erasure touching
        the date has completed. If this transaction fails, neither the file
        record nor the event exists.
        """
        now = utcnow()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO files(file_id, date, path, rows, status, created_at)"
                " VALUES (?,?,?,?,'active',?)",
                (file_id, date, path, rows, now),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO user_dates(user_id, date) VALUES (?,?)",
                [(u, date) for u in user_ids],
            )
            if not self._is_restricted_conn(conn, date):
                append_event(conn, date, "refetch", now)

    @staticmethod
    def _is_restricted_conn(conn: sqlite3.Connection, date: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM restricted_dates WHERE date=? LIMIT 1", (date,)
        ).fetchone() is not None

    def get_file(self, file_id: str) -> Optional[sqlite3.Row]:
        return self.query_one("SELECT * FROM files WHERE file_id=?", (file_id,))

    def active_files_for_date(self, date: str) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM files WHERE date=? AND status='active' ORDER BY created_at",
            (date,),
        )

    def active_files_in_dates(self, dates: list[str]) -> list[sqlite3.Row]:
        if not dates:
            return []
        marks = ",".join("?" for _ in dates)
        return self.query(
            f"SELECT * FROM files WHERE status='active' AND date IN ({marks})",
            tuple(dates),
        )

    def files_with_status(self, *statuses: str) -> list[sqlite3.Row]:
        marks = ",".join("?" for _ in statuses)
        return self.query(f"SELECT * FROM files WHERE status IN ({marks})", statuses)

    def all_known_paths(self) -> set[str]:
        return {r["path"] for r in self.query("SELECT path FROM files")}

    def mark_file_deleted(self, file_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE files SET status='deleted' WHERE file_id=?", (file_id,))

    # -------------------------------------------------------- erasure tasks

    def accept_task(self, task_id: str, user_id: str) -> None:
        """Create the task, snapshot every active file that may hold the user's
        data and restrict the affected dates — atomically. Runs under the
        service commit lock, so no upload can slip in between.

        Every date that becomes restricted by this acceptance is withdrawn
        from the change feed in the same transaction: from that instant the
        date is neither downloadable nor mirrorable, and a failure of the
        task keeps it withdrawn (no further event is written until the task
        finishes). Dates that were already withdrawn by another erasure do
        not produce a second event.
        """
        now = utcnow()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO erasure_tasks(task_id, user_id, status, attempts, created_at, updated_at)"
                " VALUES (?,?,'pending',0,?,?)",
                (task_id, user_id, now, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO erasure_files(task_id, file_id, status)"
                " SELECT ?, file_id, 'pending' FROM files"
                " WHERE status='active' AND date IN"
                "       (SELECT date FROM user_dates WHERE user_id=?)",
                (task_id, user_id),
            )
            newly_restricted = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT ud.date FROM user_dates ud"
                    " WHERE ud.user_id=?"
                    "   AND NOT EXISTS (SELECT 1 FROM restricted_dates rd"
                    "                    WHERE rd.date=ud.date)"
                    " ORDER BY ud.date",
                    (user_id,),
                )
            ]
            conn.execute(
                "INSERT OR IGNORE INTO restricted_dates(date, task_id)"
                " SELECT date, ? FROM user_dates WHERE user_id=?",
                (task_id, user_id),
            )
            for date in newly_restricted:
                append_event(conn, date, "withdraw", now)

    def get_task(self, task_id: str) -> Optional[sqlite3.Row]:
        return self.query_one("SELECT * FROM erasure_tasks WHERE task_id=?", (task_id,))

    def get_task_by_user(self, user_id: str) -> Optional[sqlite3.Row]:
        return self.query_one("SELECT * FROM erasure_tasks WHERE user_id=?", (user_id,))

    def all_tasks(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM erasure_tasks ORDER BY created_at")

    def incomplete_tasks(self) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM erasure_tasks WHERE status != 'completed' ORDER BY created_at"
        )

    def mark_processing(self, task_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE erasure_tasks SET status='processing', attempts=attempts+1,"
                " updated_at=? WHERE task_id=?",
                (utcnow(), task_id),
            )

    def mark_failed(self, task_id: str, error: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE erasure_tasks SET status='failed', last_error=?, updated_at=?"
                " WHERE task_id=?",
                (error, utcnow(), task_id),
            )

    # -------------------------------------------------------- erasure files

    def next_pending_file(self, task_id: str) -> Optional[sqlite3.Row]:
        return self.query_one(
            "SELECT file_id FROM erasure_files WHERE task_id=? AND status='pending'"
            " ORDER BY rowid LIMIT 1",
            (task_id,),
        )

    def mark_file_done(self, task_id: str, file_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE erasure_files SET status='done' WHERE task_id=? AND file_id=?",
                (task_id, file_id),
            )

    def add_pending_files(self, task_id: str, file_ids: Iterable[str]) -> None:
        with self.transaction() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO erasure_files(task_id, file_id, status)"
                " VALUES (?,?,'pending')",
                [(task_id, f) for f in file_ids],
            )

    def file_counts(self, task_id: str) -> tuple[int, int]:
        row = self.query_one(
            "SELECT COUNT(*) AS total, COALESCE(SUM(status='done'),0) AS done"
            " FROM erasure_files WHERE task_id=?",
            (task_id,),
        )
        return row["total"], row["done"]

    def record_rewrite(self, task_id: str, old_file_id: str, new_file_id: str,
                       date: str, path: str, rows: int) -> None:
        """Atomically register the replacement file, retire the old one and
        re-point every *other* task's pending rows at the new file, so
        overlapping erasures never lose track of data they still must remove."""
        now = utcnow()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO files(file_id, date, path, rows, status, created_at)"
                " VALUES (?,?,?,?,'active',?)",
                (new_file_id, date, path, rows, now),
            )
            conn.execute("UPDATE files SET status='obsolete' WHERE file_id=?", (old_file_id,))
            conn.execute(
                "UPDATE erasure_files SET status='done' WHERE task_id=? AND file_id=?",
                (task_id, old_file_id),
            )
            conn.execute(
                "UPDATE OR IGNORE erasure_files SET file_id=? WHERE file_id=? AND status='pending'",
                (new_file_id, old_file_id),
            )
            # rows that could not be re-pointed because the task already tracks the new file
            conn.execute(
                "DELETE FROM erasure_files WHERE file_id=? AND status='pending'",
                (old_file_id,),
            )

    def record_delete_only(self, task_id: str, old_file_id: str) -> None:
        """The filtered file would be empty: retire it and close every pending
        row pointing at it (its only remaining rows belonged to erased users)."""
        with self.transaction() as conn:
            conn.execute("UPDATE files SET status='obsolete' WHERE file_id=?", (old_file_id,))
            conn.execute(
                "UPDATE erasure_files SET status='done' WHERE file_id=?", (old_file_id,)
            )

    # ----------------------------------------------- user dates / restrictions

    def dates_for_user(self, user_id: str) -> list[str]:
        return [
            r["date"]
            for r in self.query("SELECT date FROM user_dates WHERE user_id=?", (user_id,))
        ]

    def is_restricted(self, date: str) -> bool:
        return (
            self.query_one(
                "SELECT 1 AS x FROM restricted_dates WHERE date=? LIMIT 1", (date,)
            )
            is not None
        )

    def restricted_dates(self) -> set[str]:
        return {r["date"] for r in self.query("SELECT DISTINCT date FROM restricted_dates")}

    def restricted_dates_for_task(self, task_id: str) -> list[str]:
        return [
            r["date"]
            for r in self.query(
                "SELECT date FROM restricted_dates WHERE task_id=? ORDER BY date", (task_id,)
            )
        ]

    def finalize_task(self, task_id: str, user_id: str) -> None:
        """Mark the task completed and lift this task's date restrictions.
        The user stays blocked forever because the task row itself remains.

        A date is only announced again once *every* erasure covering it is
        complete: when this task's restriction is removed, dates that are
        still restricted by another task stay withdrawn. For each date that
        becomes downloadable again the feed emits ``refetch`` when the date
        still has active files, or ``delete`` when nothing is left (the
        mirror must drop its stale copy). All of this happens in one
        transaction with the task completion, so the notification always
        matches what ``/dates`` and the download endpoint will serve.
        """
        now = utcnow()
        with self.transaction() as conn:
            restricted = [
                r[0]
                for r in conn.execute(
                    "SELECT date FROM restricted_dates WHERE task_id=? ORDER BY date",
                    (task_id,),
                )
            ]
            conn.execute("DELETE FROM user_dates WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM restricted_dates WHERE task_id=?", (task_id,))
            for date in restricted:
                if self._is_restricted_conn(conn, date):
                    continue  # another erasure still covers this date
                has_files = conn.execute(
                    "SELECT 1 FROM files WHERE date=? AND status='active' LIMIT 1",
                    (date,),
                ).fetchone() is not None
                append_event(conn, date, "refetch" if has_files else "delete", now)
            conn.execute(
                "UPDATE erasure_tasks SET status='completed', last_error=NULL, updated_at=?"
                " WHERE task_id=?",
                (now, task_id),
            )

    # -------------------------------------------------------- change feed

    def subscribe_snapshot(self) -> tuple[list[dict], int]:
        """Initial subscribable state: the downloadable dates and the cursor
        separating them from later events.

        Read in one transaction (and by the caller under the commit lock), so
        an upload or erasure commit cannot land between the date list and the
        cursor. Any change after this snapshot has a ``seq`` greater than the
        returned cursor and is therefore delivered by the feed.
        """
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT f.date AS date, COUNT(*) AS files, SUM(f.rows) AS rows"
                " FROM files f"
                " WHERE f.status='active'"
                "   AND NOT EXISTS (SELECT 1 FROM restricted_dates rd"
                "                   WHERE rd.date=f.date)"
                " GROUP BY f.date ORDER BY f.date"
            ).fetchall()
            cursor = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS c FROM change_events"
            ).fetchone()[0]
            return [
                {"date": r["date"], "files": r["files"], "rows": r["rows"]}
                for r in rows
            ], cursor

    def changes_page(self, cursor: int, limit: int) -> tuple[list[dict], int]:
        """Read one page of events after ``cursor`` plus the log head, in a
        single transaction so the page and ``has_more`` agree."""
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT seq, date, action, created_at FROM change_events"
                " WHERE seq > ? ORDER BY seq LIMIT ?",
                (cursor, limit),
            ).fetchall()
            latest = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS c FROM change_events"
            ).fetchone()[0]
        return [dict(r) for r in rows], latest

