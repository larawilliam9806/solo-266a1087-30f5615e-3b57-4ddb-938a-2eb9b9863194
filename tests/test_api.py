from __future__ import annotations

import csv
import io
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from app import storage
from app.config import Config
from app.main import create_app


# ------------------------------------------------------------------ helpers


def make_config(tmp_path: Path, **overrides) -> Config:
    kwargs = dict(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        retry_delay=0.2,
        worker_poll=0.05,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def make_client(tmp_path: Path, **overrides) -> TestClient:
    return TestClient(create_app(make_config(tmp_path, **overrides)))


def csv_bytes(rows) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["user_id", "page", "timestamp"])
    w.writerows(rows)
    return buf.getvalue().encode()


def upload(client: TestClient, date: str, rows):
    return client.post(
        f"/upload/{date}", files={"file": ("clicks.csv", csv_bytes(rows), "text/csv")}
    )


def wait_task(client: TestClient, task_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        r = client.get(f"/erasure/{task_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] == "completed":
            return body
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not complete in time: {body}")


def wait_status(client: TestClient, task_id: str, want: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        body = client.get(f"/erasure/{task_id}").json()
        if body["status"] == want:
            return body
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} never reached {want}: {body}")


def download_users(client: TestClient, date: str) -> set[str]:
    r = client.get(f"/dates/{date}/download")
    assert r.status_code == 200, r.text
    rows = list(csv.reader(io.StringIO(r.text)))
    return {row[0] for row in rows[1:]}


# ------------------------------------------------------------------- tests


def test_health(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_upload_and_download_roundtrip(tmp_path):
    with make_client(tmp_path) as client:
        r = upload(
            client,
            "2026-11-01",
            [
                ("alice", "/home", "2026-11-01T10:00:00Z"),
                ("bob", "/pricing", "2026-11-01 10:05:00+00:00"),
            ],
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["received"] == 2
        assert body["blocked"] == 0
        assert body["stored"] == 2

        r = client.get("/dates/2026-11-01/download")
        assert r.status_code == 200
        assert "alice" in r.text and "bob" in r.text

        # appending a second CSV to the same day keeps both
        upload(client, "2026-11-01", [("carol", "/about", "2026-11-01T11:00:00Z")])
        assert download_users(client, "2026-11-01") == {"alice", "bob", "carol"}

        # parquet download
        r = client.get("/dates/2026-11-01/download?format=parquet")
        assert r.status_code == 200
        table = pq.read_table(io.BytesIO(r.content))
        assert table.num_rows == 3
        assert str(table.schema.field("ts").type) == "timestamp[us, tz=UTC]"

        # listing
        dates = client.get("/dates").json()
        assert dates == [
            {"date": "2026-11-01", "files": 2, "rows": 3, "restricted": False}
        ]


def test_upload_validation(tmp_path):
    with make_client(tmp_path) as client:
        r = client.post(
            "/upload/2026-11-01",
            files={"file": ("x.csv", b"foo,bar\n1,2\n", "text/csv")},
        )
        assert r.status_code == 400

        r = client.post(
            "/upload/2026-13-40",
            files={"file": ("x.csv", csv_bytes([("a", "/p", "2026-11-01T00:00:00Z")]), "text/csv")},
        )
        assert r.status_code == 422

        # malformed rows are skipped and reported, valid rows are stored
        r = client.post(
            "/upload/2026-11-01",
            files={
                "file": (
                    "x.csv",
                    b"user_id,page,timestamp\nalice,/a,not-a-time\nbob,/b,2026-11-01T00:00:00Z\n",
                    "text/csv",
                )
            },
        )
        body = r.json()
        assert body["received"] == 2
        assert body["invalid"] == 1
        assert body["stored"] == 1


def test_erasure_blocks_new_uploads_immediately(tmp_path):
    with make_client(tmp_path) as client:
        upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])
        r = client.post("/erasure", json={"user_id": "alice"})
        assert r.status_code == 201
        task_id = r.json()["task_id"]

        # effective immediately, before the background worker finishes
        r = upload(
            client,
            "2026-11-02",
            [
                ("alice", "/b", "2026-11-02T00:00:00Z"),
                ("bob", "/c", "2026-11-02T00:01:00Z"),
            ],
        )
        body = r.json()
        assert body["received"] == 2
        assert body["blocked"] == 1
        assert body["stored"] == 1

        wait_task(client, task_id)
        assert download_users(client, "2026-11-02") == {"bob"}


def test_in_flight_upload_cannot_write_back_blocked_user(tmp_path):
    """An erasure accepted while an upload is being committed still wins."""
    with make_client(tmp_path) as client:
        service = client.app.state.service

        def accept_erasure_mid_commit():
            service.submit_erasure("alice")

        service.hooks.before_write = accept_erasure_mid_commit
        try:
            r = upload(
                client,
                "2026-11-01",
                [
                    ("alice", "/a", "2026-11-01T00:00:00Z"),
                    ("bob", "/b", "2026-11-01T00:01:00Z"),
                ],
            )
        finally:
            service.hooks.before_write = None

        body = r.json()
        assert body["blocked"] == 1
        assert body["stored"] == 1

        task = client.get("/erasure", params={"user_id": "alice"}).json()[0]
        wait_task(client, task["task_id"])
        assert download_users(client, "2026-11-01") == {"bob"}


def test_restricted_dates_pause_downloads_until_completion(tmp_path):
    with make_client(tmp_path) as client:
        upload(
            client,
            "2026-11-01",
            [
                ("alice", "/a", "2026-11-01T00:00:00Z"),
                ("bob", "/b", "2026-11-01T01:00:00Z"),
            ],
        )
        upload(client, "2026-11-02", [("carol", "/c", "2026-11-02T00:00:00Z")])

        service = client.app.state.service
        # hold the worker so we can observe the restricted state deterministically
        gate = threading.Event()
        original_erase = service.erase_file

        def slow_erase(task, file_id):
            gate.wait(10)
            return original_erase(task, file_id)

        service.erase_file = slow_erase
        try:
            r = client.post("/erasure", json={"user_id": "alice"})
            task_id = r.json()["task_id"]
            assert r.json()["restricted_dates"] == ["2026-11-01"]

            # the affected date refuses new downloads (old links included)
            r = client.get("/dates/2026-11-01/download")
            assert r.status_code == 409
            # other dates are unaffected
            assert client.get("/dates/2026-11-02/download").status_code == 200
            # imports for other dates keep working
            assert upload(
                client, "2026-11-02", [("dave", "/d", "2026-11-02T01:00:00Z")]
            ).json()["stored"] == 1
        finally:
            gate.set()
            service.erase_file = original_erase

        wait_task(client, task_id)
        # restriction lifted once the rewrite finished
        r = client.get("/dates/2026-11-01/download")
        assert r.status_code == 200
        assert "alice" not in r.text
        assert "bob" in r.text


def test_erasure_rewrites_files_and_deletes_old_ones(tmp_path):
    with make_client(tmp_path) as client:
        upload(
            client,
            "2026-11-01",
            [("alice", "/a", "2026-11-01T00:00:00Z"), ("bob", "/b", "2026-11-01T00:01:00Z")],
        )
        upload(
            client,
            "2026-11-01",
            [("alice", "/x", "2026-11-01T02:00:00Z"), ("carol", "/y", "2026-11-01T02:01:00Z")],
        )
        data_dir = tmp_path / "data"
        before = {p.name for p in data_dir.rglob("part-*.parquet")}
        assert len(before) == 2

        r = client.post("/erasure", json={"user_id": "alice"})
        status = wait_task(client, r.json()["task_id"])
        assert status["files_total"] == 2
        assert status["files_done"] == 2

        # every old part file is physically gone
        after = {p.name for p in data_dir.rglob("part-*.parquet")}
        assert before.isdisjoint(after)
        assert not list(data_dir.rglob("*.tmp"))

        # other users' records survive
        assert download_users(client, "2026-11-01") == {"bob", "carol"}


def test_duplicate_submission_returns_same_task_and_blocks_forever(tmp_path):
    with make_client(tmp_path) as client:
        upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])
        r1 = client.post("/erasure", json={"user_id": "alice"})
        r2 = client.post("/erasure", json={"user_id": "alice"})
        assert r1.status_code == 201
        assert r2.status_code == 200
        assert r1.json()["task_id"] == r2.json()["task_id"]

        wait_task(client, r1.json()["task_id"])

        # a completed erasure blocks the user permanently
        r = upload(client, "2026-11-03", [("alice", "/x", "2026-11-03T00:00:00Z")])
        assert r.json()["blocked"] == 1
        assert r.json()["stored"] == 0
        assert client.get("/dates/2026-11-03/download").status_code == 404


def test_erasure_for_unknown_user_completes_and_blocks(tmp_path):
    with make_client(tmp_path) as client:
        r = client.post("/erasure", json={"user_id": "ghost"})
        assert r.status_code == 201
        status = wait_task(client, r.json()["task_id"])
        assert status["files_total"] == 0

        r = upload(
            client,
            "2026-11-01",
            [("ghost", "/a", "2026-11-01T00:00:00Z"), ("real", "/b", "2026-11-01T00:00:01Z")],
        )
        assert r.json()["blocked"] == 1
        assert r.json()["stored"] == 1


def test_interleaved_erasures_and_imports(tmp_path):
    with make_client(tmp_path) as client:
        upload(
            client,
            "2026-11-01",
            [
                ("alice", "/1", "2026-11-01T00:00:00Z"),
                ("bob", "/2", "2026-11-01T00:01:00Z"),
                ("carol", "/3", "2026-11-01T00:02:00Z"),
            ],
        )
        upload(
            client,
            "2026-11-02",
            [("alice", "/4", "2026-11-02T00:00:00Z"), ("carol", "/5", "2026-11-02T00:01:00Z")],
        )
        upload(client, "2026-11-03", [("bob", "/6", "2026-11-03T00:00:00Z")])

        t1 = client.post("/erasure", json={"user_id": "alice"}).json()["task_id"]
        t2 = client.post("/erasure", json={"user_id": "bob"}).json()["task_id"]

        # imports interleave with the erasures; erased users are dropped,
        # everyone else keeps flowing in
        r = upload(
            client,
            "2026-11-01",
            [("alice", "/7", "2026-11-01T03:00:00Z"), ("dave", "/8", "2026-11-01T03:01:00Z")],
        )
        assert (r.json()["blocked"], r.json()["stored"]) == (1, 1)
        r = upload(
            client,
            "2026-11-02",
            [("bob", "/9", "2026-11-02T03:00:00Z"), ("erin", "/a", "2026-11-02T03:01:00Z")],
        )
        assert (r.json()["blocked"], r.json()["stored"]) == (1, 1)
        upload(client, "2026-11-04", [("frank", "/b", "2026-11-04T00:00:00Z")])

        wait_task(client, t1)
        wait_task(client, t2)

        # nothing of alice or bob remains anywhere, nobody else was lost
        assert download_users(client, "2026-11-01") == {"carol", "dave"}
        assert download_users(client, "2026-11-02") == {"carol", "erin"}
        assert client.get("/dates/2026-11-03/download").status_code == 404  # only bob
        assert download_users(client, "2026-11-04") == {"frank"}

        # disk and state agree: exactly the active files exist, nothing more
        service = client.app.state.service
        on_disk = {
            str(p.relative_to(tmp_path / "data"))
            for p in (tmp_path / "data").rglob("part-*.parquet")
        }
        active = {
            row["path"]
            for row in service.store.query("SELECT path FROM files WHERE status='active'")
        }
        assert on_disk == active


def test_failed_task_retries_and_eventually_completes(tmp_path, monkeypatch):
    with make_client(tmp_path) as client:
        upload(
            client,
            "2026-11-01",
            [("alice", "/a", "2026-11-01T00:00:00Z"), ("bob", "/b", "2026-11-01T00:01:00Z")],
        )

        real_write = storage.write_table_atomic
        calls = {"n": 0}

        def fail_once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("simulated disk failure")
            return real_write(*args, **kwargs)

        monkeypatch.setattr(storage, "write_table_atomic", fail_once)

        task_id = client.post("/erasure", json={"user_id": "alice"}).json()["task_id"]
        status = wait_task(client, task_id)
        assert status["attempts"] >= 2  # first attempt failed, retry succeeded
        assert download_users(client, "2026-11-01") == {"bob"}


def test_write_failure_keeps_restrictions_and_restart_resumes(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    app1 = create_app(config)
    with TestClient(app1) as client:
        upload(
            client,
            "2026-11-01",
            [("alice", "/a", "2026-11-01T00:00:00Z"), ("bob", "/b", "2026-11-01T00:01:00Z")],
        )

        def always_fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(storage, "write_table_atomic", always_fail)
        task_id = client.post("/erasure", json={"user_id": "alice"}).json()["task_id"]

        status = wait_status(client, task_id, "failed")
        assert "disk full" in status["last_error"]
        # restrictions stay in place while the task cannot proceed
        assert client.get("/dates/2026-11-01/download").status_code == 409
        r = upload(client, "2026-11-02", [("alice", "/x", "2026-11-02T00:00:00Z")])
        assert r.json()["blocked"] == 1
    # shutdown with the task still failed

    # restart with a healthy disk: recovery picks the task up again
    monkeypatch.undo()
    app2 = create_app(config)
    with TestClient(app2) as client2:
        status = wait_task(client2, task_id)
        assert status["status"] == "completed"
        assert download_users(client2, "2026-11-01") == {"bob"}
        # still blocked forever
        r = upload(client2, "2026-11-05", [("alice", "/y", "2026-11-05T00:00:00Z")])
        assert r.json()["blocked"] == 1


def test_restart_with_pending_task_continues_erasure(tmp_path):
    config = make_config(tmp_path)
    app1 = create_app(config)
    with TestClient(app1) as client:
        upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])
        service = client.app.state.service
        # stop the worker before it can process anything
        service.worker.stop(timeout=2)
        task_id = client.post("/erasure", json={"user_id": "alice"}).json()["task_id"]

    app2 = create_app(config)
    with TestClient(app2) as client2:
        wait_task(client2, task_id)
        assert client2.get("/dates/2026-11-01/download").status_code == 404


def test_progress_endpoint_reports_counts(tmp_path):
    with make_client(tmp_path) as client:
        upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])
        upload(client, "2026-11-02", [("alice", "/b", "2026-11-02T00:00:00Z")])
        r = client.post("/erasure", json={"user_id": "alice"})
        task_id = r.json()["task_id"]
        status = wait_task(client, task_id)
        assert status["files_total"] == status["files_done"]
        assert status["restricted_dates"] == []
        assert status["last_error"] is None
        assert client.get("/erasure/does-not-exist").status_code == 404


# ------------------------------------------------------------ change feed


def get_changes(client: TestClient, cursor: int | None = None, limit: int | None = None) -> dict:
    params = {}
    if cursor is not None:
        params["since"] = cursor
    if limit is not None:
        params["limit"] = limit
    r = client.get("/changes", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def page(client: TestClient, cursor: int, limit: int = 100) -> dict:
    """Claim every event after ``cursor`` by paging like a real mirror."""
    events = []
    while True:
        body = get_changes(client, cursor, limit)
        events.extend(body["events"])
        cursor = body["cursor"]
        if not body["has_more"]:
            return {"events": events, "cursor": cursor, "dates": None}


def test_subscribe_lists_downloadable_dates_and_cursor(tmp_path):
    with make_client(tmp_path) as client:
        upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])
        upload(client, "2026-11-02", [("bob", "/b", "2026-11-02T00:00:00Z")])

        body = get_changes(client)
        assert body["dates"] == [
            {"date": "2026-11-01", "files": 1, "rows": 1},
            {"date": "2026-11-02", "files": 1, "rows": 1},
        ]
        assert isinstance(body["cursor"], int)

        # nothing happened since the snapshot
        assert get_changes(client, body["cursor"])["events"] == []


def test_upload_emits_refetch_and_cursor_pages_are_fixed(tmp_path):
    with make_client(tmp_path) as client:
        cursor = get_changes(client)["cursor"]
        upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])
        upload(client, "2026-11-01", [("bob", "/b", "2026-11-01T00:01:00Z")])
        upload(client, "2026-11-02", [("carol", "/c", "2026-11-02T00:00:00Z")])

        first = get_changes(client, cursor, limit=1)
        assert len(first["events"]) == 1
        assert first["events"][0]["action"] == "refetch"
        assert first["events"][0]["date"] == "2026-11-01"
        assert first["has_more"] is True

        rest = page(client, first["cursor"], limit=1)
        assert [(e["date"], e["action"]) for e in rest["events"]] == [
            ("2026-11-01", "refetch"),
            ("2026-11-02", "refetch"),
        ]

        # sequence numbers are fixed and strictly increasing; re-reading the
        # same cursor yields the same events
        seqs = [e["seq"] for e in [first["events"][0], *rest["events"]]]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        again = page(client, first["cursor"], limit=1)
        assert again["events"] == rest["events"]

        # the mirror's cursor resumes after a restart of the service
        assert page(client, rest["cursor"])["events"] == []


def test_erasure_immediately_withdraws_then_refetches_after_completion(tmp_path):
    with make_client(tmp_path) as client:
        upload(
            client,
            "2026-11-01",
            [("alice", "/a", "2026-11-01T00:00:00Z"), ("bob", "/b", "2026-11-01T00:01:00Z")],
        )
        upload(client, "2026-11-02", [("carol", "/c", "2026-11-02T00:00:00Z")])
        cursor = page(client, get_changes(client)["cursor"])["cursor"]

        service = client.app.state.service
        gate = threading.Event()
        original_erase = service.erase_file

        def slow_erase(task, file_id):
            gate.wait(10)
            return original_erase(task, file_id)

        service.erase_file = slow_erase
        try:
            r = client.post("/erasure", json={"user_id": "alice"})
            task_id = r.json()["task_id"]

            # the withdrawal is visible at acceptance time, before the worker runs
            now = page(client, cursor)
            assert [(e["date"], e["action"]) for e in now["events"]] == [
                ("2026-11-01", "withdraw")
            ]
            cursor = now["cursor"]
            # the date is no longer offered for subscription while restricted
            assert get_changes(client)["dates"] == [
                {"date": "2026-11-02", "files": 1, "rows": 1}
            ]
        finally:
            gate.set()
            service.erase_file = original_erase

        wait_task(client, task_id)
        after = page(client, cursor)
        assert [(e["date"], e["action"]) for e in after["events"]] == [
            ("2026-11-01", "refetch")
        ]
        assert download_users(client, "2026-11-01") == {"bob"}


def test_date_with_only_erased_rows_emits_delete(tmp_path):
    with make_client(tmp_path) as client:
        upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])
        cursor = page(client, get_changes(client)["cursor"])["cursor"]

        task_id = client.post("/erasure", json={"user_id": "alice"}).json()["task_id"]
        wait_task(client, task_id)

        events = page(client, cursor)["events"]
        assert [(e["date"], e["action"]) for e in events] == [
            ("2026-11-01", "withdraw"),
            ("2026-11-01", "delete"),
        ]
        # nothing left to download, so the mirror correctly drops its copy
        assert client.get("/dates/2026-11-01/download").status_code == 404
        assert get_changes(client)["dates"] == []


def test_uploads_during_erasure_do_not_emit_refetch(tmp_path):
    with make_client(tmp_path) as client:
        upload(
            client,
            "2026-11-01",
            [("alice", "/a", "2026-11-01T00:00:00Z"), ("bob", "/b", "2026-11-01T00:01:00Z")],
        )
        cursor = page(client, get_changes(client)["cursor"])["cursor"]

        service = client.app.state.service
        gate = threading.Event()
        original_erase = service.erase_file

        def slow_erase(task, file_id):
            gate.wait(10)
            return original_erase(task, file_id)

        service.erase_file = slow_erase
        try:
            task_id = client.post("/erasure", json={"user_id": "alice"}).json()["task_id"]
            # an upload that commits while the date is restricted must not tell
            # the mirror to pull (the date is not downloadable and is changing)
            r = upload(client, "2026-11-01", [("dave", "/d", "2026-11-01T02:00:00Z")])
            assert r.json()["stored"] == 1
        finally:
            gate.set()
            service.erase_file = original_erase

        wait_task(client, task_id)
        events = page(client, cursor)["events"]
        assert [(e["date"], e["action"]) for e in events] == [
            ("2026-11-01", "withdraw"),
            ("2026-11-01", "refetch"),
        ]
        # the refetch reflects the final downloadable content
        assert download_users(client, "2026-11-01") == {"bob", "dave"}


def test_failed_erasure_stays_withdrawn_until_it_completes(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    app1 = create_app(config)
    with TestClient(app1) as client:
        upload(
            client,
            "2026-11-01",
            [("alice", "/a", "2026-11-01T00:00:00Z"), ("bob", "/b", "2026-11-01T00:01:00Z")],
        )
        cursor = page(client, get_changes(client)["cursor"])["cursor"]

        def always_fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(storage, "write_table_atomic", always_fail)
        task_id = client.post("/erasure", json={"user_id": "alice"}).json()["task_id"]
        wait_status(client, task_id, "failed")

        # still no download and no re-fetch notification
        assert client.get("/dates/2026-11-01/download").status_code == 409
        assert [e["action"] for e in page(client, cursor)["events"]] == ["withdraw"]

    monkeypatch.undo()
    app2 = create_app(config)
    with TestClient(app2) as client2:
        wait_task(client2, task_id)
        events = page(client2, cursor)["events"]
        assert [(e["date"], e["action"]) for e in events] == [
            ("2026-11-01", "withdraw"),
            ("2026-11-01", "refetch"),
        ]


def test_overlapping_erasures_emit_one_refetch_after_all_complete(tmp_path):
    with make_client(tmp_path) as client:
        upload(
            client,
            "2026-11-01",
            [
                ("alice", "/a", "2026-11-01T00:00:00Z"),
                ("bob", "/b", "2026-11-01T00:01:00Z"),
                ("carol", "/c", "2026-11-01T00:02:00Z"),
            ],
        )
        cursor = page(client, get_changes(client)["cursor"])["cursor"]

        service = client.app.state.service
        gate = threading.Event()
        original_erase = service.erase_file

        def slow_erase(task, file_id):
            gate.wait(10)
            return original_erase(task, file_id)

        service.erase_file = slow_erase
        try:
            t1 = client.post("/erasure", json={"user_id": "alice"}).json()["task_id"]
            t2 = client.post("/erasure", json={"user_id": "bob"}).json()["task_id"]
            # one withdraw for the date, not two
            assert [e["action"] for e in page(client, cursor)["events"]] == ["withdraw"]
        finally:
            gate.set()
            service.erase_file = original_erase

        wait_task(client, t1)
        wait_task(client, t2)
        events = page(client, cursor)["events"]
        actions = [e["action"] for e in events]
        # exactly one refetch, and only after both erasures completed
        assert actions.count("withdraw") == 1
        assert actions.count("refetch") == 1
        assert actions[-1] == "refetch"
        assert download_users(client, "2026-11-01") == {"carol"}


def test_first_fetch_interleaved_with_commits_has_no_gap(tmp_path):
    """A subscription racing uploads/erasures never loses a date.

    Every commit that the subscriber did not see in its initial snapshot must
    appear as an event after the returned cursor (and vice versa), otherwise
    the mirror would miss a date entirely.
    """
    with make_client(tmp_path) as client:
        upload(client, "2026-10-31", [("seed", "/s", "2026-10-31T00:00:00Z")])

        bodies: list[dict] = []
        stop = threading.Event()

        def subscriber():
            cursor = None
            while not stop.is_set():
                if cursor is None:
                    body = get_changes(client)
                    seen = {d["date"] for d in body["dates"]}
                    bodies.append({"dates": seen, "cursor": body["cursor"]})
                    cursor = body["cursor"]
                else:
                    body = get_changes(client, cursor)
                    if body["events"]:
                        bodies.append({"dates": set(), "events": body["events"]})
                    cursor = body["cursor"]

        t = threading.Thread(target=subscriber)
        t.start()
        try:
            for i in range(30):
                date = f"2026-11-{i + 1:02d}"
                assert upload(
                    client, date, [(f"u{i}", "/p", f"{date}T00:00:00Z")]
                ).status_code == 200
                if i % 10 == 9:
                    r = client.post("/erasure", json={"user_id": f"u{i}"})
                    wait_task(client, r.json()["task_id"])
            # give the subscriber a last poll
            time.sleep(0.2)
        finally:
            stop.set()
            t.join(10)

        announced = set()
        for batch in bodies:
            announced |= batch["dates"]
            for event in batch.get("events", []):
                announced.add(event["date"])

        # every date ever committed was announced to the subscriber
        for i in range(30):
            assert f"2026-11-{i + 1:02d}" in announced


def test_cursor_survives_a_restart(tmp_path):
    config = make_config(tmp_path)
    app1 = create_app(config)
    with TestClient(app1) as client:
        upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])
        cursor = page(client, get_changes(client)["cursor"])["cursor"]

    app2 = create_app(config)
    with TestClient(app2) as client2:
        upload(client2, "2026-11-02", [("bob", "/b", "2026-11-02T00:00:00Z")])
        events = page(client2, cursor)["events"]
        # only the new upload appears; earlier events are not renumbered or replayed
        assert [(e["date"], e["action"]) for e in events] == [
            ("2026-11-02", "refetch")
        ]
        # and the numbering continues after the restart
        assert events[0]["seq"] > cursor



def test_failed_write_emits_no_notification(tmp_path, monkeypatch):
    with make_client(tmp_path) as client:
        cursor = get_changes(client)["cursor"]

        def always_fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(storage, "write_table_atomic", always_fail)
        with pytest.raises(OSError, match="disk full"):
            upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])

        assert page(client, cursor)["events"] == []
        assert get_changes(client)["dates"] == []


def test_registration_failure_emits_no_notification_and_leaves_no_file(tmp_path, monkeypatch):
    with make_client(tmp_path) as client:
        service = client.app.state.service
        cursor = get_changes(client)["cursor"]

        def boom(*args, **kwargs):
            raise RuntimeError("state database unavailable")

        monkeypatch.setattr(service.store, "record_upload", boom)
        with pytest.raises(RuntimeError, match="state database unavailable"):
            upload(client, "2026-11-01", [("alice", "/a", "2026-11-01T00:00:00Z")])

        # nothing was announced, nothing is downloadable, no orphan file remains
        assert page(client, cursor)["events"] == []
        assert get_changes(client)["dates"] == []
        assert client.get("/dates/2026-11-01/download").status_code == 404
        assert not list((tmp_path / "data").rglob("part-*.parquet"))
