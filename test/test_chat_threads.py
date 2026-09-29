"""Threads: an ordinary session anchored to one message (``dashboard/chat_threads.py``).

Two halves, and they are tested for different reasons.

The STORE half is the version-1 hardening, retargeted at the anchor index: the
sidecar is still a file beside a transcript under the data home, so every rule
that made a reply durable-or-refused makes an anchor durable-or-refused, and the
link/oversize/torn-file refusals are unchanged. Those tests are the ones that
must not weaken.

The OPEN half is new behaviour and replaces version 1's turn tests wholesale.
There is no envelope, no read-only spec, no one-reply lock and no clipping to
assert, because a thread runs its own ordinary turns; what there is to assert is
that opening mints a real session, records the anchor on BOTH sides, seeds it
once, and never touches the parent's running turn.

Uses ``async with _client()`` inside each test rather than an async-gen fixture:
the CI-pinned ``pytest-asyncio`` is incompatible with the pinned ``pytest`` for
async fixtures (see test_denied_commands_api.py docstring).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state

from kiro_crew import history as history_mod
from kiro_crew.dashboard import chat_threads, create_rate_limit
from kiro_crew.dashboard import session_control as sc
from kiro_crew.dashboard.chat_threads import (
    Anchor,
    ThreadOpenError,
    api_chat_thread_close,
    api_chat_thread_detail,
    api_chat_thread_open,
    api_chat_threads_summary,
    build_seed,
    close_thread,
    in_flight_snapshot,
    open_thread,
    summarize,
    turn_anchor_mid,
)
from kiro_crew.dashboard.chat_utils import slot_history_key
from kiro_crew.dashboard.ws import THREAD_ANCHOR_EVENT, broadcast_thread_anchor
from kiro_crew.history import THREADS_MAX_ANCHORS_PER_SIDECAR, ThreadStoreUnreadable
from kiro_crew.members import DM_SLOT_MODE

_MEMBER_SLOT = "member-radar"
_PLAIN_SLOT = "chat-plain-1"


# ── Harness ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _control_enabled(monkeypatch):
    """``create_session`` is the core a thread is minted through, and it is gated
    by ``agent.session_control`` (default true). Pinned here so these tests assert
    thread behaviour rather than the shipped default of an unrelated switch."""
    monkeypatch.setattr(sc, "session_control_enabled", lambda: True)


@pytest.fixture(autouse=True)
def _agent_resolves(monkeypatch):
    """Make the parent's agent resolve and be bound to the caller's workspace.

    A thread inherits its parent's agent through the create core, which refuses a
    name nothing would dispatch -- correctly, since the slot would then advertise
    an agent that is not the one answering. Tests whose subject is threads still
    have to get past that check; forcing the honored flag keeps them focused
    without weakening it (``test_session_control.py`` owns the refusal itself).
    """
    real = sc.resolve_agent_bindings
    monkeypatch.setattr(sc, "_workspace_name_for_dir", lambda cfg, ws_dir: "default")
    monkeypatch.setattr(
        sc,
        "resolve_agent_bindings",
        lambda cfg, agent_name=None, project_dir=None, **kwargs: dataclasses.replace(
            real(cfg, None, project_dir), requested_resolved=True
        ),
    )


@pytest.fixture(autouse=True)
def _fresh_create_budget():
    """The per-caller create-rate window is process-wide module state, and every
    open in this file mints a session as the same caller: left behind, the Nth
    open in a worker process is refused ``create_rate_limited`` and whichever test
    is Nth fails for a reason it never asserted."""
    create_rate_limit.reset_for_tests()
    yield
    create_rate_limit.reset_for_tests()


def _make_app(state) -> web.Application:
    app = web.Application()
    app["state"] = state
    app.router.add_get("/api/chat/threads", api_chat_threads_summary)
    app.router.add_get("/api/chat/threads/{mid}", api_chat_thread_detail)
    app.router.add_post("/api/chat/threads/{mid}/open", api_chat_thread_open)
    app.router.add_post("/api/chat/threads/{mid}/close", api_chat_thread_close)
    return app


def _client(state, *, app_name: str = "") -> TestClient:
    app = _make_app(state)
    if app_name:

        @web.middleware
        async def _as_app(request, handler):
            request["app"] = app_name
            return await handler(request)

        app.middlewares.append(_as_app)
    return TestClient(TestServer(app))


def _chat(state, key: str = _MEMBER_SLOT, *, mode: str = DM_SLOT_MODE):
    """A chat with one flushed exchange; returns ``(slot, parent_mid)``.

    The transcript is written to DISK because the anchor index only writes beside
    a transcript that exists and that holds the parent's mid -- a chat younger
    than its first flush answers ``transcript_missing``, which is the rule that
    keeps an anchor reachable after a crash.
    """
    kwargs: dict[str, Any] = {"agent": "Radar"}
    if mode:
        kwargs["mode"] = mode
    slot = state.get_or_create_slot(key, **kwargs)
    slot.append("user", "Anything overnight?", broadcast=False)
    row = slot.append(
        "assistant", "Overnight triage: 9 new issues, one needs you.", broadcast=False
    )
    hk = slot_history_key(slot)
    state.conversation_log.append(
        hk, "user", "Anything overnight?", mid=slot.messages[0]["meta"]["mid"]
    )
    state.conversation_log.append(hk, "assistant", row["content"], mid=row["meta"]["mid"])
    return slot, row["meta"]["mid"]


def M(tag: str) -> str:
    """A canonical row id (``m-`` + 16 hex), stable per *tag*, for tests that name
    mids by hand: the store keeps only keys in the minted shape."""
    import hashlib

    return "m-" + hashlib.sha256(tag.encode()).hexdigest()[:16]


def _store_path(state, slot):
    return state.conversation_log.threads_sidecar_path(slot_history_key(slot))


def _anchors(state, slot):
    return state.conversation_log.read_thread_anchors(slot_history_key(slot))


def _identity(state, slot):
    """What the opener captures at admission: the transcript's ``created_at``."""
    return state.conversation_log.thread_transcript_identity(slot_history_key(slot))


def _anchor_row(slot_key: str = "chat-9", **over) -> dict[str, Any]:
    row = {
        "thread_slot": slot_key,
        "title": "why is triage slow",
        "opened_by": "user",
        "opened_at": "2026-09-29T01:00:00+00:00",
        "closed_at": None,
        "summary_mid": None,
    }
    row.update(over)
    return row


def _no_seed(monkeypatch) -> list[dict[str, Any]]:
    """Record seed deliveries instead of running the thread's first turn.

    ``send_to_target`` is the real delivery verb and would start a model turn;
    what these tests assert about the seed is that exactly ONE is delivered, to
    the thread's own slot, carrying the anchored message -- not what a model does
    with it.
    """
    sent: list[dict[str, Any]] = []

    async def _record(state, *, caller_session_key, target, message, steer=False, **kw):
        sent.append(
            {
                "caller": caller_session_key,
                "target": target,
                "message": message,
                "steer": steer,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(sc, "send_to_target", _record)
    return sent


def _capture_broadcasts(state) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []

    def _record(msg_type, data):
        events.append((msg_type, data))

    state.broadcast_ws = _record
    state.broadcast_ws_owners = _record
    return events


# ── Store: the version-1 hardening, retargeted at the anchor index ───────────


def test_a_version_1_sidecar_still_reads(tmp_path):
    """The legacy replies keep rendering, and the reader tells the two halves
    apart: a v1 document has no anchors, and asking for them is not an error."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    path = _store_path(state, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "threads": {
                    mid: [
                        {
                            "id": "0" * 32,
                            "role": "user",
                            "content": "why nine?",
                            "ts": "2026-09-22T07:41:00+00:00",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    assert log.read_threads(key)[mid][0]["content"] == "why nine?"
    # No anchors half: {} rather than a refusal. A v1 file is not damaged.
    assert log.read_thread_anchors(key) == {}


def test_writing_an_anchor_preserves_the_legacy_replies(tmp_path):
    """Both halves live in ONE file, so the anchor writer must carry the half it
    does not own. A v1 thread whose replies vanished when someone opened a new
    thread on another message would be data loss with no error."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    path = _store_path(state, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "threads": {
                    M("other"): [{"id": "1" * 32, "role": "user", "content": "keep me", "ts": ""}]
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        log.write_thread_anchor(key, mid, _anchor_row(), expected_created_at=_identity(state, slot))
        == "ok"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["version"] == 2
    assert document["anchors"][mid]["thread_slot"] == "chat-9"
    assert document["threads"][M("other")][0]["content"] == "keep me"
    # And both halves still read back through their own readers.
    assert log.read_threads(key)[M("other")][0]["content"] == "keep me"
    assert log.read_thread_anchors(key)[mid]["title"] == "why is triage slow"


def test_an_anchor_is_refused_without_a_flushed_parent(tmp_path):
    """An anchor is durable only through the row it hangs off. A parent that
    exists only in the slot's memory window would leave the thread unreachable if
    the process died before the flush, so it is refused and the caller retries."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    unflushed = slot.append("assistant", "not on disk yet", broadcast=False)["meta"]["mid"]
    identity = _identity(state, slot)
    assert log.thread_anchor_admissible(key, unflushed, expected_created_at=identity) == "unflushed"
    assert (
        log.write_thread_anchor(key, unflushed, _anchor_row(), expected_created_at=identity)
        == "unflushed"
    )
    assert log.read_thread_anchors(key) == {}


def test_an_anchor_never_lands_in_a_replacement_transcript(tmp_path):
    """A chat deleted and recreated under the same key is a DIFFERENT chat. The
    identity captured at admission is what tells it apart, so an anchor admitted
    against the old one is refused rather than attached to the new."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    stale = _identity(state, slot)
    log.delete_session(key)
    fresh_slot, fresh_mid = _chat(state, _MEMBER_SLOT)
    assert log.thread_anchor_admissible(key, mid, expected_created_at=stale) in (
        "replaced",
        "missing",
        "unflushed",
    )
    assert log.write_thread_anchor(key, mid, _anchor_row(), expected_created_at=stale) in (
        "replaced",
        "missing",
        "unflushed",
    )
    assert mid not in log.read_thread_anchors(slot_history_key(fresh_slot))
    assert fresh_mid


def test_no_anchor_is_written_beside_a_missing_transcript(tmp_path):
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("member-ghost", agent="Radar", mode=DM_SLOT_MODE)
    log = state.conversation_log
    key = slot_history_key(slot)
    assert log.write_thread_anchor(key, M("1"), _anchor_row()) == "missing"
    assert not _store_path(state, slot).exists()


def test_one_open_thread_per_message_and_a_closed_one_reopens(tmp_path):
    """A second thread on the same message would split the discussion in two
    places with no way to tell which is live, so an OPEN anchor refuses. A CLOSED
    one does not: closing is what makes the message available again."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    identity = _identity(state, slot)
    assert log.write_thread_anchor(key, mid, _anchor_row(), expected_created_at=identity) == "ok"
    assert (
        log.write_thread_anchor(key, mid, _anchor_row("chat-10"), expected_created_at=identity)
        == "duplicate"
    )
    assert log.read_thread_anchors(key)[mid]["thread_slot"] == "chat-9"
    assert log.update_thread_anchor(key, mid, {"closed_at": "2026-09-29T02:00:00+00:00"}) == "ok"
    assert (
        log.write_thread_anchor(key, mid, _anchor_row("chat-10"), expected_created_at=identity)
        == "ok"
    )
    assert log.read_thread_anchors(key)[mid]["thread_slot"] == "chat-10"


def test_closing_an_absent_anchor_is_named_not_invented(tmp_path):
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    assert log.update_thread_anchor(key, mid, {"closed_at": "2026-09-29T02:00:00+00:00"}) == (
        "not_found"
    )


def test_an_unreadable_sidecar_is_refused_never_overwritten(tmp_path):
    """Both readers refuse and the writer refuses, so a damaged file is never
    replaced with an empty one. ``{"threads": []}`` refuses TOO: a half that is
    present but the wrong shape is damage, and reading it as absent is exactly
    what would let a write erase it."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    path = _store_path(state, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ThreadStoreUnreadable):
        log.read_threads(key)
    with pytest.raises(ThreadStoreUnreadable):
        log.read_thread_anchors(key)
    with pytest.raises(ThreadStoreUnreadable):
        log.write_thread_anchor(key, mid, _anchor_row(), expected_created_at=_identity(state, slot))
    assert path.read_text(encoding="utf-8") == "{not json"

    path.write_text(json.dumps({"threads": []}), encoding="utf-8")
    with pytest.raises(ThreadStoreUnreadable):
        log.read_threads(key)
    with pytest.raises(ThreadStoreUnreadable):
        log.read_thread_anchors(key)

    path.write_text(json.dumps({"anchors": []}), encoding="utf-8")
    with pytest.raises(ThreadStoreUnreadable):
        log.read_thread_anchors(key)

    # A document that is neither v1 nor v2 is not this store's file.
    path.write_text(json.dumps({"something": {}}), encoding="utf-8")
    with pytest.raises(ThreadStoreUnreadable):
        log.read_thread_anchors(key)

    # An anchors-only document is v2 and reads.
    path.write_text(json.dumps({"version": 2, "anchors": {}}), encoding="utf-8")
    assert log.read_thread_anchors(key) == {}
    assert log.read_threads(key) == {}


def test_every_retained_anchor_field_is_held_to_the_writers_shape(tmp_path):
    """The file sits under the data home, so a row another writer put there must
    not reach the dashboard through a response's spread. Only ``title`` may carry
    prose; every other field is a shape that cannot, and a row that breaks one is
    dropped whole rather than half-read."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    path = _store_path(state, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "anchors": {
                    # Keys that are not minted row ids are not anchors.
                    "AKIAIOSFODNN7EXAMPLE": _anchor_row(),
                    # No thread_slot: there is nothing to open.
                    M("a"): {k: v for k, v in _anchor_row().items() if k != "thread_slot"},
                    # A slot key that could carry prose.
                    M("b"): _anchor_row("not a slot key!! ghp_secret"),
                    # An opener outside {user, agent:<key>}.
                    M("c"): _anchor_row(opened_by="ghp_secret"),
                    # Timestamps that are not instants.
                    M("d"): _anchor_row(opened_at="ghp_secret"),
                    M("e"): _anchor_row(closed_at="ghp_secret"),
                    # A summary pointer that is not a row id.
                    M("f"): _anchor_row(summary_mid="AKIAIOSFODNN7EXAMPLE"),
                    M("g"): "not-a-dict",
                    # The one good row, with a field the store never writes and an
                    # over-long title.
                    M("keep"): {**_anchor_row("chat-42"), "token": "AKIA...", "title": "t" * 900},
                },
            }
        ),
        encoding="utf-8",
    )
    read = log.read_thread_anchors(key)
    assert set(read) == {M("keep")}
    kept = read[M("keep")]
    assert set(kept) == {
        "thread_slot",
        "title",
        "opened_by",
        "opened_at",
        "closed_at",
        "summary_mid",
    }
    assert kept["thread_slot"] == "chat-42"
    assert len(kept["title"]) == 200


def test_the_anchor_count_is_bounded(tmp_path):
    """A bound on how many messages of one chat can carry a thread, so what the
    file holds never decides what the gateway holds."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    path = _store_path(state, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    over = THREADS_MAX_ANCHORS_PER_SIDECAR + 25
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "anchors": {M(f"n{i}"): _anchor_row(f"chat-{i}") for i in range(over)},
            }
        ),
        encoding="utf-8",
    )
    assert len(log.read_thread_anchors(key)) == THREADS_MAX_ANCHORS_PER_SIDECAR


def test_a_link_at_the_threads_dir_gets_no_anchor_write(tmp_path, monkeypatch):
    """The ``.threads`` directory is code-created beside the transcripts. A link
    at that name would carry the write outside the session store."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    path = _store_path(state, slot)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    linked = True
    try:
        path.parent.symlink_to(elsewhere, target_is_directory=True)
    except (OSError, NotImplementedError):
        # A platform that will not MAKE a link is still a platform the refusal has
        # to hold on, so the assertion runs either way rather than being skipped:
        # the guard is driven through the seam the writer itself reads
        # (`platform_compat.is_link_or_junction`), which is the branch under test.
        # What the real link additionally proves -- that nothing landed on the far
        # side -- needs a real link, so that half is asserted only when there is one.
        linked = False
        path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            history_mod.platform_compat,
            "is_link_or_junction",
            lambda p, _parent=path.parent: Path(p) == _parent,
        )
    with pytest.raises(ThreadStoreUnreadable):
        log.write_thread_anchor(key, mid, _anchor_row(), expected_created_at=_identity(state, slot))
    if linked:
        assert not list(elsewhere.iterdir())
    else:
        assert not path.exists()


def test_a_late_duplicate_names_the_thread_that_won_not_the_one_that_ended(tmp_path, monkeypatch):
    """A refusal's `thread_slot` is what the caller opens next, so on a race against
    a CLOSED anchor it must not be the pre-mint read's ENDED thread. Closing releases
    the message, both openers pass the probe, and the loser would otherwise point the
    reader at the conversation they already finished."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    identity = _identity(state, slot)
    # A thread lived here and was closed, so the probe admits a new one.
    assert log.write_thread_anchor(
        key, mid, _anchor_row("chat-ended"), expected_created_at=identity
    )
    assert (
        log.update_thread_anchor(
            key, mid, {"closed_at": "2026-09-29T07:31:00+00:00"}, expect_open=True
        )
        == "ok"
    )
    _no_seed(monkeypatch)
    real_write = log.write_thread_anchor

    def _lose_the_race(*a: Any, **k: Any) -> str:
        # The winner lands while this opener is minting its session.
        real_write(key, mid, _anchor_row("chat-winner"), expected_created_at=identity)
        return "duplicate"

    monkeypatch.setattr(log, "write_thread_anchor", _lose_the_race)
    with pytest.raises(ThreadOpenError) as refused:
        asyncio.run(
            open_thread(
                state, Anchor("dashboard", slot.key, mid), title="t", opened_by="user", seed="s"
            )
        )
    assert refused.value.code == "already_closed" or refused.value.code == "already_open"
    assert refused.value.thread_slot == "chat-winner"


def test_a_stale_close_cannot_end_the_thread_that_replaced_its_own(tmp_path, monkeypatch):
    """`expect_open` alone is satisfied again by a REPLACEMENT. Two closes read one
    open anchor; the first commits and an opener takes the freed message, so by the
    time the second WRITE lands the row is open again -- a different thread. Without
    naming the thread, that close ends a live one it never saw."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    identity = _identity(state, slot)
    assert log.write_thread_anchor(
        key, mid, _anchor_row("chat-first"), expected_created_at=identity
    )
    _no_seed(monkeypatch)
    real_update = log.update_thread_anchor
    interleaved = {"done": False}

    def _race(*a: Any, **k: Any) -> str:
        # Between this close's READ and its write: the other close commits, and an
        # opener installs a replacement on the message the commit just freed.
        if not interleaved["done"]:
            interleaved["done"] = True
            assert (
                real_update(
                    key,
                    mid,
                    {"closed_at": "2026-09-29T07:31:00+00:00"},
                    expect_open=True,
                    expect_thread_slot="chat-first",
                )
                == "ok"
            )
            assert (
                log.write_thread_anchor(
                    key, mid, _anchor_row("chat-second"), expected_created_at=identity
                )
                == "ok"
            )
        return real_update(*a, **k)

    monkeypatch.setattr(log, "update_thread_anchor", _race)
    with pytest.raises(ThreadOpenError) as refused:
        asyncio.run(close_thread(state, Anchor("dashboard", slot.key, mid)))
    assert refused.value.code == "already_closed"
    # The replacement is untouched: still open, still its own thread, no card.
    live = log.read_thread_anchors(key)[mid]
    assert live["thread_slot"] == "chat-second"
    assert live["closed_at"] is None
    assert live["summary_mid"] is None


def test_an_amendment_is_refused_once_a_new_thread_holds_the_message(tmp_path):
    """Closing frees the message to carry a new thread, so the card's id -- written
    after the close -- must not land on whatever anchor is there when it arrives.
    Stamped onto a replacement it would give a LIVE thread a back-link to a closed
    thread's card."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    identity = _identity(state, slot)
    assert log.write_thread_anchor(key, mid, _anchor_row("chat-9"), expected_created_at=identity)
    assert (
        log.update_thread_anchor(
            key, mid, {"closed_at": "2026-09-29T07:31:00+00:00"}, expect_open=True
        )
        == "ok"
    )
    # The message is free again, and a second thread takes it.
    assert (
        log.write_thread_anchor(key, mid, _anchor_row("chat-10"), expected_created_at=identity)
        == "ok"
    )
    assert (
        log.update_thread_anchor(key, mid, {"summary_mid": M("card1")}, expect_thread_slot="chat-9")
        == "replaced"
    )
    live = log.read_thread_anchors(key)[mid]
    assert live["thread_slot"] == "chat-10"
    assert live["summary_mid"] is None
    # The same amendment on the thread it belongs to still lands.
    assert (
        log.update_thread_anchor(
            key, mid, {"summary_mid": M("card2")}, expect_thread_slot="chat-10"
        )
        == "ok"
    )


def test_an_anchor_change_that_breaks_the_shape_is_refused(tmp_path):
    """``update_thread_anchor`` re-validates the MERGED row, so a close cannot
    write a row the reader would then drop -- which would silently delete the
    anchor it was closing."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    identity = _identity(state, slot)
    assert log.write_thread_anchor(key, mid, _anchor_row(), expected_created_at=identity) == "ok"
    with pytest.raises(ValueError):
        log.update_thread_anchor(key, mid, {"closed_at": "whenever"})
    assert log.read_thread_anchors(key)[mid]["closed_at"] is None


# ── The in-flight snapshot ───────────────────────────────────────────────────


def _stream(slot, *deltas):
    """Append ``chunk`` rows the way ``chat_runner`` streams them."""
    for i, delta in enumerate(deltas, start=1):
        row = slot.append("chunk", delta, "chunk", broadcast=False)
        row["seq"] = i


def test_the_snapshot_reads_the_streaming_text_without_consuming_it(tmp_path):
    """The dashboard does not run turns through ``TurnDriver``, so the readable
    copy of a streaming reply is the slot's own ``chunk`` rows. Reading them must
    leave the stream exactly as it was: the turn is still writing."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    _stream(slot, "Looking at ", "the queue", " now.")
    before = len(slot.messages)
    assert in_flight_snapshot(slot) == "Looking at the queue now."
    # Nothing consumed: the rows are still there, and reading twice is the same.
    assert len(slot.messages) == before
    assert in_flight_snapshot(slot) == "Looking at the queue now."
    assert sum(1 for m in slot.messages if m.get("role") == "chunk") == 3


def test_a_finished_message_has_no_snapshot(tmp_path):
    """The ordinary case. An empty snapshot is what says 'nothing is streaming',
    and it must not be confused with a read that failed."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    assert in_flight_snapshot(slot) == ""


def test_a_segment_finalizing_under_the_read_cannot_empty_the_snapshot(tmp_path):
    """``purge_chunks`` REBINDS ``slot.messages``, so a reader that copied the
    list reference first holds the pre-purge rows. That is what makes the read
    race-safe against the turn finalizing its segment mid-read."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    _stream(slot, "half ", "an answer")
    rows = list(slot.messages)
    slot.purge_chunks()
    assert not any(m.get("role") == "chunk" for m in slot.messages)
    # The copy taken before the purge still carries the text.
    assert "".join(m["content"] for m in rows if m.get("role") == "chunk") == "half an answer"


def test_the_snapshot_is_redacted(tmp_path):
    """The snapshot is quoted into a message that leaves the dashboard's storage
    (kiro-cli persists the thread's transcript), so it is scrubbed on the way out
    -- whole, before any cut, so a credential cannot survive as a fragment."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    _stream(slot, "token is AKIA", "IOSFODNN7EXAMPLE done")
    assert "AKIAIOSFODNN7EXAMPLE" not in in_flight_snapshot(slot)


# ── Anchor resolution for a streaming reply ──────────────────────────────────


def test_a_streaming_reply_anchors_to_the_message_that_started_the_turn(tmp_path):
    """A streaming assistant row has no mid -- ids are minted post-turn -- so
    there is nothing to hang an anchor off. The message that ASKED for the work
    is persisted, has a mid, and is where the thread belongs anyway."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    asked = slot.append("user", "why nine?", broadcast=False)
    state.conversation_log.append(
        slot_history_key(slot), "user", "why nine?", mid=asked["meta"]["mid"]
    )
    _stream(slot, "Because ", "three are dupes")
    rows = [r for r in slot.messages if r.get("role") in ("user", "assistant")]
    assert turn_anchor_mid(rows) == asked["meta"]["mid"]


def test_a_conversation_with_no_user_row_has_no_anchor_to_offer(tmp_path):
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-empty-1")
    assert turn_anchor_mid([]) == ""
    slot.append("assistant", "unprompted", broadcast=False)
    assert turn_anchor_mid(list(slot.messages)) == ""


# ── The seed ─────────────────────────────────────────────────────────────────


def test_the_seed_quotes_the_message_and_the_partial_reply():
    """One message, sent once. Version 1 rebuilt a whole envelope per reply
    because the session was cold; a warm session needs the quote once."""
    seed = build_seed(
        {"role": "assistant", "content": "9 new issues, one needs you."},
        in_flight_text="Looking at the queue now.",
    )
    assert "Assistant: 9 new issues, one needs you." in seed
    assert "Looking at the queue now." in seed
    assert chat_threads._SEED_IN_FLIGHT_NOTE in seed


def test_a_reply_that_was_in_flight_but_unreadable_is_said_so():
    """The honest fallback. A thread told nothing would read the quoted message as
    the end of the conversation, when in fact a reply was being written."""
    seed = build_seed({"role": "user", "content": "why nine?"}, in_flight_unread=True)
    assert chat_threads._SEED_NO_SNAPSHOT_NOTE in seed
    assert chat_threads._SEED_IN_FLIGHT_NOTE not in seed


def test_the_seed_carries_the_openers_own_note():
    seed = build_seed({"role": "user", "content": "q"}, extra="paths: src/a.py, src/b.py")
    assert "paths: src/a.py, src/b.py" in seed


# ── summarize ────────────────────────────────────────────────────────────────


def test_summarize_folds_anchors_and_the_legacy_half():
    """The footer renders one badge per message and does not care which era the
    thread came from, so both shapes come out of one map -- and an anchor wins on
    a message that has both, because the live thread is the one to open."""
    out = summarize(
        {M("live"): _anchor_row("chat-7"), M("both"): _anchor_row("chat-8")},
        {
            M("old"): [{"id": "0" * 32, "role": "user", "content": "x", "ts": "t1"}],
            M("both"): [{"id": "1" * 32, "role": "user", "content": "y", "ts": "t2"}],
        },
    )
    assert out[M("live")]["kind"] == "session"
    assert out[M("live")]["thread_slot"] == "chat-7"
    assert out[M("old")] == {
        "kind": "legacy",
        "count": 1,
        "last_reply_ts": "t1",
        "participants": ["user"],
    }
    assert out[M("both")]["kind"] == "session"
    assert out[M("both")]["thread_slot"] == "chat-8"


def test_summarize_redacts_the_one_free_text_field():
    out = summarize({M("a"): _anchor_row(title="key AKIAIOSFODNN7EXAMPLE")})
    assert "AKIAIOSFODNN7EXAMPLE" not in out[M("a")]["title"]


def test_summarize_skips_an_empty_legacy_thread():
    assert summarize({}, {M("a"): []}) == {}


# ── open_thread ──────────────────────────────────────────────────────────────


def test_opening_mints_a_real_session_and_records_the_anchor(tmp_path, monkeypatch):
    """The whole design in one test. A thread is an ordinary slot, and the anchor
    lives in ONE place on this surface -- the parent's index -- which is what lets
    the parent list its threads in one read. A second copy on the thread's own
    metadata had no reader, so it is not written.

    Opened with ``start_turn``, the opener-asked-a-question case, so the seed's
    delivery is visible here."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    sent = _no_seed(monkeypatch)
    opened = asyncio.run(
        open_thread(
            state,
            Anchor("dashboard", slot.key, mid),
            title="why is triage slow",
            opened_by="user",
            seed="seed text",
            start_turn=True,
        )
    )
    thread_key = opened["thread_slot"]
    thread = state.get_slot(thread_key)
    # An ORDINARY slot: in the table, in the sidebar, with the parent's workspace.
    assert thread is not None
    assert thread.workspace == slot.workspace
    assert thread.title == "why is triage slow"
    # The anchor is in the parent's index, and NOT copied onto the thread: a second
    # record with no reader is a shape to keep consistent for nothing.
    assert "_thread_anchor" not in state.conversation_log.get_metadata(slot_history_key(thread))
    assert _anchors(state, slot)[mid]["thread_slot"] == thread_key
    assert _anchors(state, slot)[mid]["closed_at"] is None
    # Exactly one seed, to the thread's own slot, from the parent as caller.
    assert len(sent) == 1
    assert sent[0]["target"] == thread_key
    assert sent[0]["caller"] == slot_history_key(slot)
    assert sent[0]["steer"] is False
    assert "seed text" in sent[0]["message"]
    assert opened["seeded"] is True


def test_opening_does_not_fork_the_parents_transcript(tmp_path, monkeypatch):
    """``session_fork`` is the only verb that carries a transcript, and it is the
    wrong one: it copies the whole parent and refuses an ``agent`` override, while
    a thread wants a curated seed. So the thread carries the seed and nothing else
    -- one row, and none of the parent's own messages."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    asyncio.run(
        open_thread(
            state,
            Anchor("dashboard", slot.key, mid),
            title="t",
            opened_by="user",
            seed="the seed",
        )
    )
    parent_said = [m["content"] for m in slot.messages if m.get("role") in ("user", "assistant")]
    assert parent_said  # the parent has a transcript to copy, so the next line means something
    thread = state.get_slot(_anchors(state, slot)[mid]["thread_slot"])
    assert thread is not None
    said = [m for m in thread.messages if m.get("role") in ("user", "assistant")]
    assert [m["content"] for m in said] == ["the seed"]
    assert not [m for m in said if m["content"] in parent_said]


def test_the_landed_seed_is_on_DISK_before_the_open_reports_it(tmp_path, monkeypatch):
    """``slot.append`` only marks the slot dirty for a later save, so reporting
    ``seeded`` off it alone would acknowledge a row a restart can still lose --
    and the anchor is already persisted by then, so what would survive is a thread
    whose quote never existed while the response said it did. The durable write is
    awaited, and carries the window row's own mid so a bounded read reconciles the
    two as ONE message instead of appending a second copy."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    opened = asyncio.run(
        open_thread(
            state,
            Anchor("dashboard", slot.key, mid),
            title="t",
            opened_by="user",
            seed="quoted anchor",
        )
    )
    assert opened["seeded"] is True
    thread = state.get_slot(opened["thread_slot"])
    assert thread is not None
    persisted = state.conversation_log.read_messages(slot_history_key(thread))
    said = [r for r in persisted if r.get("role") == "user"]
    assert [r["content"] for r in said] == ["quoted anchor"]
    # ONE row, carrying the window copy's identity.
    window = [m for m in thread.messages if m.get("role") == "user"]
    assert len(window) == 1
    assert said[0].get("meta", {}).get("mid") == window[0]["meta"]["mid"]


def test_a_click_with_no_note_lands_the_seed_without_running_it(tmp_path, monkeypatch):
    """The drawer's empty hint tells the person to type to start the thread, so a
    bare **Reply in thread** must not have started a turn behind that hint. The
    seed still lands, as an ordinary user row the first real turn reads as history
    -- so the quote is context, not an instruction acted on before anybody read it."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    sent = _no_seed(monkeypatch)
    opened = asyncio.run(
        open_thread(
            state,
            Anchor("dashboard", slot.key, mid),
            title="t",
            opened_by="user",
            seed="quoted anchor",
        )
    )
    assert sent == []
    assert opened["seeded"] is True
    thread = state.get_slot(opened["thread_slot"])
    assert thread is not None
    assert [m["content"] for m in thread.messages if m.get("role") == "user"] == ["quoted anchor"]


def test_opening_never_touches_the_parents_running_turn(tmp_path, monkeypatch):
    """Requirement 1. Open holds no semaphore, does not gate on the parent being
    busy, and takes only a snapshot: the parent's streamed rows and its task are
    exactly as they were."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    _stream(slot, "still ", "writing")
    chunks_before = [m["content"] for m in slot.messages if m.get("role") == "chunk"]
    asyncio.run(
        open_thread(
            state,
            Anchor("dashboard", slot.key, mid),
            title="t",
            opened_by="user",
            seed="s",
        )
    )
    assert [m["content"] for m in slot.messages if m.get("role") == "chunk"] == chunks_before


def test_a_second_thread_on_the_same_message_points_at_the_first(tmp_path, monkeypatch):
    """The refusal names the thread that already exists, so the caller can open
    it instead of making the person hunt for it."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    first = asyncio.run(
        open_thread(
            state, Anchor("dashboard", slot.key, mid), title="t", opened_by="user", seed="s"
        )
    )
    with pytest.raises(ThreadOpenError) as caught:
        asyncio.run(
            open_thread(
                state, Anchor("dashboard", slot.key, mid), title="t2", opened_by="user", seed="s"
            )
        )
    assert caught.value.code == "already_open"
    assert caught.value.thread_slot == first["thread_slot"]


def test_an_unflushed_parent_costs_no_session(tmp_path, monkeypatch):
    """The admission probe runs BEFORE the mint, so the common refusal does not
    leave an orphan session in the sidebar."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    _no_seed(monkeypatch)
    unflushed = slot.append("assistant", "not on disk", broadcast=False)["meta"]["mid"]
    before = set(state._slots)
    with pytest.raises(ThreadOpenError) as caught:
        asyncio.run(
            open_thread(
                state,
                Anchor("dashboard", slot.key, unflushed),
                title="t",
                opened_by="user",
                seed="s",
            )
        )
    assert caught.value.code == "transcript_missing"
    assert set(state._slots) == before


def test_a_failed_anchor_write_retracts_the_minted_session(tmp_path, monkeypatch):
    """An anchorless thread is worse than no thread: a session in the sidebar the
    conversation it belongs to cannot find, seeded with a quote of a message
    nothing links it to."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    log = state.conversation_log

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(log, "write_thread_anchor", _boom)
    before = set(state._slots)
    with pytest.raises(ThreadOpenError) as caught:
        asyncio.run(
            open_thread(
                state, Anchor("dashboard", slot.key, mid), title="t", opened_by="user", seed="s"
            )
        )
    assert caught.value.code == "threads_unavailable"
    assert set(state._slots) == before


def test_an_unknown_surface_is_refused_here(tmp_path):
    """The dashboard adapter is what ships in this module. A Slack anchor is
    recorded by the Slack adapter, not by reaching through this one."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    with pytest.raises(ThreadOpenError) as caught:
        asyncio.run(
            open_thread(state, Anchor("slack", "C123", mid), title="t", opened_by="user", seed="s")
        )
    assert caught.value.code == "surface_unsupported"


def test_an_unrecognisable_opener_records_as_user(tmp_path, monkeypatch):
    """Held to the store's shape before the write rather than trusted: a session
    key the index does not admit would otherwise raise AFTER the slot was minted."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    asyncio.run(
        open_thread(
            state,
            Anchor("dashboard", slot.key, mid),
            title="t",
            opened_by="agent:has spaces and ünicode",
            seed="s",
        )
    )
    assert _anchors(state, slot)[mid]["opened_by"] == "user"


def test_a_seed_that_cannot_be_delivered_still_leaves_a_usable_thread(tmp_path, monkeypatch):
    """The thread EXISTS once its anchor is recorded, so retracting it over a
    message would throw away a real thread. The result says which happened."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)

    async def _refuse(*a, **kw):
        raise RuntimeError("no")

    monkeypatch.setattr(sc, "send_to_target", _refuse)
    opened = asyncio.run(
        open_thread(
            state,
            Anchor("dashboard", slot.key, mid),
            title="t",
            opened_by="user",
            seed="s",
            start_turn=True,
        )
    )
    assert opened["seeded"] is False
    assert state.get_slot(opened["thread_slot"]) is not None
    assert _anchors(state, slot)[mid]["thread_slot"] == opened["thread_slot"]


# ── Close ────────────────────────────────────────────────────────────────────


def test_closing_posts_a_card_that_survives_a_transcript_read(tmp_path, monkeypatch):
    """The card is an ORDINARY assistant row carrying ``meta.thread_summary``.
    A bespoke role would be dropped by every reader that does not know it; a row
    with meta reads as prose in those and as a card in the one that does -- and
    its mid is the back-link the anchor then records."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    opened = asyncio.run(
        open_thread(
            state,
            Anchor("dashboard", slot.key, mid),
            title="why is triage slow",
            opened_by="user",
            seed="s",
        )
    )
    closed = asyncio.run(close_thread(state, Anchor("dashboard", slot.key, mid)))
    card = slot.messages[-1]
    assert card["role"] == "assistant"
    assert "why is triage slow" in card["content"]
    assert card["meta"]["thread_summary"]["thread_slot"] == opened["thread_slot"]
    assert card["meta"]["mid"] == closed["summary_mid"]
    anchor = _anchors(state, slot)[mid]
    assert anchor["closed_at"] is not None
    assert anchor["summary_mid"] == closed["summary_mid"]


def test_a_close_whose_anchor_write_fails_posts_no_card(tmp_path, monkeypatch):
    """Persist before you publish. The card is a durable row in the parent AND it
    is broadcast, so posting it first would leave a "thread closed" card standing
    over a thread the store still reads as open -- and `sidecar_full` fails the
    same way every time, so a retry appends another card and none converges."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    asyncio.run(
        open_thread(
            state, Anchor("dashboard", slot.key, mid), title="t", opened_by="user", seed="s"
        )
    )
    rows_before = len(slot.messages)
    monkeypatch.setattr(
        state.conversation_log, "update_thread_anchor", lambda *a, **k: "sidecar_full"
    )
    with pytest.raises(ThreadOpenError):
        asyncio.run(close_thread(state, Anchor("dashboard", slot.key, mid)))
    assert len(slot.messages) == rows_before
    # And the thread is readable as OPEN, which is what the refusal claimed.
    assert _anchors(state, slot)[mid]["closed_at"] is None


def test_a_close_whose_card_pointer_is_lost_still_closes(tmp_path, monkeypatch):
    """The other side of the order: once the transition is durable the close HAS
    happened, so a failure to store the card's mid costs the back-link and must
    not report a close that did occur as a failure."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    asyncio.run(
        open_thread(
            state, Anchor("dashboard", slot.key, mid), title="t", opened_by="user", seed="s"
        )
    )
    real = state.conversation_log.update_thread_anchor
    calls: list[dict] = []

    def once(key, anchor_mid, changes, **kw):
        calls.append(changes)
        if "summary_mid" in changes and changes["summary_mid"]:
            raise OSError("sidecar gone")
        return real(key, anchor_mid, changes, **kw)

    monkeypatch.setattr(state.conversation_log, "update_thread_anchor", once)
    closed = asyncio.run(close_thread(state, Anchor("dashboard", slot.key, mid)))
    assert closed["summary_mid"]
    assert _anchors(state, slot)[mid]["closed_at"] is not None
    # Two writes, in this order: the transition, then the pointer.
    assert [sorted(c) for c in calls] == [["closed_at", "summary_mid"], ["summary_mid"]]


def test_two_closes_racing_the_lock_post_one_card(tmp_path, monkeypatch):
    """Closing once is a property of the store's lock, not of the caller's read.
    Both closers read the anchor open; the second one's write has to be refused
    inside the lock, or both go on to post a summary card in the parent."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    asyncio.run(
        open_thread(
            state, Anchor("dashboard", slot.key, mid), title="t", opened_by="user", seed="s"
        )
    )
    rows_before = len(slot.messages)
    log = state.conversation_log
    real = log.update_thread_anchor
    # Stand in for the loser of the race: its own pre-read saw the thread open,
    # and by the time it takes the lock the winner has closed it.
    log.update_thread_anchor(slot_history_key(slot), mid, {"closed_at": "2026-09-29T08:00:00Z"})
    monkeypatch.setattr(log, "update_thread_anchor", real)
    with pytest.raises(ThreadOpenError) as caught:
        asyncio.run(close_thread(state, Anchor("dashboard", slot.key, mid)))
    assert caught.value.code == "already_closed"
    assert len(slot.messages) == rows_before


def test_a_retracted_open_leaves_no_session_behind(tmp_path, monkeypatch):
    """A minted session whose anchor could not be recorded never became a thread.
    Closing a slot ARCHIVES its conversation, so retracting by closing alone would
    leave it findable in history belonging to no conversation -- and the retraction
    runs before the seed, so there is no message of anyone's to keep."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    log = state.conversation_log
    monkeypatch.setattr(log, "write_thread_anchor", lambda *a, **k: "duplicate")
    before = {s.get("key") for s in log.list_sessions()}
    with pytest.raises(ThreadOpenError):
        asyncio.run(
            open_thread(
                state, Anchor("dashboard", slot.key, mid), title="t", opened_by="user", seed="s"
            )
        )
    after = {s.get("key") for s in log.list_sessions()}
    assert after <= before, f"retraction left {sorted(after - before)} in history"


def test_closing_twice_is_named_not_repeated(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    asyncio.run(
        open_thread(
            state, Anchor("dashboard", slot.key, mid), title="t", opened_by="user", seed="s"
        )
    )
    asyncio.run(close_thread(state, Anchor("dashboard", slot.key, mid)))
    with pytest.raises(ThreadOpenError) as caught:
        asyncio.run(close_thread(state, Anchor("dashboard", slot.key, mid)))
    assert caught.value.code == "already_closed"


def test_closing_a_message_with_no_thread_is_refused(tmp_path):
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    with pytest.raises(ThreadOpenError) as caught:
        asyncio.run(close_thread(state, Anchor("dashboard", slot.key, mid)))
    assert caught.value.code == "thread_not_found"


# ── Routes ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_open_route_mints_a_thread_on_a_plain_chat(tmp_path, monkeypatch):
    """Threads work on EVERY chat surface now. Version 1 answered
    ``409 not_crewmate_chat`` off a member-mode slot because the turn ran as the
    crewmate; a thread runs as its own session, so the gate is gone."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state, _PLAIN_SLOT, mode="")
    assert slot.mode != DM_SLOT_MODE
    _no_seed(monkeypatch)
    async with _client(state) as client:
        resp = await client.post(
            f"/api/chat/threads/{mid}/open",
            json={"slot_key": slot.key, "title": "why nine"},
        )
        assert resp.status == 201
        body = await resp.json()
    assert body["anchor"] == {"surface": "dashboard", "conversation": slot.key, "mid": mid}
    assert state.get_slot(body["thread_slot"]) is not None


@pytest.mark.asyncio
async def test_the_close_route_posts_the_card_and_stamps_the_anchor(tmp_path, monkeypatch):
    """Close has a registered route, so the summary card has a way to reach the
    parent conversation and the documented API is the one that exists. Without a
    route the whole close path would be reachable only from a test."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    async with _client(state) as client:
        opened = await client.post(
            f"/api/chat/threads/{mid}/open",
            json={"slot_key": slot.key, "title": "why nine"},
        )
        assert opened.status == 201
        thread_slot = (await opened.json())["thread_slot"]
        resp = await client.post(
            f"/api/chat/threads/{mid}/close",
            json={"slot_key": slot.key, "thread_slot": thread_slot},
        )
        assert resp.status == 200
        body = await resp.json()
        # A second close is named, not repeated: one card per thread.
        again = await client.post(
            f"/api/chat/threads/{mid}/close",
            json={"slot_key": slot.key, "thread_slot": thread_slot},
        )
        assert again.status == 409
        assert (await again.json())["code"] == "already_closed"
    assert body["thread_slot"] == thread_slot
    card_mid = body["summary_mid"]
    assert card_mid
    entry = state.conversation_log.read_thread_anchors(slot_history_key(slot))[mid]
    assert entry["closed_at"] and entry["summary_mid"] == card_mid
    # The card is an ORDINARY assistant row carrying the back-link.
    card = next(m for m in slot.messages if m.get("meta", {}).get("mid") == card_mid)
    assert card["role"] == "assistant"
    assert card["meta"]["thread_summary"]["thread_slot"] == thread_slot


@pytest.mark.asyncio
async def test_the_close_route_validates_its_inputs(tmp_path):
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    async with _client(state) as client:
        resp = await client.post("/api/chat/threads/not-a-mid/close", json={"slot_key": slot.key})
        assert resp.status == 400
        assert (await resp.json())["code"] == "invalid_mid"
        resp = await client.post(f"/api/chat/threads/{mid}/close", json={})
        assert resp.status == 400
        assert (await resp.json())["code"] == "missing_required_fields"
        # No thread on that message: named, not invented.
        resp = await client.post(
            f"/api/chat/threads/{mid}/close",
            json={"slot_key": slot.key, "thread_slot": "chat-9"},
        )
        assert resp.status == 404
        assert (await resp.json())["code"] == "thread_not_found"


@pytest.mark.asyncio
async def test_the_open_route_resolves_inflight_to_the_turns_own_prompt(tmp_path, monkeypatch):
    """The sentinel the UI sends for a reply that is still streaming, which has no
    mid of its own. The partial text travels into the seed."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    asked = slot.append("user", "why nine?", broadcast=False)
    state.conversation_log.append(
        slot_history_key(slot), "user", "why nine?", mid=asked["meta"]["mid"]
    )
    _stream(slot, "Because three ", "are dupes")
    sent = _no_seed(monkeypatch)
    async with _client(state) as client:
        resp = await client.post(
            "/api/chat/threads/inflight/open",
            # With a note, so the seed RUNS and its text is on the wire to read.
            json={"slot_key": slot.key, "title": "dupes", "note": "which three?"},
        )
        assert resp.status == 201
        body = await resp.json()
    assert body["anchor"]["mid"] == asked["meta"]["mid"]
    assert "Because three are dupes" in sent[0]["message"]
    # The parent's stream is untouched.
    assert sum(1 for m in slot.messages if m.get("role") == "chunk") == 2


@pytest.mark.asyncio
async def test_an_inflight_open_records_in_flight_true_on_its_ledger_entry(tmp_path, monkeypatch):
    """`in_flight` is what tells a crew-log reader the anchor was RESOLVED from a
    streaming turn rather than pointed at, and the route is the only place that
    knows which happened -- by the time `open_thread` has it, it is a plain mid.
    The anchor index cannot carry the answer (its rows project to a fixed field
    set), so the entry is where a reader asks."""
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    asked = slot.append("user", "why nine?", broadcast=False)
    state.conversation_log.append(
        slot_history_key(slot), "user", "why nine?", mid=asked["meta"]["mid"]
    )
    _stream(slot, "Because three ", "are dupes")
    _no_seed(monkeypatch)
    recorded: list[dict] = []
    from kiro_crew.crew_log import emit as crew_log_emit

    monkeypatch.setattr(
        crew_log_emit,
        "on_thread_opened",
        lambda sid, **fields: recorded.append(fields),
    )
    async with _client(state) as client:
        resp = await client.post(
            "/api/chat/threads/inflight/open", json={"slot_key": slot.key, "title": "dupes"}
        )
        assert resp.status == 201
    assert [entry["in_flight"] for entry in recorded] == [True]


@pytest.mark.asyncio
async def test_an_open_on_a_real_mid_records_in_flight_false(tmp_path, monkeypatch):
    """The other half of the pair: a flag that is true for every open says nothing,
    and one that is never true says nothing either."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    recorded: list[dict] = []
    from kiro_crew.crew_log import emit as crew_log_emit

    monkeypatch.setattr(
        crew_log_emit,
        "on_thread_opened",
        lambda sid, **fields: recorded.append(fields),
    )
    async with _client(state) as client:
        resp = await client.post(
            f"/api/chat/threads/{mid}/open", json={"slot_key": slot.key, "title": "nine"}
        )
        assert resp.status == 201
    assert [entry["in_flight"] for entry in recorded] == [False]


@pytest.mark.asyncio
async def test_inflight_on_a_chat_with_nothing_to_anchor_to_is_refused(tmp_path):
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-bare-1")
    async with _client(state) as client:
        resp = await client.post("/api/chat/threads/inflight/open", json={"slot_key": slot.key})
        assert resp.status == 404
        assert (await resp.json())["code"] == "parent_not_found"


@pytest.mark.asyncio
async def test_the_open_route_validates_its_inputs(tmp_path):
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    async with _client(state) as client:
        # A mid that is not a minted row id, and not the sentinel.
        resp = await client.post("/api/chat/threads/not-a-mid/open", json={"slot_key": slot.key})
        assert resp.status == 400
        assert (await resp.json())["code"] == "invalid_mid"
        # No slot named, and no caller session to take one from.
        resp = await client.post(f"/api/chat/threads/{mid}/open", json={})
        assert resp.status == 400
        assert (await resp.json())["code"] == "missing_required_fields"
        # A slot that is not open.
        resp = await client.post(f"/api/chat/threads/{mid}/open", json={"slot_key": "chat-gone-9"})
        assert resp.status == 404
        assert (await resp.json())["code"] == "slot_not_found"
        # A message that is not in this chat.
        resp = await client.post(f"/api/chat/threads/{M('nope')}/open", json={"slot_key": slot.key})
        assert resp.status == 404
        assert (await resp.json())["code"] == "parent_not_found"


@pytest.mark.asyncio
async def test_an_app_caller_cannot_open_a_thread_on_a_slot_it_does_not_own(tmp_path):
    """App tokens see only their own slots, so a foreign slot gets the
    anti-enumeration 404 rather than a refusal that confirms it exists."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    async with _client(state, app_name="notes") as client:
        resp = await client.post(f"/api/chat/threads/{mid}/open", json={"slot_key": slot.key})
        assert resp.status == 404
        assert (await resp.json())["code"] == "slot_not_found"
    assert _anchors(state, slot) == {}


@pytest.mark.asyncio
async def test_an_app_caller_is_refused_on_a_slot_whose_transcript_was_linked(tmp_path):
    """Owning the slot is not enough on a thread route. An app may claim a name a
    later binding links -- only ``member-`` is reserved -- and the binding does not
    ask who made the slot, so ownership survives while the transcript key becomes
    the binder's. These routes address that transcript, so a linked slot is refused
    even to the app that created it."""
    state = _make_state(tmp_path)
    # A PLAIN chat key: only `member-` is reserved from app tokens, which is exactly
    # the gap -- an app can claim a name a later binding links.
    slot, mid = _chat(state, "cron-nightly-digest", mode="chat")
    slot._app = "notes"
    # The binding a cron job makes, which names a transcript the app never owned.
    slot.linked_session_key = "cron:nightly-digest"
    async with _client(state, app_name="notes") as client:
        for path in (
            f"/api/chat/threads?slot={slot.key}",
            f"/api/chat/threads/{mid}?slot={slot.key}",
        ):
            resp = await client.get(path)
            assert resp.status == 404, path
            assert (await resp.json())["code"] == "slot_not_found"
        resp = await client.post(f"/api/chat/threads/{mid}/open", json={"slot_key": slot.key})
        assert resp.status == 404
    assert _anchors(state, slot) == {}


@pytest.mark.asyncio
async def test_an_app_caller_still_reaches_its_own_unlinked_slot(tmp_path, monkeypatch):
    """The control: the refusal above is about the LINK, not about apps. Without one
    the app's own slot answers as it always did."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state, "app-notes-scratch", mode="chat")
    slot._app = "notes"
    _no_seed(monkeypatch)
    async with _client(state, app_name="notes") as client:
        resp = await client.get(f"/api/chat/threads?slot={slot.key}")
        assert resp.status == 200
        # Opening is refused for a different reason that predates this change: an
        # app-scoped caller mints no sessions at all. Named here so the refusal
        # above is not mistaken for this one.
        resp = await client.post(f"/api/chat/threads/{mid}/open", json={"slot_key": slot.key})
        assert (await resp.json())["code"] == "app_scoped_caller"


@pytest.mark.asyncio
async def test_a_close_that_names_the_wrong_thread_is_refused(tmp_path, monkeypatch):
    """The mid says which MESSAGE; a message carries a succession of threads. A
    drawer left open while this message was ended and reopened elsewhere would end
    the replacement, so the caller names the thread it believes it is ending."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    async with _client(state) as client:
        opened = await client.post(f"/api/chat/threads/{mid}/open", json={"slot_key": slot.key})
        assert opened.status == 201
        live = (await opened.json())["thread_slot"]
        stale = await client.post(
            f"/api/chat/threads/{mid}/close",
            json={"slot_key": slot.key, "thread_slot": "chat-a-thread-that-ended"},
        )
        assert stale.status == 409
        assert (await stale.json())["code"] == "already_closed"
        # Untouched: still open, and no card in the parent.
        assert _anchors(state, slot)[mid]["closed_at"] is None
        # Naming the live thread ends it.
        ok = await client.post(
            f"/api/chat/threads/{mid}/close",
            json={"slot_key": slot.key, "thread_slot": live},
        )
        assert ok.status == 200
    assert _anchors(state, slot)[mid]["closed_at"] is not None


@pytest.mark.asyncio
async def test_a_close_without_a_named_thread_is_refused(tmp_path, monkeypatch):
    """Required rather than optional: an omitted field would restore exactly the
    hole -- a close that ends whatever thread is on the message when it lands."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    async with _client(state) as client:
        resp = await client.post(f"/api/chat/threads/{mid}/close", json={"slot_key": slot.key})
        assert resp.status == 400
        assert (await resp.json())["code"] == "missing_required_fields"


@pytest.mark.asyncio
async def test_the_summary_route_answers_anchors_and_the_legacy_fold(tmp_path, monkeypatch):
    """``ThreadFooter`` keeps working unchanged while the UI worker is in flight:
    one map, one badge per message, both eras in it."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    path = _store_path(state, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "threads": {
                    M("old"): [
                        {
                            "id": "1" * 32,
                            "role": "user",
                            "content": "legacy",
                            # A real instant: the reader drops a row whose ts is not
                            # in the writer's own shape, which would take the whole
                            # legacy thread with it.
                            "ts": "2026-09-22T07:41:00+00:00",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    opened = await open_thread(
        state, Anchor("dashboard", slot.key, mid), title="live", opened_by="user", seed="s"
    )
    async with _client(state) as client:
        resp = await client.get("/api/chat/threads", params={"slot": slot.key})
        assert resp.status == 200
        threads = (await resp.json())["threads"]
    assert threads[mid]["kind"] == "session"
    assert threads[mid]["thread_slot"] == opened["thread_slot"]
    assert threads[M("old")]["kind"] == "legacy"
    assert threads[M("old")]["count"] == 1


@pytest.mark.asyncio
async def test_the_summary_route_needs_a_slot(tmp_path):
    state = _make_state(tmp_path)
    _chat(state)
    async with _client(state) as client:
        resp = await client.get("/api/chat/threads")
        assert resp.status == 400
        assert (await resp.json())["code"] == "missing_query_params"


@pytest.mark.asyncio
async def test_the_detail_route_answers_the_anchor_not_the_threads_messages(tmp_path, monkeypatch):
    """A thread's messages are its own slot's transcript, read through the
    ordinary chat endpoints -- which is the whole point of it being a real
    session. This route answers what HANGS OFF the message."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    _no_seed(monkeypatch)
    opened = await open_thread(
        state, Anchor("dashboard", slot.key, mid), title="live", opened_by="user", seed="s"
    )
    async with _client(state) as client:
        resp = await client.get(f"/api/chat/threads/{mid}", params={"slot": slot.key})
        assert resp.status == 200
        body = await resp.json()
    assert body["parent"]["mid"] == mid
    assert body["anchor"]["thread_slot"] == opened["thread_slot"]
    assert body["legacy_replies"] == []


@pytest.mark.asyncio
async def test_the_detail_route_refuses_an_unknown_message(tmp_path):
    state = _make_state(tmp_path)
    slot, _ = _chat(state)
    async with _client(state) as client:
        resp = await client.get(f"/api/chat/threads/{M('nope')}", params={"slot": slot.key})
        assert resp.status == 404
        assert (await resp.json())["code"] == "parent_not_found"
        resp = await client.get("/api/chat/threads/not-a-mid", params={"slot": slot.key})
        assert resp.status == 400
        assert (await resp.json())["code"] == "invalid_mid"


@pytest.mark.asyncio
async def test_an_unreadable_sidecar_answers_503_on_every_route(tmp_path):
    """The panel cannot show threads it cannot read, and a damaged sidecar is
    never overwritten to make the error go away."""
    state = _make_state(tmp_path)
    slot, mid = _chat(state)
    path = _store_path(state, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    async with _client(state) as client:
        resp = await client.get("/api/chat/threads", params={"slot": slot.key})
        assert resp.status == 503
        assert (await resp.json())["code"] == "threads_unavailable"
        resp = await client.get(f"/api/chat/threads/{mid}", params={"slot": slot.key})
        assert resp.status == 503
        resp = await client.post(f"/api/chat/threads/{mid}/open", json={"slot_key": slot.key})
        assert resp.status == 503
    assert path.read_text(encoding="utf-8") == "{not json"


# ── The WS frame ─────────────────────────────────────────────────────────────


def test_the_anchor_frame_announces_the_relation_not_a_message(tmp_path):
    """``chat.thread_anchor`` replaces ``chat.thread_reply``, and the rename is the
    shape of the change: a thread's turns stream on its OWN slot's frames now, so
    what a parent conversation still needs told is that a thread appeared."""
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    broadcast_thread_anchor(
        state,
        slot_key="member-radar",
        mid=M("a"),
        event="opened",
        thread_slot="chat-7",
        title="why is triage slow",
        opened_by="user",
    )
    assert len(events) == 1
    kind, payload = events[0]
    assert kind == THREAD_ANCHOR_EVENT == "chat.thread_anchor"
    assert payload["slot"] == "member-radar"
    assert payload["mid"] == M("a")
    assert payload["event"] == "opened"
    assert payload["thread_slot"] == "chat-7"
    # No run_id, role or content: this frame carries no thread message.
    assert "run_id" not in payload
    assert "content" not in payload


def test_the_anchor_frames_title_is_scrubbed(tmp_path):
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    broadcast_thread_anchor(
        state,
        slot_key="member-radar",
        mid=M("a"),
        event="opened",
        thread_slot="chat-7",
        title="key AKIAIOSFODNN7EXAMPLE",
    )
    assert "AKIAIOSFODNN7EXAMPLE" not in events[0][1]["title"]
