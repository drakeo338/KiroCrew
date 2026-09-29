"""A user Stop during an automatic compaction neither fails it nor restarts the session.

The scenario: a long-running dashboard session looks stalled, the user presses Stop
while an automatic ``/compact`` turn holds it. Without these guarantees the Stop
cancels that turn, the failure arm recycles the process, and the dashboard answers
"Compaction didn't succeed, so the session was restarted instead" for a Stop the
user pressed. Four things pin the behaviour:

1. ``stop_turn`` DECLINES a cooperative Stop while the key is compacting and does
   not record it as a Stop the turn saw; a force stop still goes through.
2. When a ``/compact`` turn IS ended by a Stop (a force stop, or the race the
   pre-check cannot close), the compaction settles as ``cancelled`` -- cooldown armed,
   provider NOT shut down, notice says so -- instead of recycling.
3. The compacting set is observable: an observer is told on enter and leave, and the
   dashboard slot payload carries ``compacting`` so the composer can show it.
4. The restart notices name the transcript excerpt the successor starts from.

Fakes only: a mock provider whose ``/compact`` blocks until released or raises, no
real harness.
"""

from __future__ import annotations

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.config import KiroCrewConfig
from kiro_crew.session import SessionManager
from kiro_crew.session_compaction import (
    COMPACT_OUTCOME_CANCELLED,
    COMPACT_OUTCOME_RECYCLED,
)

KEY = "dashboard:chat-14841"


class _Compact:
    """A ``/compact`` turn the test controls: it blocks until released or is failed."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.fail_with: BaseException | None = None
        self.started = asyncio.Event()

    async def stream(self, _command: str):
        self.started.set()
        await self.release.wait()
        if self.fail_with is not None:
            raise self.fail_with
        if False:  # pragma: no cover - makes this an async generator
            yield None


def _factory(compact: _Compact, order: list[str]):
    def factory(session_key=None, agent=None, channel_id=None, **kwargs):
        m = AsyncMock()
        m.cwd = ""
        m.disown_work_dir = MagicMock()
        m.memory_mode = "persistent"
        m.is_process_alive = lambda: True
        m.context_usage_pct = lambda: 90.0
        m.context_usage_unknown = lambda: False
        m.context_window_tokens = lambda: 0
        m.has_active_turn = lambda: True
        m.runtime_info = lambda: (None, None)
        m.stream_command = MagicMock(side_effect=compact.stream)
        m.wait_for_compaction = AsyncMock(return_value={"type": "failed"})

        # A cooperative cancel is what a soft Stop does to a live turn; here it
        # ends the /compact turn the way the harness would: the stream raises.
        async def _cancel(*, wait_ack_timeout: float = 0.0):
            compact.fail_with = RuntimeError("compaction reported no result")
            compact.release.set()
            return "acked"

        m.cancel = AsyncMock(side_effect=_cancel)
        m.shutdown = AsyncMock(side_effect=lambda: order.append("shutdown"))
        return m

    return factory


async def _setup():
    order: list[str] = []
    compact = _Compact()
    mgr = SessionManager(KiroCrewConfig(), provider_factory=_factory(compact, order))
    await mgr.get_or_create(KEY)
    key = mgr._fold_key(KEY)
    mgr.release(key)
    mgr._compaction._deps = dataclasses.replace(
        mgr._compaction._deps,
        compact_wait_timeout_secs=lambda: 5.0,
        compact_result_wait_secs=lambda _elapsed: 0.05,
        compact_failure_cooldown_secs=123.0,
    )
    notices: list[tuple[bool, str]] = []

    async def _cb(key, pct, *, success, outcome="compacted"):
        notices.append((success, outcome))

    mgr.set_compact_callback(_cb)
    return mgr, key, compact, order, notices


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


# -- 1. a cooperative Stop is declined while compacting --


@pytest.mark.asyncio
async def test_a_cooperative_stop_during_compaction_is_declined_and_not_recorded():
    mgr, key, compact, order, notices = await _setup()
    session = mgr._sessions[key]
    before = mgr.stop_generation(key)

    task = asyncio.ensure_future(mgr._compact_in_place(key, session, 90.0))
    await asyncio.wait_for(compact.started.wait(), timeout=2)
    # The trigger paths add the key; _compact_in_place alone does not. Mirror
    # the state a real threshold trigger leaves.
    mgr._compacting.add(key)
    try:
        assert mgr.is_compacting(KEY) is True
        outcome = await mgr.stop_turn(KEY, force=False)
        assert outcome == "compacting"
        # Not recorded: a declined Stop is not a Stop the turn saw, and recording
        # it would make the compaction read its own later failure as cancelled.
        assert mgr.stop_generation(key) == before
        session.provider.cancel.assert_not_called()
        assert not task.done()
    finally:
        mgr._compacting.discard(key)
        compact.release.set()
        await asyncio.wait_for(task, timeout=5)
    # The compaction ran to its own (failed) end and recycled -- the ordinary
    # failure arm, untouched by the declined Stop.
    assert order == ["shutdown"]
    assert notices == [(True, COMPACT_OUTCOME_RECYCLED)]
    await mgr.close_all()


@pytest.mark.asyncio
async def test_a_force_stop_during_compaction_is_never_declined():
    """The escape hatch stays open: ``force`` is the user's second press."""
    mgr, key, compact, order, notices = await _setup()
    session = mgr._sessions[key]
    task = asyncio.ensure_future(mgr._compact_in_place(key, session, 90.0))
    await asyncio.wait_for(compact.started.wait(), timeout=2)
    mgr._compacting.add(key)
    try:
        outcome = await mgr.stop_turn(KEY, force=True)
    finally:
        mgr._compacting.discard(key)
        compact.fail_with = RuntimeError("compaction reported no result")
        compact.release.set()
    assert outcome == "hard"
    result = await asyncio.wait_for(task, timeout=5)
    # The force stop reset the session; the compaction that was running on it
    # settles as CANCELLED rather than recycling a provider the reset already
    # replaced and telling the user compaction "didn't succeed".
    assert result == "cancelled"
    assert notices[-1] == (False, COMPACT_OUTCOME_CANCELLED)
    await mgr.close_all()


@pytest.mark.asyncio
async def test_a_force_stop_hands_the_permit_to_the_waiter_and_the_compaction_does_not_reclaim_it():
    """The hard Stop pops the session and releases the compaction's permit to wake
    a parked claimant. That claimant now OWNS the permit; the compaction's own
    cleanup must not release it a second time under the claimant's feet."""
    mgr, key, compact, order, notices = await _setup()
    session = mgr._sessions[key]
    task = asyncio.ensure_future(mgr._compact_in_place(key, session, 90.0))
    await asyncio.wait_for(compact.started.wait(), timeout=2)

    # A claimant parked on the held permit, as ``_reacquire_and_validate`` does.
    async def _claimant():
        await session.semaphore.acquire()
        await asyncio.sleep(0.05)  # holds it while the compaction's finally runs
        session.semaphore.release()  # must not raise
        return "released-cleanly"

    claimant = asyncio.ensure_future(_claimant())
    await _settle()
    assert not claimant.done()

    mgr._compacting.add(key)
    try:
        assert await mgr.stop_turn(KEY, force=True) == "hard"
    finally:
        mgr._compacting.discard(key)
        compact.fail_with = RuntimeError("compaction reported no result")
        compact.release.set()
    assert await asyncio.wait_for(task, timeout=5) == "cancelled"
    assert await asyncio.wait_for(claimant, timeout=5) == "released-cleanly"
    await mgr.close_all()


@pytest.mark.asyncio
async def test_stop_turn_is_unchanged_when_nothing_is_compacting():
    mgr, key, compact, order, notices = await _setup()
    assert mgr.is_compacting(KEY) is False
    # No compaction and a mock provider that acks: the ordinary soft path.
    assert await mgr.stop_turn(KEY, force=False) == "soft"
    assert mgr.stop_generation(key) == 1
    await mgr.close_all()


# -- 2. a Stop that ends the /compact turn settles as cancelled, not recycled --


@pytest.mark.asyncio
async def test_a_stop_that_ends_the_compact_turn_does_not_recycle():
    """The race the pre-check cannot close: the Stop lands on the /compact turn.

    Driven by noting the Stop directly, which is what every channel stop path
    that cancels the provider itself does, and then failing the turn the way the
    harness reports a cancelled prompt.
    """
    mgr, key, compact, order, notices = await _setup()
    session = mgr._sessions[key]
    task = asyncio.ensure_future(mgr._compact_in_place(key, session, 90.0))
    await asyncio.wait_for(compact.started.wait(), timeout=2)

    assert mgr.note_stop(KEY) is True
    compact.fail_with = RuntimeError("compaction reported no result")
    compact.release.set()

    assert await asyncio.wait_for(task, timeout=5) == "cancelled"
    # No recycle: the provider is still the session's provider and was not shut down.
    assert order == []
    assert mgr._sessions[key] is session
    assert notices == [(False, COMPACT_OUTCOME_CANCELLED)]
    # The cooldown is armed so the next threshold reading retries later rather
    # than immediately re-entering the compaction the user just stopped.
    assert key in mgr._compact_cooldown_until
    await mgr.close_all()


@pytest.mark.asyncio
async def test_a_stop_before_the_semaphore_is_held_is_not_this_compactions_cancel():
    """A Stop that ended the PREVIOUS turn must not be read as cancelling this compaction."""
    mgr, key, compact, order, notices = await _setup()
    session = mgr._sessions[key]
    # The Stop happened earlier, on some other turn.
    assert mgr.note_stop(KEY) is True
    task = asyncio.ensure_future(mgr._compact_in_place(key, session, 90.0))
    await asyncio.wait_for(compact.started.wait(), timeout=2)
    compact.fail_with = RuntimeError("compaction reported no result")
    compact.release.set()
    # Genuine failure: the ordinary recycle arm.
    assert await asyncio.wait_for(task, timeout=5) == "recycled"
    assert order == ["shutdown"]
    assert notices == [(True, COMPACT_OUTCOME_RECYCLED)]
    await mgr.close_all()


# -- 3. the compacting set is observable --


@pytest.mark.asyncio
async def test_the_compacting_observer_sees_enter_and_leave_once_each():
    mgr, key, compact, order, notices = await _setup()
    seen: list[tuple[str, bool]] = []
    mgr.set_compacting_callback(lambda k, on: seen.append((k, on)))
    # A failing observer must not fail the compaction.
    session = mgr._sessions[key]
    provider = session.provider
    decline = mgr._compaction._trigger_compaction(key, "test", 90.0, provider)
    assert decline is None, decline
    await asyncio.wait_for(compact.started.wait(), timeout=2)
    assert seen == [(key, True)]
    assert mgr.is_compacting(KEY) is True
    compact.fail_with = RuntimeError("compaction reported no result")
    compact.release.set()
    for _ in range(200):
        if not mgr.is_compacting(KEY):
            break
        await asyncio.sleep(0.01)
    assert seen == [(key, True), (key, False)]
    await mgr.close_all()


@pytest.mark.asyncio
async def test_a_raising_observer_does_not_fail_the_compaction():
    mgr, key, compact, order, notices = await _setup()

    def _boom(_k, _on):
        raise RuntimeError("observer broke")

    mgr.set_compacting_callback(_boom)
    session = mgr._sessions[key]
    decline = mgr._compaction._trigger_compaction(key, "test", 90.0, session.provider)
    assert decline is None
    await asyncio.wait_for(compact.started.wait(), timeout=2)
    assert mgr.is_compacting(KEY) is True
    compact.release.set()  # completes with no status -> failed -> recycled
    compact.fail_with = RuntimeError("compaction reported no result")
    for _ in range(200):
        if not mgr.is_compacting(KEY):
            break
        await asyncio.sleep(0.01)
    assert mgr.is_compacting(KEY) is False
    assert notices == [(True, COMPACT_OUTCOME_RECYCLED)]
    await mgr.close_all()


def _dashboard_state(tmp_path):
    import sys

    sys.path.insert(0, "test")
    from chat_test_helpers import _make_state

    return _make_state(tmp_path)


def test_the_slot_payload_carries_compacting_beside_running(tmp_path):
    """The composer reads ``compacting`` off the slot, separate from ``running``."""
    state = _dashboard_state(tmp_path)
    slot = state.get_or_create_slot("chat-14841")
    assert slot.to_dict()["compacting"] is False
    slot._compacting = True
    payload = slot.to_dict()
    assert payload["compacting"] is True
    assert payload["running"] is False, "a compaction is not a dashboard turn"


def test_the_dashboard_stop_is_declined_while_the_session_compacts(tmp_path, monkeypatch):
    """The route-level pre-check: no cancel is sent and the press still leaves a card."""
    from unittest.mock import AsyncMock, MagicMock

    from kiro_crew.dashboard.chat_handlers import stop_slot_turn

    state = _dashboard_state(tmp_path)
    slot = state.get_or_create_slot("chat-14841")
    task = MagicMock()
    task.done.return_value = False
    slot.task = task
    state.sessions.stop_turn = AsyncMock(return_value="soft")
    state.sessions.is_compacting = MagicMock(return_value=True)
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())

    reply = asyncio.run(stop_slot_turn(state, slot))

    assert reply == {"ok": True, "info": "compacting", "compacting": True}
    state.sessions.stop_turn.assert_not_awaited()
    # ``_stop_state`` stays idle -- the queue drain reads that machine as "a stop
    # is in progress" and would persist a false "Session reset" row -- while
    # the separate decline marker arms the next press as the force stop.
    assert slot._stop_state == "idle"
    assert slot._stop_declined_at > 0.0
    assert slot.to_dict()["stop_declined"] is True
    cards = [m for m in slot.messages if '"kind": "stop_event"' in (m.get("cls") or "")]
    assert len(cards) == 1
    assert '"state": "stop_declined_compacting"' in cards[0]["cls"]


def test_a_mock_shaped_manager_does_not_decline_every_stop(tmp_path, monkeypatch):
    """The probe is ``is True``: a truthy Mock answer must read as not compacting."""
    from unittest.mock import AsyncMock, MagicMock

    from kiro_crew.dashboard.chat_handlers import stop_slot_turn

    state = _dashboard_state(tmp_path)
    slot = state.get_or_create_slot("chat-14841")
    task = MagicMock()
    task.done.return_value = False
    slot.task = task
    state.sessions.stop_turn = AsyncMock(return_value="soft")
    assert callable(state.sessions.is_compacting)  # a bare MagicMock attribute
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())

    asyncio.run(stop_slot_turn(state, slot))

    state.sessions.stop_turn.assert_awaited_once()


def test_the_race_outcome_settles_the_card_and_undoes_the_soft_stop(tmp_path, monkeypatch):
    """``stop_turn`` answering ``compacting`` after the pre-check passed."""
    from unittest.mock import AsyncMock, MagicMock

    from kiro_crew.dashboard.chat_handlers import stop_slot_turn

    state = _dashboard_state(tmp_path)
    slot = state.get_or_create_slot("chat-14841")
    task = MagicMock()
    task.done.return_value = False
    slot.task = task
    state.sessions.is_compacting = MagicMock(return_value=False)
    state.sessions.stop_turn = AsyncMock(return_value="compacting")
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())

    reply = asyncio.run(stop_slot_turn(state, slot))

    assert reply["compacting"] is True
    assert slot._stop_state == "idle"
    assert slot._stop_declined_at > 0.0
    assert slot._stop_event_id is None
    cards = [m for m in slot.messages if '"kind": "stop_event"' in (m.get("cls") or "")]
    assert len(cards) == 1
    assert '"state": "stop_declined_compacting"' in cards[0]["cls"]


def test_a_second_press_during_compaction_escalates_to_the_force_stop(tmp_path, monkeypatch):
    """The escape hatch: the decline arms ``soft_pending``, so press #2 hard-stops."""
    from unittest.mock import AsyncMock, MagicMock

    from kiro_crew.dashboard.chat_handlers import stop_slot_turn

    state = _dashboard_state(tmp_path)
    slot = state.get_or_create_slot("chat-14841")
    task = MagicMock()
    task.done.return_value = False
    slot.task = task
    state.sessions.is_compacting = MagicMock(return_value=True)
    state.sessions.stop_turn = AsyncMock(return_value="hard")
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())

    asyncio.run(stop_slot_turn(state, slot))
    state.sessions.stop_turn.assert_not_awaited()
    assert slot._stop_state == "idle", "a declined Stop is not a stop in progress"
    asyncio.run(stop_slot_turn(state, slot))

    state.sessions.stop_turn.assert_awaited_once()
    assert state.sessions.stop_turn.await_args.kwargs["force"] is True
    assert slot._stop_declined_at == 0.0, "the marker is consumed by the press it armed"


def test_a_stale_decline_does_not_turn_a_later_first_press_into_a_force_stop(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from kiro_crew.dashboard.chat_handlers import stop_slot_turn
    from kiro_crew.dashboard.slot_projection import STOP_DECLINED_ESCALATION_SECS

    state = _dashboard_state(tmp_path)
    slot = state.get_or_create_slot("chat-14841")
    task = MagicMock()
    task.done.return_value = False
    slot.task = task
    state.sessions.is_compacting = MagicMock(return_value=False)
    state.sessions.stop_turn = AsyncMock(return_value="soft")
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())
    import time as _time

    slot._stop_declined_at = _time.monotonic() - STOP_DECLINED_ESCALATION_SECS - 1
    assert slot.to_dict()["stop_declined"] is False

    asyncio.run(stop_slot_turn(state, slot))

    assert state.sessions.stop_turn.await_args.kwargs["force"] is False


def test_the_shared_channel_stop_keeps_the_queue_while_compacting():
    """Discord/Telegram/Teams/Webex stop through ``stop_running_turn``, which cancels
    the provider itself and clears the queue; a declined Stop must do neither."""
    from unittest.mock import AsyncMock, MagicMock

    from kiro_crew.messaging.commands import STOP_REPLY_COMPACTING, stop_running_turn

    sessions = MagicMock()
    sessions.is_compacting = MagicMock(return_value=True)
    sessions.is_busy = MagicMock(return_value=True)
    sessions.get_provider = MagicMock(return_value=MagicMock(cancel=AsyncMock()))
    queue = MagicMock()
    queue.lock = asyncio.Lock()
    queue.finish_cancelled_locked = AsyncMock()

    reply = asyncio.run(
        stop_running_turn(
            sessions, "telegram:1", queue=queue, surface=MagicMock(label="telegram"), owner="u1"
        )
    )

    assert reply == STOP_REPLY_COMPACTING
    sessions.note_stop.assert_not_called()
    sessions.clear_queue.assert_not_called()
    sessions.get_provider.return_value.cancel.assert_not_awaited()
    queue.finish_cancelled_locked.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_claude_arm_ignores_a_stop_that_ended_the_previous_turn():
    """The Claude compaction waits for the semaphore; a Stop recorded BEFORE it is
    held ended that earlier turn and must settle a genuine failure as failed, not
    cancelled."""
    mgr, key, compact, order, notices = await _setup()
    session = mgr._sessions[key]
    session.provider.compact = AsyncMock(side_effect=RuntimeError("provider failed"))
    mgr._compaction._deps = dataclasses.replace(
        mgr._compaction._deps, is_claude_backend=lambda _p: True
    )
    # A turn holds the session; the compaction queues behind it. The Stop lands
    # on THAT turn, after the compaction task started but before it holds the
    # permit -- the window a counter read before the wait mistakes for its own.
    await session.semaphore.acquire()
    task = asyncio.ensure_future(mgr._compaction._compact_session(key, 90.0))
    await _settle()
    assert mgr.note_stop(KEY) is True
    session.semaphore.release()
    result = await asyncio.wait_for(task, timeout=5)
    assert result == "failed"
    assert notices == [(False, "compacted")]
    await mgr.close_all()


@pytest.mark.asyncio
async def test_the_claude_arm_settles_cancelled_when_the_stop_lands_on_its_turn():
    mgr, key, compact, order, notices = await _setup()
    session = mgr._sessions[key]

    async def _compact_then_stopped():
        mgr.note_stop(KEY)
        raise RuntimeError("cancelled by the harness")

    session.provider.compact = AsyncMock(side_effect=_compact_then_stopped)
    mgr._compaction._deps = dataclasses.replace(
        mgr._compaction._deps, is_claude_backend=lambda _p: True
    )
    result = await mgr._compaction._compact_session(key, 90.0)
    assert result == "cancelled"
    assert notices == [(False, COMPACT_OUTCOME_CANCELLED)]
    await mgr.close_all()


def _slack_orch():
    """The Slack orchestrator double the events suite uses, with a live !stop target."""
    import sys
    from unittest.mock import AsyncMock, MagicMock

    sys.path.insert(0, "test")
    from test_slack_events_coverage import _make_orch

    orch = _make_orch()
    orch.sessions.has_session = MagicMock(return_value=True)
    orch.sessions.get_session_for_thread = MagicMock(return_value=None)
    orch.sessions.note_stop = MagicMock(return_value=True)
    orch.sessions.clear_queue = MagicMock()
    task = MagicMock()
    task.done.return_value = False
    orch._session_tasks = {"100.0": task}
    orch._pending_queue = {"100.0": [("ts", "text", {"paths": []})]}
    orch.slack.post_ephemeral = AsyncMock()
    return orch, task


def _run_slack_stop(orch):
    from unittest.mock import patch

    from kiro_crew.slack import events as ev

    with patch("kiro_crew.slack.events.is_allowed_user", return_value=True):
        with patch("kiro_crew.slack.events.is_owner", return_value=True):
            with patch("kiro_crew.slack.events.unlink_queued_temp_paths") as unlink:
                from test_slack_events_coverage import _event

                asyncio.run(ev._route_message(orch, _event(text="!stop"), ev.SeenCache()))
    return unlink


def test_slack_stop_declined_by_the_precheck_touches_nothing():
    from unittest.mock import AsyncMock, MagicMock

    orch, task = _slack_orch()
    orch.sessions.is_compacting = MagicMock(return_value=True)
    orch.sessions.stop_turn = AsyncMock()
    unlink = _run_slack_stop(orch)
    orch.sessions.stop_turn.assert_not_awaited()
    orch.sessions.note_stop.assert_not_called()
    orch.sessions.clear_queue.assert_not_called()
    unlink.assert_not_called()
    assert orch._session_tasks == {"100.0": task}
    assert "100.0" in orch._pending_queue
    task.cancel.assert_not_called()


def test_slack_stop_declined_by_stop_turn_keeps_queue_pending_files_and_task():
    """The race the pre-check cannot close: a compaction commits during the
    ephemeral post, and ``stop_turn`` declines. Nothing queued may be lost."""
    from unittest.mock import AsyncMock, MagicMock

    orch, task = _slack_orch()
    orch.sessions.is_compacting = MagicMock(return_value=False)
    orch.sessions.stop_turn = AsyncMock(return_value="compacting")
    unlink = _run_slack_stop(orch)
    orch.sessions.stop_turn.assert_awaited_once()
    orch.sessions.clear_queue.assert_not_called()
    unlink.assert_not_called()
    assert orch._session_tasks == {"100.0": task}
    assert "100.0" in orch._pending_queue
    task.cancel.assert_not_called()


def test_slack_stop_that_goes_through_still_clears_and_cancels():
    """The control: an ordinary soft stop keeps the destructive half."""
    from unittest.mock import AsyncMock, MagicMock

    orch, task = _slack_orch()
    orch.sessions.is_compacting = MagicMock(return_value=False)
    orch.sessions.stop_turn = AsyncMock(return_value="soft")
    unlink = _run_slack_stop(orch)
    orch.sessions.clear_queue.assert_called_once_with("100.0")
    unlink.assert_called_once()
    assert orch._session_tasks == {}
    assert "100.0" not in orch._pending_queue
    task.cancel.assert_called_once()


def _interrupt_state():
    """The interrupt route's state double, as ``test_chat_slot_interrupt`` builds it."""
    from unittest.mock import AsyncMock, MagicMock

    from kiro_crew.dashboard.state import DashboardState, _ChatSlot

    slot = _ChatSlot("test")
    task = MagicMock()
    task.done.return_value = False
    slot.task = task
    slot.queue_append("msg")
    slot._auto_run = True
    fut = asyncio.get_event_loop_policy().new_event_loop().create_future()
    slot._approval_futures["req-1"] = fut
    state = MagicMock(spec=DashboardState)
    state._slots = {"test": slot}
    state.push_slots_update = MagicMock()
    state.sessions = MagicMock()
    state.sessions.stop_turn = AsyncMock(return_value="soft")
    state.broadcast_ws = MagicMock()
    return state, slot, fut


async def _post_interrupt(state):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from kiro_crew.dashboard.chat import api_chat_slot_interrupt

    app = web.Application()
    app["state"] = state
    app.router.add_post("/api/chat/slots/{slot}/interrupt", api_chat_slot_interrupt)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/chat/slots/test/interrupt", json={})
        return resp.status, await resp.json()


@pytest.mark.asyncio
async def test_interrupt_declined_by_the_precheck_leaves_the_turn_untouched(monkeypatch):
    """Auto-run stays on, pending approvals stay pending, no cancel is sent."""
    from unittest.mock import MagicMock

    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())
    state, slot, fut = _interrupt_state()
    state.sessions.is_compacting = MagicMock(return_value=True)

    status, data = await _post_interrupt(state)

    assert status == 200 and data["outcome"] == "compacting"
    state.sessions.stop_turn.assert_not_awaited()
    assert slot._auto_run is True
    assert not fut.done(), "a declined interrupt must not reject the turn's approvals"
    assert slot._stop_state == "idle"
    assert slot._stop_declined_at > 0.0
    cards = [m for m in slot.messages if '"kind": "stop_event"' in (m.get("cls") or "")]
    assert len(cards) == 1 and '"state": "stop_declined_compacting"' in cards[0]["cls"]


@pytest.mark.asyncio
async def test_interrupt_race_outcome_restores_auto_run(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())
    state, slot, _fut = _interrupt_state()
    state.sessions.is_compacting = MagicMock(return_value=False)
    state.sessions.stop_turn = AsyncMock(return_value="compacting")

    status, data = await _post_interrupt(state)

    assert status == 200 and data["outcome"] == "compacting"
    assert slot._auto_run is True
    assert slot._stop_state == "idle"
    assert slot._stop_declined_at > 0.0
    assert slot._stop_event_id is None


# -- 4. the notices --


def test_the_cancelled_notice_names_a_stop_and_no_restart():
    from kiro_crew.dashboard.chat_compaction_notice import notice_text
    from kiro_crew.dashboard.state import (
        _AUTO_COMPACT_CANCELLED_NOTICE,
        _AUTO_COMPACT_FAILED_NOTICE,
        _AUTO_RECYCLE_NOTICE,
    )

    dashboard = _AUTO_COMPACT_CANCELLED_NOTICE.format(pct=87)
    assert "Stop" in dashboard
    assert "87% of the context limit" in dashboard, "the percentage names its referent"
    assert "kept running" in dashboard
    assert "restarted" not in dashboard, "the sibling restart notice owns that verb"
    assert dashboard != _AUTO_COMPACT_FAILED_NOTICE.format(pct=87)
    assert dashboard != _AUTO_RECYCLE_NOTICE.format(pct=87)

    channel = notice_text("slack", 87.0, success=False, outcome=COMPACT_OUTCOME_CANCELLED)
    assert "stop" in channel.lower()
    assert "87% of the context limit" in channel
    assert "restarted" not in channel
    assert "`!compact`" in channel
    assert channel != notice_text("slack", 87.0, success=False, outcome="compacted")
