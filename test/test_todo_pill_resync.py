"""The checklist pill after the native conversation restarts, and a person's tick.

kiro-cli keeps the ``todo_list`` tool's state inside ONE native conversation.
Kiro Crew replaces that conversation on an agent switch, a failed
``session/load``, a poisoned-conversation discard and ``/clear``, while the
pill's slot snapshot survives all of them. The agent then holds an empty list
under a pill that shows the old one, and its next ``complete`` fails with
"Task N not found" (reproduced against kiro-cli 2.25.0), which it reports to
the person as "I cannot update the checklist".

Two repairs, both covered here:

* ``_ChatSlot.todo_recovery_prompt`` -- the block the runner prepends to the
  first prompt of a fresh native session, telling the agent to rebuild the
  list with its own tool so the two copies agree again.
* ``PATCH /api/chat/slots/{slot}/todo`` -- a person ticking one row of the pill
  directly, which was impossible before (the tool result was the only writer).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.dashboard.chat_todo import api_chat_slot_todo
from kiro_crew.dashboard.state import DashboardState, _ChatSlot
from kiro_crew.dashboard.token_auth import MEMBER_CHAT_PRINCIPAL_KEY


def _slot(key: str = "s1", tasks: list[tuple[str, bool]] | None = None) -> _ChatSlot:
    slot = _ChatSlot.__new__(_ChatSlot)
    slot.key = key
    slot._todo = None
    slot._app = ""
    if tasks is not None:
        slot.set_todo(
            {
                "description": "Config workflow",
                "tasks": [
                    {"id": str(i + 1), "text": text, "completed": done}
                    for i, (text, done) in enumerate(tasks)
                ],
            }
        )
    return slot


class TestRecoveryPrompt:
    def test_empty_when_no_list(self) -> None:
        assert _slot().todo_recovery_prompt() == ""

    def test_empty_when_list_has_no_tasks(self) -> None:
        assert _slot(tasks=[]).todo_recovery_prompt() == ""

    def test_carries_every_task_with_its_state_in_order(self) -> None:
        text = _slot(
            tasks=[("read runbook", True), ("RCA SC", False)]
        ).todo_recovery_prompt()
        assert text.startswith("[Task checklist — automatic recovery]")
        assert "Description: Config workflow" in text
        assert text.index("1. [x] read runbook") < text.index("2. [ ] RCA SC")
        assert text.rstrip().endswith("[End task checklist]")

    def test_names_the_tool_and_both_commands(self) -> None:
        """The agent must recreate the list, not just be told about it."""
        text = _slot(tasks=[("a", True)]).todo_recovery_prompt()
        assert "todo_list" in text
        assert "`create`" in text
        assert "`complete`" in text


class TestSetTodoTaskCompleted:
    def test_flips_by_id_and_reports_change(self) -> None:
        slot = _slot(tasks=[("a", False), ("b", False)])
        assert slot.set_todo_task_completed("2", True) is True
        payload = slot.todo_payload()
        assert payload is not None
        assert payload["completed"] == 1
        assert payload["current"] == "a"

    def test_same_state_reports_no_change(self) -> None:
        """Gates the broadcast the way ``set_todo`` does."""
        slot = _slot(tasks=[("a", True)])
        assert slot.set_todo_task_completed("1", True) is False

    def test_unknown_id_and_absent_list_change_nothing(self) -> None:
        assert _slot().set_todo_task_completed("1", True) is False
        slot = _slot(tasks=[("a", False)])
        assert slot.set_todo_task_completed("9", True) is False
        assert slot.todo_payload()["completed"] == 0  # type: ignore[index]


# ── the route ────────────────────────────────────────────────────────────────


def _state(*slots: _ChatSlot) -> DashboardState:
    state = MagicMock(spec=DashboardState)
    state._slots = {s.key: s for s in slots}
    state.broadcast_ws = MagicMock()
    return state


def _app(
    state: DashboardState, *, declared_app: str = "", member: str = ""
) -> web.Application:
    app = web.Application()
    app["state"] = state

    @web.middleware
    async def _claims(request: web.Request, handler):
        request["app"] = declared_app
        if member:
            request[MEMBER_CHAT_PRINCIPAL_KEY] = member
        return await handler(request)

    app.middlewares.append(_claims)
    app.router.add_patch("/api/chat/slots/{slot}/todo", api_chat_slot_todo)
    return app


async def _patch(app: web.Application, slot: str, body: Any) -> tuple[int, Any]:
    async with TestClient(TestServer(app)) as client:
        resp = await client.patch(
            f"/api/chat/slots/{slot}/todo",
            json=body,
            headers={"X-Session-Key": "dashboard:s1"},
        )
        return resp.status, await resp.json()


@pytest.mark.asyncio
async def test_tick_writes_the_slot_and_broadcasts_the_same_delta_the_tool_does() -> (
    None
):
    slot = _slot(tasks=[("a", True), ("b", False)])
    state = _state(slot)
    status, body = await _patch(_app(state), "s1", {"id": "2", "completed": True})
    assert status == 200
    assert body["todo"]["completed"] == 2
    state.broadcast_ws.assert_called_once_with(
        "todo_update", {"slot": "s1", "todo": slot.todo_payload()}
    )
    # The tick is what the next fresh native session rebuilds from.
    assert "2. [x] b" in slot.todo_recovery_prompt()


@pytest.mark.asyncio
async def test_untick_is_the_same_write_in_reverse() -> None:
    slot = _slot(tasks=[("a", True)])
    status, body = await _patch(
        _app(_state(slot)), "s1", {"id": "1", "completed": False}
    )
    assert status == 200
    assert body["todo"]["completed"] == 0


@pytest.mark.asyncio
async def test_no_op_tick_answers_ok_without_a_broadcast() -> None:
    slot = _slot(tasks=[("a", True)])
    state = _state(slot)
    status, _ = await _patch(_app(state), "s1", {"id": "1", "completed": True})
    assert status == 200
    state.broadcast_ws.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_task_is_404() -> None:
    status, _ = await _patch(
        _app(_state(_slot(tasks=[("a", False)]))), "s1", {"id": "7", "completed": True}
    )
    assert status == 404


@pytest.mark.asyncio
async def test_slot_without_a_checklist_is_404() -> None:
    status, _ = await _patch(
        _app(_state(_slot())), "s1", {"id": "1", "completed": True}
    )
    assert status == 404


@pytest.mark.asyncio
async def test_missing_slot_is_404() -> None:
    status, _ = await _patch(_app(_state()), "nope", {"id": "1", "completed": True})
    assert status == 404


@pytest.mark.asyncio
async def test_malformed_body_is_400() -> None:
    app = _app(_state(_slot(tasks=[("a", False)])))
    for body in ({"id": "1"}, {"completed": True}, {"id": "1", "completed": "yes"}):
        status, _ = await _patch(app, "s1", body)
        assert status == 400, body


@pytest.mark.asyncio
async def test_app_caller_is_refused_with_the_indistinguishable_404() -> None:
    """An app agent's writer is its own todo_list tool; a person ticks the pill."""
    slot = _slot(tasks=[("a", False)])
    state = _state(slot)
    status, body = await _patch(
        _app(state, declared_app="some-app"), "s1", {"id": "1", "completed": True}
    )
    assert status == 404
    assert body.get("code") == "slot_not_found"
    assert slot.todo_payload()["completed"] == 0  # type: ignore[index]
    state.broadcast_ws.assert_not_called()


# ── the runner prepends the recovery block on a cold start only ─────────────


def _runner_state(tmp_path, monkeypatch, *, is_new: bool, resumed: bool):
    """The real ``_run_chat`` with the provider and the builder mocked.

    The same seam ``test_chat_runner_folder_steering`` drives; here the assertion
    is on the prompt the provider's ``stream`` receives, which is the prompt kiro
    receives.
    """
    from unittest.mock import AsyncMock

    from chat_test_helpers import _make_state

    from kiro_crew.context import ContextBuilder
    from kiro_crew.dashboard import chat_runner
    from kiro_crew.memory import MemoryStore
    from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent
    from kiro_crew.skills import SkillsLoader

    builder = ContextBuilder(
        memory=MemoryStore(workspace=tmp_path / "workspace"),
        skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
    )
    state = _make_state(tmp_path, context_builder=builder)
    state.context_builder.build_message = MagicMock(return_value=("BUILT", None))
    state.context_builder.ensure_store = AsyncMock(return_value=object())
    provider = MagicMock()
    sent: list[str] = []

    async def stream(message, *args, **kwargs):
        sent.append(message)
        yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="Done.")
        yield LLMEvent(kind=EVENT_COMPLETE)

    provider.stream = stream
    provider.client = None  # no native ``resumed`` attribute to read
    state.sessions.get_or_create = AsyncMock(return_value=(provider, is_new, resumed))
    state.sessions.consume_replay_suppression = MagicMock(return_value=False)
    state.sessions.consume_needs_reinjection = MagicMock(return_value=False)
    state.sessions.provider_switch_replay_pending = MagicMock(return_value=False)
    state.sessions.record_failure = AsyncMock()
    monkeypatch.setattr(chat_runner, "title_then_refresh", AsyncMock())
    monkeypatch.setattr(chat_runner, "generate_session_summary", AsyncMock())
    return state, sent


async def _runner_turn(state, slot, message="carry on") -> None:
    import asyncio

    from chat_test_helpers import drain_background_tasks

    from kiro_crew.dashboard import chat_runner

    slot.append("user", message)
    await asyncio.wait_for(chat_runner._run_chat(state, slot, message), 30)
    await asyncio.wait_for(drain_background_tasks(state), 10)


def _seed_pill(slot) -> None:
    slot.set_todo(
        {
            "description": "Config workflow",
            "tasks": [
                {"id": "1", "text": "read runbook", "completed": True},
                {"id": "2", "text": "RCA SC", "completed": False},
            ],
        }
    )


@pytest.mark.asyncio
async def test_cold_start_with_a_pill_prepends_the_recovery_block(
    tmp_path, monkeypatch
) -> None:
    """A fresh native session holds no list, so the prompt tells the agent to rebuild it."""
    state, sent = _runner_state(tmp_path, monkeypatch, is_new=True, resumed=False)
    slot = state.get_or_create_slot("pill-chat")
    _seed_pill(slot)
    await _runner_turn(state, slot)
    assert len(sent) == 1
    prompt = sent[0]
    assert "[Task checklist" in prompt
    assert "1. [x] read runbook" in prompt and "2. [ ] RCA SC" in prompt
    # A prepend: the builder's own prompt stays the trusted tail, byte for byte.
    assert prompt.endswith("BUILT")


@pytest.mark.asyncio
async def test_resumed_session_gets_no_recovery_block(tmp_path, monkeypatch) -> None:
    """``session/load`` brought the native list back; nothing to rebuild."""
    state, sent = _runner_state(tmp_path, monkeypatch, is_new=True, resumed=True)
    slot = state.get_or_create_slot("pill-chat")
    _seed_pill(slot)
    await _runner_turn(state, slot)
    assert sent and "[Task checklist" not in sent[0]


@pytest.mark.asyncio
async def test_warm_turn_gets_no_recovery_block(tmp_path, monkeypatch) -> None:
    state, sent = _runner_state(tmp_path, monkeypatch, is_new=False, resumed=False)
    slot = state.get_or_create_slot("pill-chat")
    _seed_pill(slot)
    await _runner_turn(state, slot)
    assert sent and "[Task checklist" not in sent[0]


@pytest.mark.asyncio
async def test_cold_start_without_a_pill_is_untouched(tmp_path, monkeypatch) -> None:
    state, sent = _runner_state(tmp_path, monkeypatch, is_new=True, resumed=False)
    slot = state.get_or_create_slot("pill-chat")
    await _runner_turn(state, slot)
    assert sent and "[Task checklist" not in sent[0]
