"""A person ticking a row of the agent's checklist pill.

The pill above the composer mirrors the list the agent keeps with kiro-cli's
``todo_list`` tool. Until now the tool was the only writer: the pill could not
be edited from the dashboard, and when the native conversation restarted
(agent switch, failed ``session/load``, poisoned-conversation discard,
``/clear``) the agent's own list came back empty while the pill kept the old
one -- so neither the agent nor the person could tick the remaining rows.

This route lets the person flip one row. It writes the slot's copy only; the
agent learns of it the way it learns of the whole list after a restart, through
``Slot.todo_recovery_prompt`` on the next fresh native session.
"""

from __future__ import annotations

from aiohttp import web

from kiro_crew.dashboard.chat_folders import member_slot_write_refused
from kiro_crew.dashboard.handlers._shared import read_bounded_json
from kiro_crew.dashboard.state import DashboardState
from kiro_crew.dashboard.token_auth import (
    effective_request_app,
    refuse_unattributable_caller,
)
from kiro_crew.sel import sel

_OPERATION = "chat.slot_todo"


async def api_chat_slot_todo(request: web.Request) -> web.Response:
    """PATCH /api/chat/slots/{slot}/todo -- tick or untick one checklist row.

    Body: ``{"id": "<task id>", "completed": true|false}``. Answers the slot's
    refreshed ``todo`` payload (the same shape the ``slots`` snapshot carries)
    and broadcasts the same ``todo_update`` delta the agent's own tool result
    does, so every open tab repaints the pill.
    """
    state: DashboardState = request.app["state"]
    name = request.match_info["slot"]
    slot = state._slots.get(name)
    if not slot:
        return web.json_response({"error": "not found"}, status=404)
    if (
        refusal := refuse_unattributable_caller(state, request, _OPERATION)
    ) is not None:
        return refusal
    if (
        refusal := member_slot_write_refused(state, request, slot, _OPERATION)
    ) is not None:
        return refusal
    # A person's click on a pill. An app agent's session has no pill a person
    # can click, and the agent's own writer is its todo_list tool, so an app
    # caller is refused with the indistinguishable 404 the other slot writes use.
    request_app = effective_request_app(state, request)
    if request_app:
        sel().log_api_access(
            caller=request_app,
            operation=_OPERATION,
            outcome="denied",
            source="app_isolation",
            resources=f"slot={slot.key}",
            error="checklist rows are ticked by the person, not by an app",
        )
        return web.json_response(
            {"error": "not found", "code": "slot_not_found"}, status=404
        )
    body, err = await read_bounded_json(request)
    if err is not None:
        return err
    assert body is not None
    task_id = body.get("id")
    completed = body.get("completed")
    if not isinstance(task_id, (str, int)) or not isinstance(completed, bool):
        return web.json_response(
            {"error": "body must carry a task id and a boolean completed"}, status=400
        )
    if slot.todo_payload() is None:
        return web.json_response({"error": "this session has no checklist"}, status=404)
    if slot.set_todo_task_completed(str(task_id), completed):
        state.broadcast_ws(
            "todo_update", {"slot": slot.key, "todo": slot.todo_payload()}
        )
        sel().log_api_access(
            caller="dashboard",
            operation=_OPERATION,
            outcome="allowed",
            source="dashboard",
            resources=f"slot={slot.key} task={task_id} completed={completed}",
        )
    elif not any(
        str(t.get("id")) == str(task_id) for t in slot.todo_payload()["tasks"]
    ):
        return web.json_response({"error": "no such task"}, status=404)
    return web.json_response({"ok": True, "todo": slot.todo_payload()})
