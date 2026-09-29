"""The ``subagents`` fold -- one test per property it promises.

The fold answers one question from one savepoint read: which children this session
dispatched, and for each one its outcome, its duration and what it cost. The
properties worth pinning are the ones a reader would otherwise have to trust:

* a closer lands on its own row, and nowhere else;
* a closer with NO row is still counted, because crash-repair writes one for a child
  whose ``spawned`` fell past the retention cap, and dropping it would under-report
  what the session actually spent;
* ``totals.spawned`` counts every dispatch, while ``by_id`` holds only the retained
  ones -- so the two disagree by ``omitted``, on purpose;
* absent credits are never read as zero, which is the same posture ``usage`` takes.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from aiohttp.test_utils import make_mocked_request

from kiro_crew import crew_log as lg
from kiro_crew.crew_log import CrewLog
from kiro_crew.crew_log import projection as crew_log
from kiro_crew.dashboard.handlers import crew_log as routes

SESSION = "s-subagents"
GATEWAY = "gateway"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test writes into its own data home, never the live one."""
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    yield


def _log(unit_id: str = SESSION, slot: str = "dashboard:1") -> CrewLog:
    return CrewLog.create(lg.KIND_SESSION, unit_id, owner="raymond", agent="kirocrew", slot=slot)


def _opened(handle: CrewLog) -> None:
    handle.append(
        "session/opened",
        {
            "agent": "kirocrew",
            "slot": "dashboard:1",
            "model": "opus",
            "cwd": "/w",
            "owner": "raymond",
            "resumed": False,
        },
        src=GATEWAY,
    )


def _spawned(
    handle: CrewLog,
    agent_id: str,
    *,
    turn: int = 1,
    agent: str = "kirocrew-worker",
    model: str = "opus",
    scope: dict | None = None,
) -> None:
    data: dict = {"agent_id": agent_id, "turn": turn, "agent": agent, "model": model}
    if scope is not None:
        data["scope"] = scope
    handle.append("subagent/spawned", data, src=GATEWAY)


def _steered(handle: CrewLog, agent_id: str, *, mode: str = "interrupt") -> None:
    handle.append("subagent/steered", {"agent_id": agent_id, "mode": mode}, src=GATEWAY)


def _completed(
    handle: CrewLog, agent_id: str, *, ms: int = 1000, credits: float | None = None
) -> None:
    data: dict = {"agent_id": agent_id, "ms": ms}
    if credits is not None:
        data["credits"] = credits
    handle.append("subagent/completed", data, src=GATEWAY)


def _failed(
    handle: CrewLog,
    agent_id: str,
    *,
    outcome: str = "failed",
    ms: int = 500,
    reason: str = "boom",
    credits: float | None = None,
) -> None:
    data: dict = {"agent_id": agent_id, "outcome": outcome, "ms": ms, "reason": reason}
    if credits is not None:
        data["credits"] = credits
    handle.append("subagent/failed", data, src=GATEWAY)


def _fold(unit_id: str = SESSION) -> dict:
    """The ``subagents`` value for *unit_id*, folded from the start of its log."""
    return crew_log.fold_session(unit_id, names=("subagents",)).projection("subagents").value


# --- the six the plan asks for --------------------------------------------- #


def test_a_child_that_finished_lands_its_outcome_duration_and_cost_on_its_own_row():
    handle = _log()
    _opened(handle)
    _spawned(handle, "a-1", agent="kirocrew-worker", model="opus")
    _completed(handle, "a-1", ms=1200, credits=2.0)

    value = _fold()
    row = value["by_id"]["a-1"]
    assert row["outcome"] == "completed"
    assert row["ms"] == 1200
    assert row["credits"] == 2.0
    assert row["agent"] == "kirocrew-worker"
    assert row["model"] == "opus"
    assert value["totals"]["completed"] == 1
    assert value["totals"]["credits"] == 2.0
    assert value["totals"]["ms"] == 1200
    # A closed child is out of ``open``.
    assert value["open"] == []


def test_a_stopped_child_counts_as_stopped_and_not_as_a_failure():
    """The runtime's three-way outcome exists so 'not success' is not read as 'failed'."""
    handle = _log()
    _opened(handle)
    _spawned(handle, "a-1")
    _failed(handle, "a-1", outcome="stopped", ms=300)

    value = _fold()
    assert value["by_id"]["a-1"]["outcome"] == "stopped"
    assert value["totals"]["stopped"] == 1
    assert value["totals"]["failed"] == 0


def test_a_steer_lands_on_its_child_s_row():
    handle = _log()
    _opened(handle)
    _spawned(handle, "a-1")
    _steered(handle, "a-1")
    _steered(handle, "a-1", mode="follow_up")

    value = _fold()
    assert value["by_id"]["a-1"]["steers"] == 2
    assert value["totals"]["steers"] == 2


def test_a_steer_for_a_child_with_no_row_is_counted_but_invents_none():
    """A steer can name a child whose spawn fell past the cap, or predates this fold."""
    handle = _log()
    _opened(handle)
    _steered(handle, "ghost")

    value = _fold()
    assert "ghost" not in value["by_id"]
    # Counted, so the total is not silently short of what the session did.
    assert value["totals"]["steers"] == 1


def test_past_the_retention_cap_a_further_dispatch_is_counted_and_not_retained():
    handle = _log()
    _opened(handle)
    for index in range(crew_log.OPEN_RETAIN_LIMIT + 1):
        _spawned(handle, f"a-{index}")

    value = _fold()
    assert len(value["by_id"]) == crew_log.OPEN_RETAIN_LIMIT
    assert len(value["open"]) == crew_log.OPEN_RETAIN_LIMIT
    assert value["omitted"] == 1
    assert value["limit"] == crew_log.OPEN_RETAIN_LIMIT


def test_a_closer_with_no_row_is_counted_and_builds_none():
    """crash-repair closes a child whose ``spawned`` this fold never retained."""
    handle = _log()
    _opened(handle)
    _failed(handle, "ghost", outcome="unknown", ms=700, credits=1.5)

    value = _fold()
    assert value["by_id"] == {}
    totals = value["totals"]
    assert totals["unknown"] == 1
    assert totals["ms"] == 700
    assert totals["credits"] == 1.5
    # And it is legible as unmatched rather than looking like a retained child.
    assert totals["closed_unmatched"] == 1


@pytest.mark.asyncio
async def test_the_route_serves_the_fold_with_no_route_change():
    """``GET /api/sessions/{id}/crew-log/projection/subagents`` on the existing door."""
    handle = _log()
    _opened(handle)
    _spawned(handle, "a-1")
    _completed(handle, "a-1", ms=900, credits=0.5)

    request = make_mocked_request("GET", f"/api/sessions/{SESSION}/crew-log/projection/subagents")
    request.match_info["id"] = SESSION
    request.match_info["name"] = "subagents"
    with patch(
        "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
        return_value=True,
    ):
        response = await routes.api_session_crew_log_projection(request)
    assert response.status == 200
    value = json.loads(response.body)["value"]
    assert value["by_id"]["a-1"]["outcome"] == "completed"
    assert value["totals"]["credits"] == 0.5


# --- the conductor's two amendments ---------------------------------------- #


def test_totals_spawned_counts_every_dispatch_while_by_id_holds_the_retained_ones():
    """The two disagree by ``omitted``, and the render says so rather than hiding it."""
    handle = _log()
    _opened(handle)
    for index in range(crew_log.OPEN_RETAIN_LIMIT + 3):
        _spawned(handle, f"a-{index}")

    value = _fold()
    assert value["totals"]["spawned"] == crew_log.OPEN_RETAIN_LIMIT + 3
    assert len(value["by_id"]) == crew_log.OPEN_RETAIN_LIMIT
    assert value["omitted"] == 3
    # The identity a reader can check: every dispatch is either retained or omitted.
    assert value["totals"]["spawned"] == len(value["by_id"]) + value["omitted"]


def test_an_orphan_closer_still_bills_its_duration_and_cost_into_the_totals():
    """Amendment (b): a closer is never silently dropped for want of a row."""
    handle = _log()
    _opened(handle)
    _spawned(handle, "kept")
    _completed(handle, "kept", ms=100, credits=1.0)
    # Two closers whose spawn this fold never saw.
    _completed(handle, "ghost-1", ms=200, credits=2.0)
    _failed(handle, "ghost-2", outcome="stopped", ms=300, credits=3.0)

    totals = _fold()["totals"]
    assert totals["completed"] == 2
    assert totals["stopped"] == 1
    assert totals["ms"] == 600
    assert totals["credits"] == 6.0
    assert totals["closed_unmatched"] == 2


# --- absent is not zero ---------------------------------------------------- #


def test_a_closer_that_reported_no_credits_leaves_the_row_without_a_number():
    """Absent credits are not a measurement of zero, the posture ``usage`` already takes."""
    handle = _log()
    _opened(handle)
    _spawned(handle, "a-1")
    _completed(handle, "a-1", ms=400)

    value = _fold()
    assert value["by_id"]["a-1"]["credits"] is None
    assert value["totals"]["credits"] == 0.0
    # What tells a reader the zero is 'nobody said' rather than 'it cost nothing'.
    assert value["totals"]["credits_reported"] == 0


def test_an_integer_too_large_to_be_a_float_does_not_crash_the_fold():
    """Python's ``int`` has no magnitude limit; ``float()`` raises past about 1.8e308.

    Such a value passes the numeric type check, so the conversion is where it bites -- and
    the line stays on disk, so an escaping ``OverflowError`` would turn EVERY later read of
    this session's fold into a crash rather than costing one number.
    """
    fold = crew_log._FOLDS["subagents"]
    state = fold.start()
    fold.step(
        state,
        crew_log.Entry(
            type="subagent/spawned", seq=1, time=1, src=GATEWAY, data={"agent_id": "a-1"}
        ),
    )
    fold.step(
        state,
        crew_log.Entry(
            type="subagent/completed",
            seq=2,
            time=2,
            src=GATEWAY,
            data={"agent_id": "a-1", "ms": 10, "credits": 10**400},
        ),
    )

    value = fold.render(state)
    # The child still closed -- only its unusable charge was dropped.
    assert value["by_id"]["a-1"]["outcome"] == "completed"
    assert value["by_id"]["a-1"]["credits"] is None
    assert value["totals"]["credits"] == 0.0
    assert value["totals"]["credits_reported"] == 0


def test_the_running_count_stays_exact_when_retention_drops_a_dispatch():
    """``open`` lists retained rows; ``running`` is derived from the totals.

    A dropped dispatch has no row to be missing an outcome from, so a reader counting
    ``open`` would report a session with more children in flight than the cap as having at
    most the cap -- understating, silently, exactly when the number matters most.
    """
    handle = _log()
    _opened(handle)
    for index in range(crew_log.OPEN_RETAIN_LIMIT + 20):
        _spawned(handle, f"a-{index}")
    # Close two of the retained ones, so running is not simply "everything dispatched".
    _completed(handle, "a-0", ms=5)
    _failed(handle, "a-1", outcome="stopped", ms=5)

    value = _fold()
    assert value["omitted"] == 20
    # The listed subset is capped by what was retained.
    assert len(value["open"]) == crew_log.OPEN_RETAIN_LIMIT - 2
    # The exact answer is not.
    assert value["running"] == crew_log.OPEN_RETAIN_LIMIT + 20 - 2
    assert value["running"] > len(value["open"])


def test_more_closers_than_openers_reads_as_none_running_not_a_negative():
    handle = _log()
    _opened(handle)
    _completed(handle, "ghost-1", ms=1)
    _failed(handle, "ghost-2", outcome="failed", ms=1)

    assert _fold()["running"] == 0


def test_a_row_retains_only_fields_something_reads():
    """A field kept against a reader that does not exist is state paid for on every copy.

    The panel draws agent, model, outcome, ms, credits, and -- in the row's detail line --
    turn, steers and reason; the render orders by ``seq_spawned``. Nothing draws the child's
    inherited scope or its spawn time, so neither is retained.
    """
    handle = _log()
    _opened(handle)
    _spawned(handle, "a-1", scope={"memory": True, "lessons": True, "project": False})

    row = _fold()["by_id"]["a-1"]
    assert set(row) == {
        "agent_id",
        "seq_spawned",
        "turn",
        "agent",
        "model",
        "outcome",
        "ms",
        "credits",
        "reason",
        "steers",
    }


@pytest.mark.parametrize("planted", [True, "2.0", None, {"n": 1}, [2.0]])
def test_a_credit_charge_that_is_not_a_number_is_refused_rather_than_coerced(planted):
    """Driven at the FOLD, because the writer's declaration already refuses these.

    A declaration binds the writer; a damaged or planted line is exactly the input
    that ignores it, so the coercion has to hold on the read side too. ``True`` is the
    one that matters most: it is an ``int`` in Python, so a check that forgot to exclude
    ``bool`` would bill a boolean as one credit.
    """
    fold = crew_log._FOLDS["subagents"]
    state = fold.start()
    fold.step(
        state,
        crew_log.Entry(
            type="subagent/spawned", seq=1, time=1, src=GATEWAY, data={"agent_id": "a-1"}
        ),
    )
    fold.step(
        state,
        crew_log.Entry(
            type="subagent/completed",
            seq=2,
            time=2,
            src=GATEWAY,
            data={"agent_id": "a-1", "ms": 10, "credits": planted},
        ),
    )

    value = fold.render(state)
    assert value["by_id"]["a-1"]["credits"] is None
    assert value["totals"]["credits"] == 0.0
    assert value["totals"]["credits_reported"] == 0


# --- registration ---------------------------------------------------------- #


def test_the_fold_is_registered_and_advertised():
    assert "subagents" in crew_log.PROJECTION_NAMES
    assert "subagents" in crew_log.FOLD_NAMES
    assert crew_log.require_name("subagents") == "subagents"


def test_the_fold_is_lazy_because_eager_folding_is_slot_keyed():
    """Pinned so the reason survives, rather than reading as an oversight.

    The plan asked for this fold EAGER. It cannot be, and the constraint is not this
    fold's: eager folding continues the warm SLOT memo -- ``_fold_batch`` resolves a
    slot from the unit header and calls ``read_slot_projection`` -- and ``subagents`` is
    keyed by one SESSION. ``projection.py`` states the rule at import
    (``EAGER_FOLD_NAMES <= SLOT_PROJECTION_NAMES``) and ``eager._WAKE_TYPES`` is a fixed
    three-type set that no ``subagent/*`` type is in, so declaring this fold eager would
    register a mode the process cannot honour. Making it eager means building a warm
    SESSION path, which is a change to the eager module's contract rather than a fold.
    """
    assert crew_log._FOLDS["subagents"].mode == "lazy"
    assert "subagents" not in crew_log.EAGER_FOLD_NAMES
    assert set(crew_log.EAGER_FOLD_NAMES) <= set(crew_log.SLOT_PROJECTION_NAMES)


def test_the_fold_declares_the_four_subagent_types_it_reads():
    fold = crew_log._FOLDS["subagents"]
    assert fold.affects == frozenset(
        {
            "subagent/spawned",
            "subagent/steered",
            "subagent/completed",
            "subagent/failed",
        }
    )


def test_the_copier_covers_every_container_the_step_reaches():
    """A shared nested container would let a step edit the state it was handed."""
    fold = crew_log._FOLDS["subagents"]
    state = fold.start()
    state["by_id"]["a-1"] = {"steers": 0, "ms": 0}
    state["open"].append("a-1")
    copied = fold.copied(state)
    copied["by_id"]["a-1"]["steers"] = 9
    copied["open"].append("a-2")
    copied["totals"]["spawned"] = 9
    assert state["by_id"]["a-1"]["steers"] == 0
    assert state["open"] == ["a-1"]
    assert state["totals"]["spawned"] == 0


def test_an_entry_this_fold_does_not_read_moves_nothing():
    """``affects`` is not the only guard, so the step's own branching has to be total.

    ``affects`` spares this fold the entries it does not read on the KERNEL's path, but
    ``advance`` and ``fold`` call ``step`` for every entry in the file. A step that
    reached its closer branch by falling through would bill every ``turn/completed`` in
    the log as a child that closed -- which is what the savepoint suite caught: 293
    phantom children, and the session's own turn credits billed as theirs.
    """
    fold = crew_log._FOLDS["subagents"]
    state = fold.start()
    for seq, (entry_type, data) in enumerate(
        (
            ("turn/completed", {"turn": 1, "credits": 0.5, "duration_ms": 900, "ms": 120}),
            ("step/completed", {"turn": 1, "step": 1, "ms": 120}),
            ("background/completed", {"kind": "title", "credits": 0.5, "ms": 20}),
            ("tool/completed", {"name": "fs_read", "call_id": "c-1", "status": "ok"}),
            ("session/opened", {"agent": "kirocrew", "model": "opus"}),
        ),
        start=1,
    ):
        fold.step(state, crew_log.Entry(type=entry_type, seq=seq, time=seq, src=GATEWAY, data=data))

    assert fold.render(state) == fold.render(fold.start())


def test_rows_render_in_dispatch_order():
    handle = _log()
    _opened(handle)
    _spawned(handle, "z-first")
    _spawned(handle, "a-second")

    assert list(_fold()["by_id"]) == ["z-first", "a-second"]
