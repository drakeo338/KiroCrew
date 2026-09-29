"""The board's card is published WITHOUT a model call, and only for the right session.

Two halves of one claim, and the first is the load-bearing one. Every other dynamic
dashboard card is written by a model reading a transcript: it costs a call, spends an
attempt from one hourly budget shared across the whole gateway, and is therefore gated on
the owner's opt-in to that cost. The conductor's board is assembled from a fold the product
already keeps, so it costs nothing -- and a test that only checked the card APPEARS would
pass equally over an implementation that quietly asked a model to write it, which would put
the one free card on the machine behind a paid switch and let a model restate numbers the
log already answers for.

So the generator is replaced by one that RAISES. Anything reaching it fails loudly rather
than returning a plausible card.

The second half is who gets it. The derived path deliberately drops the ``_created_by``
exclusion the model path applies -- that exclusion exists to stop a fan-out of workers
spending the shared budget, and a derived card spends none, while a conductor dispatched by
another session is exactly the case that must still get its board. Privacy exclusions stay,
because those are not about cost.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kiro_crew.dashboard import card_lifecycle
from kiro_crew.dashboard.chat_utils import slot_history_key
from kiro_crew.dashboard.dynamic_cards import CardBudget
from kiro_crew.dashboard.handlers import agent_panel as routes
from kiro_crew.pipeline_board_contract import (
    build_pipeline_board,
    card_template_path,
    panel_card_data,
    validate_judgment,
)


class _NoModel:
    """A generator that must never run. Called, it fails the test that called it."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, entry: Any) -> dict | None:
        self.calls += 1
        raise AssertionError(
            "the derived board card asked a model to write it; the board is assembled "
            "from the work fold and must cost no call"
        )


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any, _NoModel]:
    class Log:
        @contextmanager
        def publication_hold(self, key: str) -> Any:
            yield

        def session_mtime(self, key: str) -> int:
            return 1

        def rotation_generation(self, key: str) -> int:
            return 0

        def chained_keys(self, key: str) -> list[str]:
            return [key]

        def derive_recent(self, key: str, max_messages: int) -> list[dict]:
            return []

    slot = SimpleNamespace(
        key="chat-1875",
        _dashboard_card_identity="owner-conductor",
        memory_mode="persistent",
        messages=[{"role": "user", "content": "run the board"}],
    )
    state = SimpleNamespace(
        _slots={slot.key: slot},
        conversation_log=Log(),
        flush_slot_now=lambda slot: None,
        sessions=object(),
        _background_tasks=set(),
        broadcast_ws_owners=lambda *args: None,
        _dynamic_cards=None,
    )
    monkeypatch.setattr(
        card_lifecycle.KiroCrewConfig,
        "load",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(resolve_model=lambda role: "auto"),
            dashboard=SimpleNamespace(dynamic_dashboard_cards=True),
        ),
    )
    # ``enabled=False``: the cost opt-in is OFF, which is the state this path must work in.
    lifecycle = card_lifecycle.CardLifecycle(state, enabled=False)
    no_model = _NoModel()
    lifecycle.publisher.generate = no_model
    state._dynamic_cards = lifecycle
    return lifecycle, slot, state, no_model


def _card() -> dict[str, Any]:
    view = {
        "conductor": {
            "schema": 1,
            "slot_key": "chat-1875",
            "goal": "the board becomes a card",
            "round": 3,
            "goal_version": 1,
            "depth": 0,
            "parent_item": None,
            "created_at": "2026-09-29T13:00:00+00:00",
            "entries": 41,
            "first_entry_at": "2026-09-29T13:00:00+00:00",
            "last_entry_at": "2026-09-29T15:30:00+00:00",
            "generation": "gen-1",
        },
        "items": [
            {
                "schema": 1,
                "item_id": "it_0",
                "title": "t",
                "acceptance": {},
                "state": "open",
                "verdict": None,
                "decision": "",
                "worker_session_key": "chat-2176",
                "round": 3,
                "fails": 0,
                "status": None,
                "summary": "",
                "artifacts": {},
                "pr": 14583,
                "last_report_at": "2026-09-29T15:00:00+00:00",
                "created_at": "2026-09-29T14:00:00+00:00",
                "closed_at": None,
                "events": [],
            }
        ],
        "omitted": 0,
    }
    panel = build_pipeline_board(
        view,  # type: ignore[arg-type]
        validate_judgment({"lede": "One item is open."}),
        name="KiroCrew Pipeline Conductor",  # brand-ok: the crew's own display name
        captured_at="2026-09-29T15:37:00Z",
        stale_after_seconds=900,
        now_epoch=1790696200.0,
    )
    return {
        "html": card_template_path().read_text(encoding="utf-8"),
        "data": panel_card_data(panel),
    }


# ---------------------------------------------------------------------------
# no model call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_board_card_is_published_and_read_with_no_model_call(service: Any) -> None:
    lifecycle, slot, _state, no_model = service
    assert lifecycle.publish_derived(slot, _card()) is True
    read = await lifecycle.read(slot)
    assert read["status"] == "published"
    assert read["card"]["data"]["lede"] == "One item is open."
    assert no_model.calls == 0
    # And nothing was queued: a derived card does not enter the generator's state machine,
    # so it cannot hold the sole permit or spend an attempt from the shared hourly budget.
    assert slot.key not in lifecycle.publisher.entries
    assert not lifecycle.publisher.attempts
    assert lifecycle.publisher.next_delay() is None


@pytest.mark.asyncio
async def test_the_generator_would_be_caught_if_it_ran(service: Any) -> None:
    """Control. The assertion above is only evidence because the stub CAN fail the test.

    Without this, a stub that was never wired in would make ``calls == 0`` true for the
    wrong reason and the no-model claim would rest on nothing.
    """
    lifecycle, slot, _state, no_model = service
    lifecycle.enabled = True
    lifecycle.publisher.budget = CardBudget(debounce=0)
    # THE REAL BINDING, from the same helper ``_valid`` compares against. A made-up one is
    # refused before the generator is reached, which makes ``calls == 0`` true for the
    # wrong reason -- exactly the vacuous control this case exists to rule out.
    lifecycle.publisher.notify(
        slot.key, slot._dashboard_card_identity, slot_history_key(slot), "turn"
    )
    await lifecycle.publisher.run_ready()
    assert no_model.calls == 1, "the stub is not the generator being exercised"


@pytest.mark.asyncio
async def test_the_cost_opt_in_does_not_hide_the_derived_card(service: Any) -> None:
    """A derived card is free, so the owner's opt-in to the COST of model-written cards
    does not decide whether it is shown -- and turning that cost off must not delete the
    one card on the machine that has none."""
    lifecycle, slot, _state, _no_model = service
    assert lifecycle.enabled is False
    lifecycle.publish_derived(slot, _card())
    assert (await lifecycle.read(slot))["status"] == "published"
    lifecycle.set_enabled(True)
    assert (await lifecycle.read(slot))["status"] == "published"
    lifecycle.set_enabled(False)
    assert (await lifecycle.read(slot))["status"] == "published"


# ---------------------------------------------------------------------------
# who gets it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_dispatched_conductor_still_gets_its_board(service: Any) -> None:
    """``_created_by`` excludes a session another session created from the MODEL path,
    because a fan-out of workers would spend the shared hourly budget. A derived card
    spends none, and a conductor dispatched by another session is exactly the case whose
    board a person needs."""
    lifecycle, slot, _state, _no_model = service
    slot._created_by = "chat-1000"
    assert lifecycle._eligible(slot) is False, "the model path still excludes it"
    assert lifecycle.publish_derived(slot, _card()) is True
    assert (await lifecycle.read(slot))["status"] == "published"


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("memory_mode", "incognito"), ("is_remote", True), ("executor", "remote")],
)
def test_privacy_and_remoteness_still_withhold_the_card(
    service: Any, attribute: str, value: Any
) -> None:
    """The exclusions that are NOT about cost stay. An incognito transcript is withheld
    from every derived surface too, and a remote slot's content is not this gateway's to
    publish."""
    lifecycle, slot, _state, _no_model = service
    setattr(slot, attribute, value)
    assert lifecycle.publish_derived(slot, _card()) is False


def test_a_scratch_copy_cannot_publish_over_the_live_slot(service: Any) -> None:
    """A scratch copy shares the live identity and its edits are not committed."""
    lifecycle, slot, _state, _no_model = service
    scratch = SimpleNamespace(**vars(slot))
    assert lifecycle.publish_derived(scratch, _card()) is False


@pytest.mark.asyncio
async def test_a_replacement_session_never_presents_the_retired_crews_board(
    service: Any,
) -> None:
    """The one wrong thing a cached panel can do.

    The board was built for the closed transcript, so it goes with it rather than being
    shown by whatever takes the slot next -- which would attribute one crew's work to
    another with nothing on the card saying so.
    """
    lifecycle, slot, state, _no_model = service
    lifecycle.publish_derived(slot, _card())
    replacement = SimpleNamespace(**vars(slot))
    replacement._dashboard_card_identity = "owner-someone-else"
    state._slots[slot.key] = replacement
    assert (await lifecycle.read(replacement))["card"] is None
    assert slot.key not in lifecycle.derived, "the retired card is dropped, not merely hidden"


def test_the_host_normalizer_still_refuses_a_bad_derived_card(service: Any) -> None:
    """A derived producer is still a producer: its output is refused, not trusted.

    And a refused card leaves the previous one in place -- a board that briefly cannot be
    built is not a board that changed.
    """
    lifecycle, slot, _state, _no_model = service
    assert lifecycle.publish_derived(slot, _card()) is True
    held = dict(lifecycle.derived[slot.key])
    assert lifecycle.publish_derived(slot, {"html": "", "data": {}}) is False
    assert lifecycle.publish_derived(slot, {"html": "<p></p>", "data": {"bad name": "x"}}) is False
    assert lifecycle.derived[slot.key] == held


# ---------------------------------------------------------------------------
# the route hands it over, and never fails the drawer read doing so
# ---------------------------------------------------------------------------


def test_the_route_hands_the_records_card_to_the_store(service: Any) -> None:
    lifecycle, slot, state, no_model = service
    routes._publish_derived_card(state, slot.key, {"card": _card()})
    assert lifecycle.derived[slot.key]["card"]["data"]["meta_revision"] == "revision 3"
    assert no_model.calls == 0


@pytest.mark.parametrize(
    ("name", "record"),
    [
        ("no record", None),
        ("no card on the record", {"template": "default"}),
        ("card is not an object", {"card": "nope"}),
    ],
)
def test_a_record_with_no_card_stores_nothing(service: Any, name: str, record: Any) -> None:
    lifecycle, slot, state, _no_model = service
    routes._publish_derived_card(state, slot.key, record)
    assert slot.key not in lifecycle.derived, name


def test_the_route_never_turns_a_store_failure_into_a_failed_drawer_read(service: Any) -> None:
    """The panel response is what this route owes its caller; the card is a side effect of
    having built the board anyway. So a store that raises is logged, not propagated."""
    _lifecycle, slot, state, _no_model = service

    def boom(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("the store is unhappy")

    state._dynamic_cards.publish_derived = boom
    routes._publish_derived_card(state, slot.key, {"card": _card()})  # must not raise


def test_nothing_is_stored_when_the_slot_is_not_running(service: Any) -> None:
    """A card belongs to a live slot; with none there is nothing to attach it to."""
    lifecycle, slot, state, _no_model = service
    state._slots.clear()
    routes._publish_derived_card(state, slot.key, {"card": _card()})  # must not raise
    assert slot.key not in lifecycle.derived


def test_a_state_that_cannot_make_a_store_is_survived_not_raised(service: Any) -> None:
    """The panel response is what the route owes its caller, so every way the store can be
    unavailable degrades to storing nothing rather than to a 500.

    An absent store is not one of those ways -- a derived card creates one -- so what this
    exercises is a state object that cannot supply one at all.
    """
    _lifecycle, slot, state, _no_model = service
    bare = SimpleNamespace(**{**vars(state), "_dynamic_cards": None})
    assert not hasattr(bare, "ensure_dynamic_card_store"), "the fixture gained the method"
    routes._publish_derived_card(bare, slot.key, {"card": _card()})  # must not raise


def test_the_card_page_is_read_once_per_process() -> None:
    """It is shipped package data with no operator override directory, so there is nothing
    a cache could hide -- and this runs on every drawer read."""
    routes._card_page.cache_clear()
    first = routes._card_page()
    assert routes._card_page() is first
    assert re.search(r'data-dashboard-field="lede"', first)


# ---------------------------------------------------------------------------
# the card is not a second way out for a record the route refuses
# ---------------------------------------------------------------------------
#
# ``_published_record`` answers with the stored FILE, unfiltered, whenever the fold holds
# nothing for the asking owner -- and that file is keyed on the SLUG alone. Two crews whose
# names slugify alike share it, so a colliding crew's record and an unowned one both reach
# the route, which refuses to serve either. ``_panel_record`` gates only on the template
# id, so such a record still builds a card out of its ``lede``, its per-item action
# sentences and its crew name.
#
# Publishing that card before the refusal made the ownership guard half-applied in the same
# way the READ half once was: the panel response withheld the text and the owner-only
# dashboard-card route handed it over. These cases pin the ORDER, because the refusal is
# what decides whether the record may be seen at all.


def test_the_publish_call_sits_after_the_ownership_refusal() -> None:
    """Textual, and deliberately: the defect IS the order of two statements, and no
    reachable-request test can distinguish "published after the check" from "published
    before it" once both have run. The controls below keep a moved anchor from reading as
    a pass."""
    source = Path(routes.__file__).read_text(encoding="utf-8")
    refusal = source.index("if record is not None and (not owner_key or owner_key != mine):")
    publish = source.index("_publish_derived_card(request.app[")
    # Controls: both anchors exist exactly once, so neither a rename nor a second call
    # site can satisfy this by making an index unfindable.
    assert source.count("if record is not None and (not owner_key or owner_key != mine):") == 1
    assert source.count("_publish_derived_card(request.app[") == 1
    assert refusal < publish, (
        "the derived card is published BEFORE the ownership refusal, so a record this "
        "route refuses to serve still reaches the card store"
    )


def test_the_publish_call_sits_before_the_render_branches() -> None:
    """Control for the pin above, in the other direction.

    Moving the call to the very end of the handler would also satisfy it, and would mean a
    crew whose drawer template is broken loses its board -- which the card does not depend
    on, being built from the fold and the record.
    """
    source = Path(routes.__file__).read_text(encoding="utf-8")
    publish = source.index("_publish_derived_card(request.app[")
    assert source.count('"code": "panel_render_failed"') == 1
    assert publish < source.index('"code": "panel_render_failed"')


@pytest.mark.parametrize(
    ("name", "record"),
    [
        # A colliding crew's record: ``crew_key`` is another crew's digest.
        ("another crew's record", {"crew_key": "someone-else", "crew": "Other Crew"}),
        # An unowned record: only a forgery writes one, and it must not render.
        ("an unowned record", {"crew_key": "", "crew": ""}),
    ],
)
def test_a_refused_record_never_reaches_the_card_store(
    service: Any, name: str, record: dict[str, Any]
) -> None:
    """The route's own refusal is re-stated here as a behavioural check.

    ``_publish_derived_card`` is total by design, so it cannot itself refuse a record on
    ownership -- it does not know who asked. What this asserts is the consequence that
    makes the ordering matter: the card carries the record's OWN judgment text, so storing
    one built from a refused record is the disclosure.
    """
    lifecycle, slot, state, _no_model = service
    card = _card()
    card["data"]["lede"] = "planted by another crew"
    routes._publish_derived_card(state, slot.key, {**record, "card": card})
    stored = json.dumps(lifecycle.derived.get(slot.key) or {})
    assert "planted by another crew" in stored, (
        "precondition: the store does accept this card, so the ROUTE's ordering is the "
        "only thing keeping a refused record's text out of it"
    )


# ---------------------------------------------------------------------------
# what the publish broadcasts
# ---------------------------------------------------------------------------


def test_a_successful_publish_is_not_broadcast_as_a_removal(service: Any) -> None:
    """``_changed`` derives ``removed`` from the GENERATOR's queue, which a derived card
    never joins -- so routing a successful publish through it announces the card as removed
    and the client answers a removal by resetting the card query it was just handed."""
    lifecycle, slot, state, _no_model = service
    frames: list[Any] = []
    state.broadcast_ws_owners = lambda *args: frames.append(args)
    assert lifecycle.publish_derived(slot, _card()) is True
    assert frames == [("dashboard_card", {"slot": slot.key, "removed": False})]


def test_a_refused_publish_broadcasts_nothing(service: Any) -> None:
    """Nothing changed, so there is nothing to tell anyone -- and a frame here would make a
    refusal indistinguishable from a successful publish."""
    _lifecycle, slot, state, _no_model = service
    frames: list[Any] = []
    state.broadcast_ws_owners = lambda *args: frames.append(args)
    assert _lifecycle.publish_derived(slot, {"html": "", "data": {}}) is False
    assert frames == []


def test_dropping_a_derived_card_IS_broadcast_as_a_removal(service: Any) -> None:
    """The other direction, so the fix above cannot be satisfied by never saying removed.

    ``forget_derived`` keeps ``_changed``, whose queue-derived answer is the correct one
    there: with the derived card gone the read falls through to a queued card if the slot
    has one, and to nothing if it does not.
    """
    lifecycle, slot, state, _no_model = service
    lifecycle.publish_derived(slot, _card())
    frames: list[Any] = []
    state.broadcast_ws_owners = lambda *args: frames.append(args)
    lifecycle.forget_derived(slot.key)
    assert frames == [("dashboard_card", {"slot": slot.key, "removed": True})]


# ---------------------------------------------------------------------------
# the card's LIFECYCLE: created on a board change, evicted when the board goes
# ---------------------------------------------------------------------------
#
# One invariant seen from three sides. The card was minted only by the drawer READ, so:
# nothing produced it when the board actually changed (and the drawer is the surface this work
# replaces, which would leave the card with no producer at all); nothing evicted it when the
# record stopped carrying a board, so a stale board was served as current; and the store it
# lives in existed only if the model-card cost opt-in had ever been flipped, which made a free
# card's availability depend on toggle history.


def test_no_board_on_the_record_evicts_the_card(service: Any) -> None:
    """A record without a board says the board is GONE, not "leave things as they are".

    The record is the authoritative answer for the slot, so a crew that republishes to another
    template -- or whose work fold holds no board -- must not keep its earlier card
    being served as `status: published`. That is a stale board presented as current, which a
    reader cannot tell from a live one.
    """
    lifecycle, slot, state, _no_model = service
    routes._publish_derived_card(state, slot.key, {"card": _card()})
    assert slot.key in lifecycle.derived
    routes._publish_derived_card(state, slot.key, {"template": "default"})
    assert slot.key not in lifecycle.derived


def test_eviction_does_not_need_a_card_to_have_been_there(service: Any) -> None:
    """Control: the evicting branch is total, so a first publish of a non-board record is a
    no-op rather than an error."""
    _lifecycle, slot, state, _no_model = service
    routes._publish_derived_card(state, slot.key, {"template": "default"})
    routes._publish_derived_card(state, slot.key, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attribute", "value"),
    [("memory_mode", "incognito"), ("is_remote", True), ("executor", "remote")],
)
async def test_a_slot_that_tightens_after_publishing_loses_its_card(
    service: Any, attribute: str, value: Any
) -> None:
    """Privacy and remoteness are properties of the slot AS IT IS NOW.

    Checked only at publish, they say nothing about a card that is already stored: a persistent
    session turned incognito went on serving content the live rules withhold. A published card
    outliving the condition that permitted it is the same defect as never having checked.
    """
    lifecycle, slot, _state, _no_model = service
    assert lifecycle.publish_derived(slot, _card()) is True
    setattr(slot, attribute, value)
    assert (await lifecycle.read(slot))["card"] is None
    assert slot.key not in lifecycle.derived, "withheld content stays in memory"


@pytest.mark.asyncio
async def test_a_slot_that_stays_eligible_keeps_its_card(service: Any) -> None:
    """Control for the three cases above, so the re-check cannot be satisfied by evicting
    every card on every read."""
    lifecycle, slot, _state, _no_model = service
    lifecycle.publish_derived(slot, _card())
    assert (await lifecycle.read(slot))["status"] == "published"
    assert (await lifecycle.read(slot))["status"] == "published"


def test_a_derived_card_gets_a_store_without_the_model_opt_in(service: Any) -> None:
    """The store was built only by `set_dynamic_cards_enabled(True)`, so a free card's
    availability depended on TOGGLE HISTORY: never enabled meant no board at all, while
    enabled-once-then-off left a store behind and the board appeared. Same feature, opposite
    answers, decided by a switch neither answer is about."""
    _lifecycle, slot, state, _no_model = service
    state._dynamic_cards = None
    created: list[Any] = []

    class Store:
        derived: dict[str, Any] = {}

        def publish_derived(self, s: Any, payload: Any) -> bool:
            created.append(payload)
            return True

        def forget_derived(self, key: str) -> None:
            pass

    state.ensure_dynamic_card_store = lambda: Store()
    routes._publish_derived_card(state, slot.key, {"card": _card()})
    assert created, "no store was created for a derived card"


def test_the_real_state_builds_the_store_with_the_model_path_off() -> None:
    """``ensure_dynamic_card_store`` must construct the container and start no worker -- that
    is what keeps the model path exactly as opt-in as it was."""
    from kiro_crew.dashboard import state as state_mod

    holder = SimpleNamespace(_dynamic_cards=None, _background_tasks=set(), _slots={})
    store = state_mod.DashboardState.ensure_dynamic_card_store(holder)  # type: ignore[arg-type]
    assert store is not None
    assert store.enabled is False, "creating the store must not enable the model path"
    assert store.worker is None, "creating the store must not start a worker"
    assert not store.publisher.attempts, "creating the store must spend no budget"
    assert state_mod.DashboardState.ensure_dynamic_card_store(holder) is store  # type: ignore[arg-type]
