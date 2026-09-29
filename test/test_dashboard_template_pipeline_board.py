"""The pipeline board CARD: the page binds exactly what the flattener writes.

``mypy src/kiro_crew/`` checks both ends of ``build_pipeline_board`` -- a
:class:`~kiro_crew.work_vocab.WorkBoardView` in, a ``PipelineBoardPanel`` out, required
keys and all. It cannot see inside ``dashboard_templates/pipeline_board.html``, so the
html end of the agreement needs a reader of its own. This is that reader.

TWO GATES, because the card format needs both and neither implies the other.

The first is PARITY, strict and in both directions: the page's
``data-dashboard-field`` set equals ``panel_card_data``'s output keys. The host binds a
field it was not given to the EMPTY STRING, so a binding with no writer is a blank cell
on a page of counts -- indistinguishable from a real zero -- and a written key nothing
binds is a value nobody sees.

The second is COVERAGE, which the drawer template's parity gate gets for free: it compares
that template's reads against the contract's own leaves, so a contract field nobody renders
fails there. A card cannot do that: it has 24 fields for 17
contract leaves whose lists are unbounded, so several leaves reach the page inside a
composed sentence rather than as a field of their own. :func:`test_every_contract_leaf_
reaches_a_card_field` closes that by SEEDING each leaf with a unique marker and requiring
it to come out the other side -- which proves the value travels, not merely that a name
matches.

Every extractor here refuses rather than returning a partial answer. A short set reads
as a page that genuinely binds less, which is the one direction an equality assertion
cannot catch alone -- so the preconditions run as their own cases.

Written with the stdlib html parser, not a regular expression. An attribute scan by
regex mis-reads an attribute inside a comment, a value in the other quote character and
an upper-case tag name, and each of those makes this gate QUIETLY incomplete.
"""

from __future__ import annotations

import re
import subprocess
import sys
from html.parser import HTMLParser
from typing import Any

import pytest

from kiro_crew.dashboard.dynamic_cards import (
    _FIELD_NAME,
    MAX_DATA_BYTES,
    MAX_HTML_BYTES,
    normalize_card,
)
from kiro_crew.pipeline_board_contract import (
    BOARD_COLUMN_NAMES,
    BOARD_STAT_KEYS,
    CONTRACT_VERSION,
    MAX_CARD_DATA_BYTES,
    NOT_SAID,
    UNREADABLE,
    PipelineBoardPanel,
    build_pipeline_board,
    card_template_path,
    panel_card_data,
    validate_judgment,
)
from kiro_crew.subprocess_utf8 import UTF8_TEXT
from kiro_crew.work_vocab import WORK_ITEM_STATES, WorkBoardView

# ``MAX_DATA_BYTES``/``MAX_HTML_BYTES`` and ``_FIELD_NAME`` are IMPORTED FROM THE HOST,
# not restated. The contract module keeps its own copies so it can bound a card without
# importing the dashboard package, and the two pairs are asserted equal below -- a cap the
# host raises or lowers has to reach the producer, and a producer that bounds a card against
# a number the host does not use builds cards the host drops whole.


# ---------------------------------------------------------------------------
# read the page
# ---------------------------------------------------------------------------


class ExtractionRefused(Exception):
    """The page is not something this reader can answer about."""


#: The binding attribute the host reads. One attribute, one field, one value.
_BINDING = "data-dashboard-field"

#: Tags that would make the page a control surface or reach outside it. The host strips
#: every one, so a page authoring them ships a layout with holes in it and an author who
#: believes the control exists. A dashboard card states facts and offers no actions.
_FORBIDDEN_TAGS = frozenset(
    {
        "script",
        "form",
        "input",
        "button",
        "textarea",
        "select",
        "iframe",
        "object",
        "embed",
        "applet",
        "template",
        "noscript",
        "link",
        "meta",
        "base",
        "frame",
        "animate",
        "set",
    }
)

#: CSS properties ``dashboardDocument.ts`` DELETES in card mode, because they make the
#: browser draw characters the backend's text scan never reads. A label put in one of
#: them is removed and leaves its row unlabelled, with nothing saying why.
_STRIPPED_PROPERTIES = ("content", "list-style", "text-emphasis", "text-overflow", "quotes")


class _Scanner(HTMLParser):
    """Collects the page's bindings, tags, and anything pointing outside it."""

    def __init__(self) -> None:
        super().__init__()
        self.fields: list[str] = []
        self.tags: set[str] = set()
        self.outbound: set[tuple[str, str]] = set()
        self.empty_bindings = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag.lower())
        for name, value in attrs:
            lowered = name.lower()
            if lowered == _BINDING:
                text = (value or "").strip()
                if not text:
                    # A binding with no name binds nothing: the host looks the empty
                    # string up in the data and writes the empty string back, so the
                    # element renders blank forever with nothing to trace it to.
                    self.empty_bindings += 1
                    continue
                self.fields.append(text)
                continue
            # The LOCAL name, so an SVG ``xlink:href`` is judged as an ``href``.
            local = lowered.rsplit(":", 1)[-1]
            if local not in {"src", "srcset", "href"}:
                continue
            target = (value or "").strip()
            if local == "href" and target.startswith("#"):
                continue
            self.outbound.add((lowered, target))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _scan(html: str) -> _Scanner:
    scanner = _Scanner()
    scanner.feed(html)
    scanner.close()
    return scanner


def card_fields(html: str) -> list[str]:
    """Every field *html* binds, as a LIST so a duplicated binding stays visible.

    Refuses a page that binds nothing: an extractor returning an empty answer would make
    the equality gate report the flattener as writing entirely unread keys, which sends a
    reader to delete the flattener instead of fixing the page.
    """
    scanner = _scan(html)
    if not scanner.fields:
        raise ExtractionRefused(f"the page binds no {_BINDING} at all -- renamed?")
    if scanner.empty_bindings:
        raise ExtractionRefused(f"{scanner.empty_bindings} binding(s) name no field")
    return scanner.fields


def _page() -> str:
    return card_template_path().read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# a real board to flatten
# ---------------------------------------------------------------------------


def _item(
    ident: str,
    *,
    state: str = "open",
    pr: int | None = None,
    worker: str | None = None,
    round_: int = 1,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "item_id": ident,
        "title": "an item",
        "acceptance": {},
        "state": state,
        "verdict": None,
        "decision": "",
        "worker_session_key": worker,
        "round": round_,
        "fails": 0,
        "status": None,
        "summary": "",
        "artifacts": {},
        "pr": pr,
        "last_report_at": "2026-09-29T15:00:00+00:00",
        "created_at": "2026-09-29T14:00:00+00:00",
        "closed_at": None,
        "events": [],
    }


def _view(items: list[dict[str, Any]], *, omitted: int = 0, entries: int = 41) -> WorkBoardView:
    return {
        "conductor": {
            "schema": 1,
            "slot_key": "chat-1875",
            "goal": "the board becomes a dashboard card",
            "round": 3,
            "goal_version": 1,
            "depth": 0,
            "parent_item": None,
            "created_at": "2026-09-29T13:00:00+00:00",
            "entries": entries,
            "first_entry_at": "2026-09-29T13:00:00+00:00",
            "last_entry_at": "2026-09-29T15:30:00+00:00",
            "generation": "gen-1",
        },
        "items": items,
        "omitted": omitted,
    }


#: A board with something in every state, an action, a check tally and a metric gloss,
#: so a leaf that only appears when a judgment was written is still exercised.
def _full_board() -> PipelineBoardPanel:
    items = [
        _item("it_0", state="open", pr=14583, worker="chat-1875", round_=3),
        _item("it_1", state="accepted", pr=14689),
        _item("it_2", state="rejected", worker="chat-2176"),
        _item("it_3", state="abandoned", pr=12046),
    ]
    judgment = validate_judgment(
        {
            "lede": "Four items, one needs a person.",
            "you": {"it_0": "push the rebase once the base fix lands"},
            "notes": {"items": "items the fold folded"},
            "checks": {"it_0": "41/47"},
        }
    )
    return build_pipeline_board(
        _view(items, omitted=2),
        judgment,
        name="KiroCrew Pipeline Conductor",  # brand-ok: the crew's own display name
        captured_at="2026-09-29T15:37:00Z",
        stale_after_seconds=900,
        now_epoch=1790696200.0,
    )


# ---------------------------------------------------------------------------
# the extractor's own preconditions
# ---------------------------------------------------------------------------


def test_the_extractor_reads_the_card_page() -> None:
    """A precondition for every assertion below: it finds bindings at all."""
    found = card_fields(_page())
    assert found, "extracted no bindings from the card page"
    for expected in ("lede", "meta_name", "column_0_rows", "stat_items"):
        assert expected in found, f"{expected} not extracted"


def test_a_page_that_binds_nothing_is_refused() -> None:
    with pytest.raises(ExtractionRefused, match="binds no"):
        card_fields("<div><p>nothing bound here</p></div>")


def test_a_binding_naming_no_field_is_refused() -> None:
    """An empty binding renders blank forever with nothing to trace it to."""
    with pytest.raises(ExtractionRefused, match="name no field"):
        card_fields('<p data-dashboard-field=""></p><p data-dashboard-field="lede"></p>')


def test_an_upper_case_binding_is_still_read() -> None:
    """Attribute names are case-insensitive, so the scan must be.

    Read case-sensitively, ``DATA-DASHBOARD-FIELD`` is not a binding at all: the parity
    set silently omits it and the equality gate passes while a bound element has no
    writer. A gate that is quietly incomplete is worse than one that fails.
    """
    assert card_fields('<p DATA-DASHBOARD-FIELD="lede"></p>') == ["lede"]


def test_a_binding_inside_a_comment_is_not_read() -> None:
    """The page documents its own field names beside the markup that binds them.

    The page's header comment names ``meta.name`` and two field names in prose; counted
    as bindings they would enter the parity set and the gate would pass over a field the
    flattener does not write.
    """
    assert card_fields(
        '<!-- data-dashboard-field="phantom" --><p data-dashboard-field="lede"></p>'
    ) == ["lede"]


# ---------------------------------------------------------------------------
# GATE ONE: parity
# ---------------------------------------------------------------------------


def test_the_card_binds_exactly_what_the_flattener_writes() -> None:
    """THE assertion: the html end of the contract, which mypy cannot see.

    STRICT equality with no exemption list. An exemption list is a hiding place: a field
    inconvenient to fill can be bound in the page and named in the list, and every check
    still passes.
    """
    bound = set(card_fields(_page()))
    written = set(panel_card_data(_full_board()))
    assert bound - written == set(), f"bound with no writer: {sorted(bound - written)}"
    assert written - bound == set(), f"written but never bound: {sorted(written - bound)}"


def test_no_field_is_bound_twice() -> None:
    """Two elements bound to one field both show it, which is a layout bug the set
    comparison above cannot see: sets collapse the duplicate and the page reads as
    correct while one number is printed in two places."""
    found = card_fields(_page())
    duplicates = sorted({name for name in found if found.count(name) > 1})
    assert not duplicates, f"bound more than once: {duplicates}"


def test_a_bound_field_with_no_writer_is_caught() -> None:
    """Direction one, planted. Without this the equality above could be vacuous."""
    page = _page()
    seeded = page.replace(
        '<p class="lede" data-dashboard-field="lede"></p>',
        '<p class="lede" data-dashboard-field="lede"></p>'
        '<p data-dashboard-field="invented"></p>',
        1,
    )
    assert seeded != page, "failed to plant the binding -- the anchor line moved"
    assert set(card_fields(seeded)) - set(panel_card_data(_full_board())) == {"invented"}


def test_a_written_key_with_no_binding_is_caught() -> None:
    """Direction two, planted by removing one binding from the page."""
    page = _page()
    anchor = '<span class="dim" data-dashboard-field="meta_revision"></span>'
    assert page.count(anchor) == 1, "the revision binding moved"
    bound = set(card_fields(page.replace(anchor, "", 1)))
    assert set(panel_card_data(_full_board())) - bound == {"meta_revision"}


# ---------------------------------------------------------------------------
# GATE TWO: every contract leaf reaches the card
# ---------------------------------------------------------------------------
#
# The drawer template's parity gate got this for free by comparing against the contract's
# own leaves. The card cannot: several leaves reach the page inside a composed sentence
# (``meta.age_seconds`` is part of ``meta_when``; ``progress.total`` is the denominator in
# ``progress_legend`` and in every column head), so a name comparison would report them
# as unrendered. Seeding the VALUE and looking for it on the other side proves the leaf
# travels, which is the thing actually worth proving.


#: One leaf per row: where it lives in a panel, the value to seed there, and the text
#: that must come out. The two differ only where the flattener FORMATS the leaf rather
#: than printing it -- an age of 7442 seconds reads "2h 4m", and looking for "7442" would
#: report a leaf that is rendered correctly as unrendered. So the expected text is
#: written out for those rows instead of derived, which is also the only way this table
#: says what the card does with each leaf.
_LEAF_MARKERS: tuple[tuple[str, Any, str], ...] = (
    ("contract_version", 4242, "4242"),
    ("lede", "MARK-lede", "MARK-lede"),
    ("since", "MARK-since", "MARK-since"),
    ("meta.name", "MARK-name", "MARK-name"),
    ("meta.captured_at", "MARK-captured", "MARK-captured"),
    # Formatted: seconds become a duration phrase.
    ("meta.age_seconds", 7442, "2h 4m"),
    # Read as a THRESHOLD, not printed: an age past it earns the word. Its two-sided
    # behaviour is pinned separately, because a constant present in the page is not
    # evidence that the comparison is made.
    ("meta.stale_after_seconds", 1, "stale"),
    ("meta.revision", 5150, "5150"),
    ("columns[].cards[].id", "MARK-id", "MARK-id"),
    ("columns[].cards[].sub", "MARK-sub", "MARK-sub"),
    ("columns[].cards[].of", "MARK-of", "MARK-of"),
    ("columns[].cards[].you", "MARK-you sentence", "MARK-you sentence"),
    ("progress.total", 6161, "6161"),
    ("progress.added_since", 3131, "3131"),
    ("progress.segments[].name", "MARK-band", "MARK-band"),
    ("progress.segments[].n", 9191, "9191"),
    ("stats[].v", "MARK-value", "MARK-value"),
    ("stats[].note", "MARK-note", "MARK-note"),
    ("omitted", 8181, "8181"),
)

#: The leaves a marker round-trip CANNOT prove, each with the test that proves it
#: instead. Both are read rather than printed as published, so a seeded value does not
#: come out the other side and looking for one would report a working leaf as unrendered.
#: Two entries, each named with its covering test, so this is a stated exception and not
#: a list anything inconvenient can be added to -- and the table above is asserted to be
#: exactly the remaining leaves, so adding a row here without a test is the change that
#: would have to be made deliberately.
_LEAF_READ_NOT_PRINTED: dict[str, str] = {
    # Reaches the card as the field NAME each tile's value binds by, never as a value.
    # Joined to the page's bindings and literal labels against a REAL board by:
    "stats[].k": "test_the_card_binds_a_value_and_a_note_for_every_stat_the_provider_emits",
    # Decides WHICH field group a column's cards land in -- matched against the closed
    # item states rather than printed, so a seeded name selects no group at all. Pinned
    # two-sided, cards following the name rather than the position, by:
    "columns[].name": "test_a_columns_cards_follow_its_name_not_its_position",
}


def _seed(panel: dict[str, Any], path: str, marker: Any) -> dict[str, Any]:
    """*panel* with the leaf at *path* set to *marker*, everywhere it occurs."""
    node: Any = panel
    parts = path.split(".")
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if part.endswith("[]"):
            key = part[:-2]
            children = node[key]
            assert children, f"nothing to seed under {path}: {key} is empty"
            for child in children:
                _seed(child, ".".join(parts[index + 1 :]), marker)
            return panel
        if last:
            node[part] = marker
            return panel
        node = node[part]
    return panel


def test_the_leaf_table_is_exactly_the_contracts_own_leaves() -> None:
    """The coverage table cannot go stale: a leaf added to the contract with no row here
    would never be checked, and a row for a leaf the contract dropped would test nothing
    while reading as coverage."""
    declared = _contract_leaves(PipelineBoardPanel)
    exempt = set(_LEAF_READ_NOT_PRINTED)
    listed = {path for path, _seed, _want in _LEAF_MARKERS} | exempt
    assert listed == declared, f"leaf table against the contract: {sorted(listed ^ declared)}"
    assert exempt <= declared, "an exception names a leaf the contract dropped"
    # The named covering test must EXIST. An exception pointing at a test that was
    # renamed or deleted is an unchecked leaf wearing a citation.
    own = _own_source()
    for leaf, covering in _LEAF_READ_NOT_PRINTED.items():
        assert f"def {covering}(" in own, f"{leaf} cites a test this file does not define"


def _own_source() -> str:
    from pathlib import Path

    return Path(__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "seed", "want"), _LEAF_MARKERS, ids=[p for p, _s, _w in _LEAF_MARKERS]
)
def test_every_contract_leaf_reaches_a_card_field(path: str, seed: Any, want: str) -> None:
    """Seed one leaf, and require its text out the other side.

    A leaf that reaches no field is a number the provider derives and nobody can read --
    the same defect the drawer's gate caught by name, caught here by value.
    """
    seeded = _seed(dict(_full_board()), path, seed)
    text = "\n".join(panel_card_data(seeded).values())  # type: ignore[arg-type]
    assert want in text, f"{path} reaches no field on the card"


def test_the_stale_threshold_is_compared_rather_than_printed() -> None:
    """Two-sided, because the marker row for this leaf looks for a word the page could
    carry for another reason. An age past the threshold is stale and an age under it is
    not, so the comparison is what the word reports."""
    late = _full_board()
    late["meta"]["age_seconds"] = 1000
    late["meta"]["stale_after_seconds"] = 100
    assert "stale" in panel_card_data(late)["meta_when"]
    early = _full_board()
    early["meta"]["age_seconds"] = 100
    early["meta"]["stale_after_seconds"] = 1000
    assert "stale" not in panel_card_data(early)["meta_when"]


def test_the_marker_probe_can_fail() -> None:
    """Control. A probe that always passes is not evidence about the card.

    A marker never written into the panel must NOT be found, otherwise the parametrized
    cases above would pass over a flattener that ignored the panel entirely.
    """
    text = "\n".join(panel_card_data(_full_board()).values())
    assert "MARK-never-seeded" not in text


def _contract_leaves(td: Any, prefix: str = "") -> set[str]:
    """Every LEAF field of a TypedDict tree, dotted, ``[]`` marking a list element."""
    import typing

    out: set[str] = set()
    for name, tp in typing.get_type_hints(td).items():
        path = f"{prefix}{name}"
        if isinstance(tp, type) and issubclass(tp, dict) and hasattr(tp, "__annotations__"):
            out |= _contract_leaves(tp, f"{path}.")
            continue
        if typing.get_origin(tp) is list:
            args = typing.get_args(tp)
            item = args[0] if args else None
            if isinstance(item, type) and issubclass(item, dict):
                out |= _contract_leaves(item, f"{path}[].")
                continue
        out.add(path)
    return out


# ---------------------------------------------------------------------------
# the host's caps, which drop a card WHOLE rather than degrading it
# ---------------------------------------------------------------------------


def test_the_producers_byte_cap_IS_the_hosts_own_object() -> None:
    """Not merely equal to the host's cap -- the same value, imported from it.

    Equality was the weaker assertion this once made, and it left the producer free to carry
    its own copy of the number the host enforces. A copy can drift, and a producer bounding a
    card against a stale limit builds cards the host drops whole. There is no FIELD-cap
    constant to compare because the host spells 24 inline and exports nothing; the field count
    is asserted instead by handing the real card to the real ``normalize_card``.
    """
    assert MAX_CARD_DATA_BYTES is MAX_DATA_BYTES


def test_the_card_page_fits_the_html_cap() -> None:
    assert len(_page().encode("utf-8")) <= MAX_HTML_BYTES


def test_a_full_board_fits_the_field_and_byte_caps() -> None:
    data = panel_card_data(_full_board())
    assert sum(len(k.encode()) + len(v.encode()) for k, v in data.items()) <= MAX_CARD_DATA_BYTES


def test_a_board_of_three_hundred_items_is_bounded_and_says_so() -> None:
    """A board may hold hundreds of items; the host drops a card over the cap WHOLE.

    So the rows are trimmed and the trim is COUNTED in the same text. A reader is never
    shown part of a board as if it were all of it -- which is the failure a silent prefix
    produces, and the reason the remainder carries its own denominator.
    """
    items = [_item(f"it_{i}", pr=10000 + i, worker=f"chat-{i:06d}") for i in range(300)]
    judgment = validate_judgment(
        {
            "you": {
                f"it_{i}": "a long sentence about what a person must do here " * 3
                for i in range(300)
            }
        }
    )
    panel = build_pipeline_board(
        _view(items),
        judgment,
        name="C",
        captured_at="2026-09-29T15:37:00Z",
        stale_after_seconds=900,
        now_epoch=1790696200.0,
    )
    data = panel_card_data(panel)
    assert len(data) == len(set(card_fields(_page()))), "trimming must not drop a field"
    assert sum(len(k.encode()) + len(v.encode()) for k, v in data.items()) <= MAX_CARD_DATA_BYTES
    assert re.search(r"\+\d+ of 300 not shown", data["column_0_rows"]), data["column_0_rows"]


# ---------------------------------------------------------------------------
# no blanks, and three states rather than two
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("empty", {}),
        ("no stats", {"stats": []}),
        ("stats not a list", {"stats": None}),
        ("meta not a mapping", {"meta": "nope"}),
        ("columns not a list", {"columns": "nope"}),
        ("progress unreadable", {"progress": {"total": "six", "segments": {"b": 1}}}),
    ],
)
def test_no_field_is_ever_blank(name: str, payload: dict[str, Any]) -> None:
    """The host binds a field it was not given to ``""``.

    So a blank is what a DROPPED field looks like, and must not also be what a real value
    looks like: on a page of counts the two are the same pixel and one of them is a lie.
    """
    data = panel_card_data(payload)  # type: ignore[arg-type]
    blank = sorted(key for key, value in data.items() if not value.strip())
    # ``contract_note`` is the ONE exemption, and only because it carries a NOTICE rather than a
    # value: an empty notice means there is nothing to notice. Every other field holds something
    # a reader counts, where a blank is indistinguishable from a zero. The exemption is named
    # here rather than dropped from the check, and the two cases that MUST speak are pinned by
    # test_a_board_from_another_contract_version_says_the_two_disagree and
    # test_an_unreadable_contract_version_says_so.
    assert [k for k in blank if k != "contract_note"] == [], f"{name}: blank field(s) {blank}"


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("empty", {}),
        ("hostile", {"stats": None, "meta": 3, "columns": (), "progress": []}),
    ],
)
def test_the_field_set_does_not_depend_on_the_payload(name: str, payload: dict[str, Any]) -> None:
    """Every field the page binds is written on EVERY board.

    Deriving the field set from what arrived is the one mistake this card cannot survive:
    a tile whose key went missing binds to the empty string and renders as a blank number
    beside two real ones, which reads as a zero.
    """
    assert set(panel_card_data(payload)) == set(card_fields(_page()))  # type: ignore[arg-type]


class _Hostile:
    """A published value whose ``str()`` raises -- which would take the whole card out."""

    def __str__(self) -> str:
        raise RuntimeError("this value cannot be read as text")

    __repr__ = __str__


def test_a_published_value_that_cannot_be_read_says_so_in_its_own_words() -> None:
    """THREE states, not two.

    A field nobody filled and a field holding an object are different facts: told the
    same way, a reader is sent looking for a publisher who never existed. The drawer's
    script distinguished them, and the card -- which has no script -- distinguishes them
    in the flattener instead.
    """
    data = panel_card_data({"lede": _Hostile(), "since": None})  # type: ignore[arg-type]
    assert data["lede"] == UNREADABLE
    assert data["since"] == NOT_SAID
    assert UNREADABLE != NOT_SAID


def test_an_unsaid_value_never_reaches_the_card_as_a_zero() -> None:
    """A board with no work entry has no AGE -- which is not an age of zero."""
    panel = build_pipeline_board(
        _view([]),
        validate_judgment({}),
        name="C",
        captured_at="2026-09-29T15:37:00Z",
        stale_after_seconds=900,
        now_epoch=1790696200.0,
    )
    panel["meta"]["age_seconds"] = "__unsaid__"  # type: ignore[typeddict-item]
    assert NOT_SAID in panel_card_data(panel)["meta_when"]
    assert "0s" not in panel_card_data(panel)["meta_when"]


# ---------------------------------------------------------------------------
# the VALUES a field may take, which parity cannot see
# ---------------------------------------------------------------------------


def test_the_card_binds_a_value_and_a_note_for_every_stat_the_provider_emits() -> None:
    """A tile's LABEL is literal text in the page, not a bound field.

    ``stats[].k`` is a closed provider vocabulary, so a label is the same on every board
    ever published and belongs in the inert half. The key still reaches the card as the
    field NAME its value binds by -- and that is the join this asserts, against a real
    board rather than against the constant, so renaming a metric reddens here instead of
    leaving a tile bound to nothing.
    """
    emitted = [stat["k"] for stat in _full_board()["stats"]]
    assert emitted == list(BOARD_STAT_KEYS), "the provider's tiles against the closed set"
    bound = set(card_fields(_page()))
    page = _page()
    for key in emitted:
        assert f"stat_{key}" in bound, f"no value bound for the {key} tile"
        assert f"stat_{key}_note" in bound, f"no note bound for the {key} tile"
        assert f">{key}</span>" in page, f"the {key} tile has no literal label on the page"


def test_the_column_field_groups_are_the_closed_item_states() -> None:
    """One field group per state, by POSITION over the closed vocabulary.

    Keyed on what arrived instead, a board missing a column shifts every later column's
    field one place and prints one state's items under another state's heading.
    """
    assert tuple(BOARD_COLUMN_NAMES) == tuple(WORK_ITEM_STATES)
    bound = set(card_fields(_page()))
    for index in range(len(WORK_ITEM_STATES)):
        assert f"column_{index}_head" in bound
        assert f"column_{index}_rows" in bound
    assert f"column_{len(WORK_ITEM_STATES)}_head" not in bound, "a column no state fills"


def test_a_board_missing_a_column_does_not_shift_the_others() -> None:
    """Indexing the arrived list by position looks equivalent, because the provider emits
    the four states in order. A board missing one column then shifts every later column's
    field one place and prints one state's items under another state's heading -- a WRONG
    board rather than an incomplete one, with nothing on the page saying so."""
    panel = _full_board()
    removed = panel["columns"].pop(1)
    data = panel_card_data(panel)
    for index, state in enumerate(BOARD_COLUMN_NAMES):
        assert data[f"column_{index}_head"].startswith(state), data[f"column_{index}_head"]
    gone = removed["cards"][0]["id"]
    assert gone not in "\n".join(data.values()), "a removed column's cards appear elsewhere"


def test_a_columns_cards_follow_its_name_not_its_position() -> None:
    """The covering test for ``columns[].name``: the name decides the field group.

    Two columns swapped in the payload must swap on the card too. Placed by position
    instead, each column's cards would print under the other's heading and every count
    would still add up -- the failure a marker round-trip cannot see.
    """
    panel = _full_board()
    panel["columns"][0], panel["columns"][1] = panel["columns"][1], panel["columns"][0]
    data = panel_card_data(panel)
    # ``it_1`` is the accepted item, which is the second closed state.
    assert "#14689" in data["column_1_rows"], data["column_1_rows"]
    assert "#14583" in data["column_0_rows"], data["column_0_rows"]


def test_every_count_on_the_card_carries_its_denominator() -> None:
    """A bare count says nothing about the size of the thing it counts."""
    data = panel_card_data(_full_board())
    for index in range(len(BOARD_COLUMN_NAMES)):
        assert re.search(r"\d+ of \d+", data[f"column_{index}_head"]), data[f"column_{index}_head"]
    assert re.search(r"\d+ of \d+ accepted", data["progress_legend"]), data["progress_legend"]
    assert data["meta_revision"] == "revision 3", data["meta_revision"]


def test_the_card_reads_the_version_the_contract_writes() -> None:
    """Otherwise the disclosure inverts: the shipped card would declare a mismatch on
    every board the shipped provider publishes, and a REAL mismatch -- an operator
    override serving an older page -- would be indistinguishable from that noise."""
    # SILENT when they agree: the healthy case is the one field on this card that is empty,
    # because "contract" is a word about this code's internals and a permanent header phrase no
    # reader can parse costs more than the disclosure it protects. The guard moved here.
    assert panel_card_data(_full_board())["contract_note"] == ""


def test_a_board_from_another_contract_version_says_the_two_disagree() -> None:
    panel = _full_board()
    panel["contract_version"] = CONTRACT_VERSION + 1
    note = panel_card_data(panel)["contract_note"]
    assert str(CONTRACT_VERSION + 1) in note and "may be missing" in note


def test_an_unreadable_contract_version_says_so() -> None:
    """The second case that must never be silent.

    ``contract_note`` is empty when the versions agree, so the guard against a DROPPED
    disclosure is these two tests rather than a permanent phrase on the page. Something IS in
    the field and it is not a version, so the two halves cannot be compared at all -- a
    different fact from a version that merely differs, and it keeps its own words.
    """
    panel = _full_board()
    panel["contract_version"] = {"nested": "object"}  # type: ignore[typeddict-item]
    note = panel_card_data(panel)["contract_note"]
    assert note, "an unreadable contract version said nothing"
    assert UNREADABLE in note and str(CONTRACT_VERSION) in note


def test_dropped_entries_are_stated_either_way() -> None:
    """Zero says "all entries shown" rather than nothing.

    A field bound to the empty string is exactly what a dropped disclosure looks like, so
    the agreeing case needs words of its own -- and an item missing from a board is not
    recoverable by whoever reads the board.
    """
    assert "2 entries not shown" in panel_card_data(_full_board())["omitted"]
    quiet = _full_board()
    quiet["omitted"] = 0
    assert panel_card_data(quiet)["omitted"] == "all entries shown"


# ---------------------------------------------------------------------------
# the page is inert, and survives the host's own sanitizer
# ---------------------------------------------------------------------------


def test_the_card_page_declares_no_control_and_no_script() -> None:
    """The host strips every one of these, so a page authoring one ships a layout with a
    hole in it and an author who believes the control exists."""
    used = _scan(_page()).tags & _FORBIDDEN_TAGS
    assert not used, f"the page authors control/script tags the host strips: {sorted(used)}"


def test_the_card_page_points_nowhere_outside_itself() -> None:
    """``src`` always reaches outside; ``href`` does unless it is a same-document
    fragment. The host removes both, and CSP does not govern navigation."""
    assert not _scan(_page()).outbound


def test_the_card_page_avoids_the_properties_the_host_deletes() -> None:
    """``dashboardDocument.ts`` deletes these in card mode, so a label in one of them is
    removed and leaves its row unlabelled with nothing saying why.

    Textual, and deliberately: no Python test runs that TypeScript. The control below
    keeps an empty result from reading as a pass over a page that lost its styles.
    """
    style = re.search(r"<style>(.*?)</style>", _page(), re.DOTALL)
    assert style, "the page has no style block -- this check would pass over a blank page"
    css = style.group(1)
    for prop in _STRIPPED_PROPERTIES:
        assert not re.search(
            rf"(?<![-\w]){re.escape(prop)}\s*:", css
        ), f"{prop} is deleted in card mode"
    assert "data-dashboard-field" in _page(), "control: the page still binds fields"


def test_no_local_token_shares_a_name_with_the_host_var_it_reads() -> None:
    """``--x: var(--x, fallback)`` is a SELF REFERENCE: invalid at computed-value time, so
    every shorthand using it collapses to nothing, silently. The hairline the whole layout
    rests on would not exist and a declaration count would still read as correct."""
    for name, inner in re.findall(r"(--[\w-]+)\s*:\s*var\(\s*(--[\w-]+)", _page()):
        assert name != inner, f"{name} reads a host var of its own name"


def test_the_real_host_normalizer_accepts_the_card() -> None:
    """THE end-to-end check: the host's own validator, not a restatement of its rules.

    ``normalize_card`` returning ``None`` is how a card is dropped whole, so a page and a
    flattener that satisfy every gate above and fail here would ship a panel that is
    simply never there.
    """
    card = normalize_card({"html": _page(), "data": panel_card_data(_full_board())})
    assert card is not None, "the host refused the card"
    assert set(card["data"]) == set(card_fields(_page()))


def test_every_field_name_is_one_the_host_accepts() -> None:
    """The host's own pattern admits ``[a-zA-Z][a-zA-Z0-9_-]{0,47}`` -- NO DOTS.

    Which is why the flattening rule spells a nested path with ``_``: a dotted name is
    not a name the host merely dislikes, it is one ``normalize_card`` refuses, taking the
    whole card with it.
    """
    for name in card_fields(_page()):
        assert _FIELD_NAME.fullmatch(name), f"{name} is not a binding the host accepts"
        assert "." not in name, f"{name} carries a dot, which the host's pattern refuses"


def test_the_card_page_ships_in_the_wheel_and_the_sdist() -> None:
    """A page missing from the build renders nothing while every gate above stays green.

    Both entries, because the sdist is built from the manifest and the wheel from the
    sdist: an entry in one alone ships a half fix that works perfectly from a checkout.
    """
    root = card_template_path().parents[3]
    directory = card_template_path().parent.name
    assert f"{directory}/*.html" in (root / "setup.cfg").read_text(encoding="utf-8")
    assert f"src/kiro_crew/{directory} *.html" in (root / "MANIFEST.in").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the publisher's own text is bounded too, not just the row lists
# ---------------------------------------------------------------------------
#
# ``panel_publish`` caps a whole payload at 64 KB -- sixteen times this card's entire data
# budget -- and gates ``lede`` and each ``notes`` value only on being a string. The row
# ladder can shorten only rows, so one long sentence in a publisher field makes the card
# oversized at EVERY row limit: the ladder retreats to one row per column, still does not
# fit, and ``normalize_card`` refuses the card whole. A refused card is not a smaller card,
# it is NO card -- and with the cost opt-in off the reader is told "disabled", which names
# the wrong cause; with a previous card in place the reader is served last round's counts
# marked fresh, which is worse.


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("a long lede", {"lede": "x" * 4000}),
        ("a long metric gloss", {"stats": [{"k": "items", "v": "6", "note": "y" * 4000}]}),
        (
            "a long action sentence",
            {
                "columns": [
                    {
                        "name": "open",
                        "cards": [{"id": "#1", "sub": "s", "of": "1/1", "you": "z" * 4000}],
                    }
                ]
            },
        ),
        (
            "all three at once",
            {
                "lede": "x" * 4000,
                "stats": [{"k": key, "v": "1", "note": "y" * 2000} for key in BOARD_STAT_KEYS],
                "columns": [
                    {
                        "name": "open",
                        "cards": [
                            {"id": f"#{i}", "sub": "s" * 60, "of": "1/1", "you": "z" * 900}
                            for i in range(40)
                        ],
                    }
                ],
            },
        ),
    ],
)
def test_a_long_publisher_field_still_produces_a_card_the_host_accepts(
    name: str, payload: dict[str, Any]
) -> None:
    """The whole point of the bound: a card, not a refusal."""
    data = panel_card_data(payload)  # type: ignore[arg-type]
    size = sum(len(k.encode()) + len(v.encode()) for k, v in data.items())
    assert size <= MAX_CARD_DATA_BYTES, f"{name}: {size} bytes"
    assert normalize_card({"html": _page(), "data": data}) is not None, f"{name}: host refused it"
    assert set(data) == set(card_fields(_page())), f"{name}: the field set changed"


def test_a_clipped_value_says_that_it_was_clipped() -> None:
    """A bare truncation reads as the publisher's own sentence ending there.

    So a reader cannot tell a trimmed sentence from a complete one, which is the same
    ambiguity a silent row prefix creates and the reason the row retreat states its count.
    """
    data = panel_card_data({"lede": "x" * 4000})  # type: ignore[arg-type]
    assert data["lede"].endswith("[trimmed]")
    assert data["lede"].startswith("xxx")


def test_a_short_publisher_field_is_left_exactly_as_written() -> None:
    """Control. A clip that fires on every value would be indistinguishable above."""
    lede = "Six items this round; one needs a person."
    assert panel_card_data({"lede": lede})["lede"] == lede  # type: ignore[arg-type]
    assert "[trimmed]" not in panel_card_data(_full_board())["lede"]


def test_the_words_for_a_gap_are_never_themselves_clipped() -> None:
    """``not said`` and ``unreadable`` are this module's own words, not publisher text."""
    data = panel_card_data({})  # type: ignore[arg-type]
    assert data["lede"] == NOT_SAID
    assert panel_card_data({"lede": _Hostile()})["lede"] == UNREADABLE  # type: ignore[arg-type]


def test_a_clip_never_cuts_a_character_in_half() -> None:
    """Slicing UTF-8 mid-sequence yields text that is not decodable, and the card's data is
    JSON -- so the cut is on a character boundary and the result must survive a round trip
    through the encoding the host serializes it with."""
    for char in ("\u00e9", "\u4e2d", "\U0001f680"):  # 2, 3 and 4 UTF-8 bytes
        data = panel_card_data({"lede": char * 3000})  # type: ignore[arg-type]
        text = data["lede"]
        assert text.encode("utf-8").decode("utf-8") == text
        assert len(text.encode("utf-8")) <= 400, f"{char!r}: {len(text.encode())} bytes"
        assert text.endswith("[trimmed]")


# ---------------------------------------------------------------------------
# the website fixture is generated, so something has to notice when it goes stale
# ---------------------------------------------------------------------------


def test_the_website_fixture_matches_what_the_generator_produces() -> None:
    """THE CALLER for the fixture script's ``--check``.

    ``website/src/test/fixtures/pipelineBoardCard.json`` holds the page and two real data sets,
    written by ``scripts/pipeline_board_card_fixture.py`` so the website test asserts on the
    bytes the gateway publishes rather than a sample someone typed. Nothing under ``.github/``
    runs that script, so the claim that a stale fixture fails rather than passing quietly had
    no mechanism behind it: edit the page or the flattener without regenerating, and the
    website test keeps passing against the OLD card while every Python gate stays green.

    This is that mechanism, in the suite that already runs. It re-runs the generator's own
    ``--check`` mode, so there is one definition of "current" rather than a second copy of the
    comparison here.
    """
    root = card_template_path().parents[3]
    script = root / "scripts" / "pipeline_board_card_fixture.py"
    assert script.is_file(), f"the fixture generator is missing: {script}"
    done = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        cwd=root,
        stdin=subprocess.DEVNULL,
        timeout=120,
        # The repo's own splat, not a bare ``text=True``: text mode with no ``encoding=``
        # decodes the child's output with the Windows ANSI code page, which the backend lint
        # gate refuses -- and this assertion PRINTS that output in its failure message, so a
        # mis-decoded byte would land in the diagnostic a reader acts on.
        **UTF8_TEXT,
    )
    assert done.returncode == 0, (
        "the website fixture no longer matches the page and flattener on this branch -- "
        f"regenerate it with `python {script.relative_to(root)}`.\n{done.stdout}{done.stderr}"
    )


def test_that_check_can_actually_fail() -> None:
    """Control. A `--check` that answered 0 unconditionally would make the test above a
    decoration, and the fixture could still go stale unnoticed."""
    root = card_template_path().parents[3]
    fixture = root / "website" / "src" / "test" / "fixtures" / "pipelineBoardCard.json"
    original = fixture.read_bytes()
    try:
        fixture.write_bytes(original.replace(b'"lede"', b'"lede_moved"', 1))
        done = subprocess.run(
            [sys.executable, str(root / "scripts" / "pipeline_board_card_fixture.py"), "--check"],
            capture_output=True,
            cwd=root,
            stdin=subprocess.DEVNULL,
            timeout=120,
            **UTF8_TEXT,
        )
        assert done.returncode != 0, "--check passed over a fixture that does not match"
    finally:
        fixture.write_bytes(original)
