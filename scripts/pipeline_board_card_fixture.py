#!/usr/bin/env python3
"""Write the pipeline board card's real bytes for the website's render test.

The Python gates read the card page as text. Whether the HOST puts the written values on
the page is a question about ``dashboardDocument.ts``, so that half is tested in the
website suite -- and it must test the bytes the gateway actually publishes rather than a
sample someone typed into a TypeScript file.

This runs the real provider (``build_pipeline_board``) and the real flattener
(``panel_card_data``) over two boards and writes the page plus both data sets to
``website/src/test/fixtures/pipelineBoardCard.json``. The website test re-asserts parity
against the page's own bindings, so a fixture that has gone stale fails there instead of
passing quietly.

Regenerate with::

    PYTHONPATH=src python scripts/pipeline_board_card_fixture.py

``--check`` rewrites nothing and exits non-zero when the file on disk differs, which is how
CI can tell a stale fixture from a current one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiro_crew.pipeline_board_contract import (  # noqa: E402
    build_pipeline_board,
    card_template_path,
    panel_card_data,
    validate_judgment,
)

OUT = ROOT / "website" / "src" / "test" / "fixtures" / "pipelineBoardCard.json"


def _item(
    ident: str,
    *,
    state: str,
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


def _full() -> dict[str, str]:
    """A board with something in every column, an action, a tally and a metric gloss."""
    view = {
        "conductor": {
            "schema": 1,
            "slot_key": "chat-1875",
            "goal": "the board becomes the first dashboard template",
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
            _item("it_0", state="open", pr=14583, worker="chat-1875", round_=3),
            _item("it_1", state="open", pr=14966, worker="chat-2176", round_=3),
            _item("it_2", state="accepted", pr=14689),
            _item("it_3", state="accepted", pr=14111),
            _item("it_4", state="rejected", worker="chat-2301"),
            _item("it_5", state="abandoned", pr=12046),
        ],
        "omitted": 2,
    }
    judgment = validate_judgment(
        {
            "lede": "Six items this round; one needs a person before CI can go green.",
            "you": {
                "it_0": "push the rebase once the base fix lands on main",
                "it_4": "decide whether the acceptance bar itself was wrong",
            },
            "notes": {
                "items": "items this board has created over its life",
                "entries": "work entries the crew log holds for it",
            },
            "checks": {"it_0": "41/47", "it_2": "161/161"},
        }
    )
    panel = build_pipeline_board(
        view,  # type: ignore[arg-type]
        judgment,
        # NOT the conductor crew's real display name, whose first word is deliberately
        # joined. This fixture is written to JSON, which carries no comment syntax and so
        # cannot carry the ``brand-ok`` marker the brand gate needs -- the joined spelling
        # would be an unexplainable product-name misspelling in a checked-in file. The
        # slugification that makes the joined form load-bearing belongs to the drawer's
        # template selection and is asserted against the real constant in
        # ``test_pipeline_board_contract_parity``; what this fixture exercises is
        # RENDERING, for which the name is just text of a realistic length.
        name="Pipeline Conductor",
        captured_at="2026-09-29 15:37 UTC",
        stale_after_seconds=900,
        now_epoch=1790696420.0,
    )
    return panel_card_data(panel)


def _hostile() -> dict[str, str]:
    """A payload from before this contract existed, plus values that cannot be read.

    Reachable rather than theoretical: a record stored in the free shape reaches the
    flattener untouched, and the operator override directory can serve an older page
    against a newer provider.
    """

    class Unreadable:
        def __str__(self) -> str:
            raise RuntimeError("this value cannot be read as text")

        __repr__ = __str__

    payload = {
        "contract_version": {"nested": "object"},
        "lede": Unreadable(),
        "since": None,
        "meta": {
            "name": ["a", "list"],
            "captured_at": "2026-09-29 15:37 UTC",
            "age_seconds": "half an hour",
            "stale_after_seconds": 900,
            "revision": Unreadable(),
        },
        "columns": [
            {"name": "open", "cards": [{"id": Unreadable(), "sub": {}, "of": 7, "you": 3}]}
        ],
        "progress": {"total": "six", "added_since": True, "segments": {"open": 1}},
        "stats": [{"k": "items", "v": Unreadable(), "note": []}, "junk"],
        "omitted": -4,
    }
    return panel_card_data(payload)  # type: ignore[arg-type]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero if stale")
    args = parser.parse_args()

    document = {
        # NEWLINES NORMALISED. The fixture JSON is pinned to LF by .gitattributes while the
        # page it embeds carried the platform's ending, so a CRLF checkout made the
        # generated bytes differ from the committed ones and the staleness check failed on
        # Windows for a reason that is not staleness. The page is pinned to LF now too;
        # this keeps the comparison true however a tree was checked out.
        "html": card_template_path().read_text(encoding="utf-8").replace("\r\n", "\n"),
        "boards": {"full": _full(), "hostile": _hostile()},
    }
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"stale fixture: {OUT} -- regenerate with {Path(__file__).name}")
            return 1
        print(f"fixture current: {OUT}")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text)} bytes)")
    for name, data in document["boards"].items():  # type: ignore[union-attr]
        blank = sorted(k for k, v in data.items() if not v.strip())
        print(f"  {name}: {len(data)} fields, blank={blank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
