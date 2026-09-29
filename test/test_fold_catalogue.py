"""The committed fold catalogue equals the one the code generates.

An agent authoring a dashboard template picks a fold, then writes a contract naming that
fold's fields with their types. It has no way to learn either except from the catalogue,
so a stale catalogue is worse than no catalogue: it sends the author to write a provider
reading a field no fold produces, and nothing downstream can tell that from a typo.

These tests are the gate that makes staleness impossible, plus the generator's own
self-test. Both are needed: the equality check proves the committed bytes are current,
and the self-test proves the check can observe a difference at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from skill_script_helpers import load_skill_script

from kiro_crew.crew_log.projection import (
    _FOLDS,
    FOLD_NAMES,
    INTERNAL_PROJECTION_NAMES,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "fold_catalogue.py"
SKILL_DIR = ROOT / "src" / "kiro_crew" / "builtin_skills" / "kirocrew-dev" / "dashboard-template"
JSON_PATH = SKILL_DIR / "folds.json"
MARKDOWN_PATH = SKILL_DIR / "FOLDS.md"


@pytest.fixture(scope="module")
def generator() -> Any:
    return load_skill_script("fold_catalogue_generator", GENERATOR)


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


class TestTheCommittedCatalogueIsCurrent:
    def test_it_matches_what_the_code_generates(self, generator: Any) -> None:
        """THE gate. Byte equality, both artifacts, no tolerance.

        The remedy is in the failure message rather than left to the reader, because the
        person who trips this is usually not the person who added the fold.
        """
        drifted = generator.check()
        assert not drifted, (
            f"the committed fold catalogue is stale: {drifted}. Regenerate it with "
            "`python3 scripts/fold_catalogue.py --write` and commit the result."
        )

    def test_the_generators_own_selftest_passes(self, generator: Any) -> None:
        """Proves the check above can SEE a drift.

        Without this, "nothing differs" reads the same way whether the comparison works
        or is broken -- and a broken comparison is the one failure mode that makes every
        other assertion here vacuous.
        """
        assert generator.selftest() == 0


class TestTheCatalogueCoversTheKernel:
    def test_it_names_exactly_the_folds_the_product_keeps(self, committed: dict[str, Any]) -> None:
        names = [fold["name"] for fold in committed["folds"]]
        assert names == list(FOLD_NAMES), (
            "the catalogue and the projection registry disagree about which folds exist: "
            f"{sorted(set(names) ^ set(FOLD_NAMES))}"
        )

    def test_an_internal_fold_is_marked_as_one(self, committed: dict[str, Any]) -> None:
        """A fold no panel draws is still listed, and still labelled.

        Hiding it would make the catalogue disagree with the registry; listing it
        unlabelled would invite a template to draw from a fold with no advertised reader.
        """
        by_name = {fold["name"]: fold for fold in committed["folds"]}
        for name in INTERNAL_PROJECTION_NAMES:
            assert by_name[name]["advertised"] is False, name
        advertised = [f["name"] for f in committed["folds"] if f["advertised"]]
        assert advertised, "no advertised fold -- the probe is broken, not the registry"

    def test_every_fold_says_what_it_answers(self, committed: dict[str, Any]) -> None:
        for fold in committed["folds"]:
            assert fold["answers"].strip(), f"{fold['name']} has no sentence"
            assert fold["fields"], f"{fold['name']} lists no fields"

    def test_a_fold_moved_by_every_entry_says_so_as_null(self, committed: dict[str, Any]) -> None:
        """``affects: null`` means every entry moves it.

        An empty list would read as "no entry moves this fold", which is the opposite
        fact and would tell an author the fold never updates.
        """
        status = next(fold for fold in committed["folds"] if fold["name"] == "status")
        assert status["affects"] is None
        work = next(fold for fold in committed["folds"] if fold["name"] == "work")
        assert work["affects"] == ["work/recorded"]

    def test_no_field_is_typed_as_the_absence_marker(self, committed: dict[str, Any]) -> None:
        """``NoneType`` is never a useful answer: it is what an empty fold shows, not
        what the field holds. Such a field is reported as optional with a real type, or
        as ``unknown``."""
        for fold in committed["folds"]:
            for row in fold["fields"]:
                assert row["type"] != "NoneType", f"{fold['name']}.{row['name']}"

    def test_an_unknown_type_is_always_optional(self, committed: dict[str, Any]) -> None:
        """The two are the same finding seen twice: the type is unknown BECAUSE the empty
        fold left the field null. An unknown on a non-optional row would mean the
        generator lost track of which branch it took."""
        for fold in committed["folds"]:
            for row in fold["fields"]:
                if row["type"] == "unknown":
                    assert row["optional"] is True, f"{fold['name']}.{row['name']}"


class TestANullFieldIsNeverGivenAGuessedType:
    """A null field on an empty fold reports ``unknown``, and nothing cleverer.

    The clever version -- look the field NAME up in the entry-type registry -- shipped
    first and was wrong: a rendered name is owned by no one entry type, so ``status``
    reported ``previous: dict`` and ``turn: int``. A confident wrong row is worse than a
    missing one, because a reader cannot tell which rows to trust. These tests redden if
    any inference comes back.
    """

    def test_every_optional_row_in_the_committed_catalogue_is_unknown(
        self, committed: dict[str, Any]
    ) -> None:
        guessed = [
            f"{fold['name']}.{row['name']}={row['type']}"
            for fold in committed["folds"]
            for row in fold["fields"]
            if row["optional"] is True and row["type"] != "unknown"
        ]
        assert guessed == [], f"a type was inferred for a null field: {guessed}"

    def test_at_least_one_optional_row_exists_so_the_check_is_not_vacuous(
        self, committed: dict[str, Any]
    ) -> None:
        optional = [
            f"{fold['name']}.{row['name']}"
            for fold in committed["folds"]
            for row in fold["fields"]
            if row["optional"] is True
        ]
        assert optional, "no null fields at all: the rule above is asserting nothing"

    def test_the_row_builder_itself_refuses_to_type_a_null(
        self, generator: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Held separately from the committed artifact: this one reddens even if a future
        kernel happens to leave no field null."""
        monkeypatch.setattr(generator, "_rendered", lambda name: {"n": None, "said": "x"})
        assert generator._field_rows("probe") == [
            {"name": "n", "type": "unknown", "optional": True},
            {"name": "said", "type": "str", "optional": False},
        ]


class TestOnlyFoldsDeclaringABinderAreBound:
    """``_rendered`` binds a slot to the folds that DECLARE a binder, not to every
    slot-keyed fold -- today one fold of the four declares one.

    Both directions matter. Binding a fold that declares no binder is impossible; NOT
    binding one that does would render a board-wide fold against no board, and it would
    still return a dict, so nothing else in the suite would notice.
    """

    def test_the_binder_set_is_derived_from_the_kernel_and_is_not_empty(self) -> None:
        binders = {name for name, fold in _FOLDS.items() if fold.bind_slot is not None}
        assert binders, "no fold declares a slot binder: the bind branch in _rendered is dead"
        for name in binders:
            assert _FOLDS[name].bind_slot is not None

    def test_rendered_binds_exactly_those_folds(self, generator: Any) -> None:
        bound: list[str] = []
        for name, fold in _FOLDS.items():
            original = fold.bind_slot
            if original is None:
                continue

            def spy(state: Any, slot: str, _name: str = name, _inner: Any = original) -> Any:
                bound.append(_name)
                return _inner(state, slot)

            object.__setattr__(fold, "bind_slot", spy)
            try:
                generator._rendered(name)
            finally:
                object.__setattr__(fold, "bind_slot", original)
        expected = sorted(name for name, fold in _FOLDS.items() if fold.bind_slot is not None)
        assert sorted(bound) == expected

    def test_a_fold_with_no_binder_still_renders(self, generator: Any) -> None:
        unbound = [name for name, fold in _FOLDS.items() if fold.bind_slot is None]
        assert unbound, "every fold declares a binder: the docstring's other half is stale"
        for name in unbound:
            assert isinstance(generator._rendered(name), dict)


class TestTheScaffoldReadsTheCatalogue:
    """The scaffold must not carry its own copy of the fold list.

    Two lists are two things to update, and the one that gets forgotten is the one that
    refuses a fold that exists.
    """

    def test_the_scaffolds_fold_choices_come_from_the_committed_json(
        self, committed: dict[str, Any]
    ) -> None:
        scaffold = load_skill_script(
            "dashboard_template_scaffold_catalogue", SKILL_DIR / "scripts" / "scaffold.py"
        )
        assert scaffold.folds() == tuple(fold["name"] for fold in committed["folds"])

    def test_an_unreadable_catalogue_is_a_named_error_not_a_fallback_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fallback list would be reached exactly when the catalogue is missing, which
        is the one moment a guess is least likely to be right."""
        scaffold = load_skill_script(
            "dashboard_template_scaffold_missing", SKILL_DIR / "scripts" / "scaffold.py"
        )
        monkeypatch.setattr(scaffold, "CATALOGUE_PATH", tmp_path / "absent.json")
        with pytest.raises(scaffold.ScaffoldError, match="cannot read the fold catalogue"):
            scaffold.folds()

    def test_an_empty_catalogue_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / "folds.json"
        empty.write_text(json.dumps({"catalogue_version": 1, "folds": []}), encoding="utf-8")
        scaffold = load_skill_script(
            "dashboard_template_scaffold_empty", SKILL_DIR / "scripts" / "scaffold.py"
        )
        monkeypatch.setattr(scaffold, "CATALOGUE_PATH", empty)
        with pytest.raises(scaffold.ScaffoldError, match="lists no folds"):
            scaffold.folds()


class TestTheSkillPointsAtTheCatalogue:
    def test_the_skill_body_names_the_generated_file(self) -> None:
        """The skill must send the reader to the generated catalogue rather than carrying
        its own table, which is the copy that goes stale."""
        body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "FOLDS.md" in body
        assert "fold_catalogue.py" in body

    def test_the_markdown_warns_against_editing_it(self) -> None:
        text = MARKDOWN_PATH.read_text(encoding="utf-8")
        assert "GENERATED by scripts/fold_catalogue.py" in text
        assert "--write" in text
