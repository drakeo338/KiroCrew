"""The gates on a template, in one implementation both the tests import.

``mypy`` checks the provider: a :class:`~typing.TypedDict` in, the contract out,
required keys and all. It cannot see inside an html file, so the html end of the
agreement needs a reader of its own. This is that reader, plus the two structural
rules a contract must satisfy.

Written with the stdlib html parser rather than a regular expression, deliberately.
An attribute scan by regex mis-reads the cases that matter -- an attribute inside a
comment, a value quoted with the other quote character, a tag name in upper case --
and each of those makes the gate QUIETLY incomplete, which is worse than a gate that
fails: a field the extractor missed reads as a field the html does not use, so the
equality assertion passes over a real mismatch.

Every function here refuses rather than returning a partial answer. A partial answer
is indistinguishable from a template that genuinely uses fewer fields, and that is
the one direction an equality assertion cannot catch on its own.
"""

from __future__ import annotations

import typing
from html.parser import HTMLParser
from typing import Any, Final

__all__ = [
    "CONTROL_TAGS",
    "ExtractionRefused",
    "contract_keys",
    "control_tags_used",
    "denominator_gaps",
    "html_fields",
    "dropped_text_attributes",
    "outbound_references",
]

#: The binding attribute the host reads. One attribute, one field, one value.
_BINDING: Final[str] = "data-dashboard-field"

#: Attributes that reach OUTSIDE the page. ``src`` and ``srcset`` always do; ``href``
#: does unless it is a same-document fragment. Matched on the attribute's local name, so
#: an SVG ``xlink:href`` is read as an ``href`` rather than slipping past a check on the
#: plain spelling.
_OUTBOUND: Final[frozenset[str]] = frozenset({"src", "srcset", "href"})

#: Attributes the host removes from a card because they render as text its own scan
#: cannot read -- a broken image's ``alt``, a hover ``title``, a list marker's ``start``.
#: They reach nothing outside the page, so they are not outbound; they are simply
#: deleted, which makes authoring one a way to put a fact where the reader never sees it.
_DROPPED_TEXT_ATTRS: Final[frozenset[str]] = frozenset({"alt", "title", "start", "value"})

#: Elements that cannot hold text. The host binds by assigning ``textContent``, which on
#: one of these is accepted and renders nothing, so a binding here is a field that reads
#: as present and shows nothing -- the same silence as a missing binding, harder to see.
_VOID_TAGS: Final[frozenset[str]] = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

#: Tags that make a page a control surface, refused for two different reasons.
#:
#: Most of them the host strips, so a template authoring one ships a layout with a hole
#: in it and an author who believes the button exists. Three -- ``label``, ``fieldset``
#: and ``option`` -- the host KEEPS, and this package refuses them on its own account:
#: they render as the furniture of a control, so a page carrying them tells a reader an
#: action is available where none is. :func:`control_tags_used` does not distinguish the
#: two, because the answer to both is the same, but the claim that the host removes every
#: one of these would be false and ``TestTheControlTagsSplit`` holds the real division.
#:
#: A dashboard template states facts and offers no actions: a decision goes through the
#: product's own question and approval surfaces, which are outside the rendered frame and
#: carry the identity of the session that owns them.
CONTROL_TAGS: Final[frozenset[str]] = frozenset(
    {
        "form",
        "input",
        "button",
        "textarea",
        "select",
        "option",
        "label",
        "fieldset",
        "script",
        "iframe",
        "object",
        "embed",
        "link",
        "meta",
        "base",
        "template",
        "noscript",
    }
)

#: Tags the host RENDERS but its binder steps over, so a binding on one never receives
#: text. Read out of the binder itself, which skips ``style`` before it reads the field
#: name: writing provider text into live CSS is its own deliberate refusal.
_BINDING_SKIPPED_TAGS: Final[frozenset[str]] = frozenset({"style"})

#: Tags the host's sanitizer removes outright and that :data:`CONTROL_TAGS` does not
#: already name. A binding on one leaves with the element that carried it.
#:
#: ``foreignobject`` is here rather than with the controls because it is not one: it is an
#: SVG container for html, and the sanitizer's default profile drops it, so a binding
#: parked inside looks like ordinary layout and never renders.
_BINDING_STRIPPED_TAGS: Final[frozenset[str]] = frozenset(
    {"frame", "applet", "animate", "set", "foreignobject"}
)

#: Where the binder looks. It queries the BODY, so a binding the source puts in the head
#: is parsed, kept, and never visited.
_HEAD: Final[str] = "head"

#: The suffix a count's denominator field carries. ``failed`` needs ``failed_of``.
_DENOMINATOR_SUFFIX: Final[str] = "_of"


class ExtractionRefused(Exception):
    """The template is not something this reader can answer about.

    Raised instead of returning what it managed to find. A short set reads as a
    template using fewer fields, which is exactly the mismatch the gate exists to
    catch, so an unreadable page has to be loud.
    """


class _FieldScanner(HTMLParser):
    """Collects every ``data-dashboard-field`` value, and the tags used.

    ``convert_charrefs`` stays on (the default) because the value is an attribute and
    the host reads the DECODED attribute, so decoding here matches what it binds.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fields: list[str] = []
        self.tags: set[str] = set()
        self.outbound: set[tuple[str, str]] = set()
        self.dropped_text: set[tuple[str, str]] = set()
        self.empty_bindings = 0
        self.unbindable: list[str] = []
        self._head_depth = 0
        #: Bound elements still open around the cursor, innermost last. The binder walks
        #: matches in document order and assigns ``textContent``, which replaces the
        #: outer element's children -- so an inner bound element is detached before the
        #: loop reaches it, and its value is written to a node outside the document.
        self._open_bound: list[str] = []
        #: Depth inside each open bound element, so the right one closes.
        self._open_depth: list[int] = []

    def _unreachable(self, tag: str) -> str | None:
        """Why the binder will never write to this element, or ``None`` if it will.

        Recording the binding anyway would make the equality gate PASS on a page whose
        cell stays blank at every render -- the precise failure this whole package is
        built to make impossible, reached through its own reader.
        """
        if self._head_depth:
            return f"inside <{_HEAD}>, and the binder queries the body"
        if tag in _BINDING_SKIPPED_TAGS:
            return f"on <{tag}>, which the binder steps over rather than filling"
        if tag in _BINDING_STRIPPED_TAGS:
            return f"on <{tag}>, which the sanitizer removes along with the binding"
        if tag in _VOID_TAGS:
            return f"on <{tag}>, which holds no text, so the bound value renders nowhere"
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.tags.add(tag)
        if tag == _HEAD:
            self._head_depth += 1
        if self._open_depth and tag not in _VOID_TAGS:
            self._open_depth[-1] += 1
        opened: str | None = None
        bindings_here = 0
        for name, value in attrs:
            lowered = name.lower()
            if lowered == _BINDING:
                raw = value or ""
                text = raw.strip()
                bindings_here += 1
                if bindings_here > 1:
                    # The parser keeps the FIRST of repeated attributes and the binder
                    # reads that one, so a second name here is a field the contract is
                    # then required to carry and the page will never show.
                    self.unbindable.append(
                        f"{text or '<empty>'!r} is a repeated {_BINDING} on one "
                        f"<{tag}>; only the first is read"
                    )
                    continue
                if not text:
                    # A binding with no name binds nothing: the host looks the empty
                    # string up in the data and writes the empty string back, so the
                    # element renders blank forever with nothing to trace it to.
                    self.empty_bindings += 1
                    continue
                if raw != text:
                    # The binder looks the attribute up EXACTLY, without trimming, so
                    # " lede " is a key the data never has and the element renders
                    # blank. Trimming here would report the field as bound and let the
                    # equality gate agree with a contract the page cannot reach.
                    self.unbindable.append(
                        f"{raw!r} carries surrounding whitespace; the binder looks the "
                        f"name up exactly, so it never matches {text!r}"
                    )
                    continue
                unreachable = self._unreachable(tag)
                if unreachable is not None:
                    self.unbindable.append(f"{text!r} is bound {unreachable}")
                    continue
                if self._open_bound:
                    self.unbindable.append(
                        f"{text!r} is bound inside the element bound to "
                        f"{self._open_bound[-1]!r}; filling the outer one replaces its "
                        f"children, so this value is written to a detached node"
                    )
                    continue
                self.fields.append(text)
                opened = text
                continue
            # The local name, so ``xlink:href`` is judged as an ``href``.
            local = lowered.rsplit(":", 1)[-1]
            if local in _DROPPED_TEXT_ATTRS:
                self.dropped_text.add((lowered, (value or "").strip()))
                continue
            if local not in _OUTBOUND:
                continue
            target = value or ""
            # Judged on the RAW value, because the host's own test is
            # ``!value.startsWith('#')`` with no trimming: a leading space makes
            # `` #top`` fail that test and the attribute is removed. Stripping first
            # would call it a same-document fragment and report nothing.
            if local == "href" and target.startswith("#"):
                continue
            self.outbound.add((lowered, target.strip()))
        if opened is not None and tag not in _VOID_TAGS:
            # Pushed after the attribute loop so the element does not count as nested
            # inside itself. A void element opens no scope, so nothing can sit in it.
            self._open_bound.append(opened)
            self._open_depth.append(0)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == _HEAD and self._head_depth:
            self._head_depth -= 1
        if not self._open_depth:
            return
        if self._open_depth[-1]:
            self._open_depth[-1] -= 1
        else:
            self._open_bound.pop()
            self._open_depth.pop()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)


def _scan(html: str) -> _FieldScanner:
    scanner = _FieldScanner()
    scanner.feed(html)
    scanner.close()
    return scanner


def html_fields(html: str) -> set[str]:
    """Every field *html* binds, as a set.

    Refuses a page that binds nothing -- an extractor returning an empty set would
    make the equality gate report the contract as entirely over-specified, which sends
    the reader to delete the contract instead of fixing the page.
    """
    scanner = _scan(html)
    if scanner.empty_bindings:
        raise ExtractionRefused(
            f"{scanner.empty_bindings} {_BINDING} attribute(s) carry no field name; "
            "such an element renders blank forever and names nothing to fix"
        )
    if scanner.unbindable:
        listed = "; ".join(scanner.unbindable)
        raise ExtractionRefused(
            f"{len(scanner.unbindable)} binding(s) the host will never fill: {listed}. "
            "Counting them would let the equality gate pass on a page whose cell is "
            "blank at every render. Move each onto a visible element inside the body."
        )
    if not scanner.fields:
        raise ExtractionRefused(
            f"no {_BINDING} bindings found; either the page renders no data or this "
            "reader cannot see it, and those must not look alike"
        )
    return set(scanner.fields)


def control_tags_used(html: str) -> set[str]:
    """The control tags *html* contains. Empty is the only acceptable answer.

    Tags only. Attributes that reach outside the page are a separate rule with a
    separate reader, :func:`outbound_references`, so a reader of either gate is never
    surprised by what it silently also covers.
    """
    return _scan(html).tags & CONTROL_TAGS


def outbound_references(html: str) -> set[tuple[str, str]]:
    """Every ``(attribute, value)`` in *html* that reaches outside the page.

    A ``src`` or ``srcset`` of any value and an ``href`` that is not a same-document
    ``#fragment``, with the attribute matched on its local name so ``xlink:href`` counts.
    The host strips these like it strips a control, so a page carrying one ships a hole
    where the author believes an image or a link is. Empty is the only acceptable answer.
    """
    return _scan(html).outbound


def dropped_text_attributes(html: str) -> set[tuple[str, str]]:
    """Every ``(attribute, value)`` the host DELETES because it carries unreadable text.

    A separate reader from :func:`outbound_references` because the reason differs and a
    reader of either should not be surprised by what the other silently also covers:
    these reach nothing outside the page. The host removes them from a card so that no
    fact reaches the reader through a channel its own text scan cannot audit -- which
    means an ``alt`` or a ``title`` carrying part of the answer is a fact deleted before
    anyone reads it. Empty is the only acceptable answer.
    """
    return _scan(html).dropped_text


# --------------------------------------------------------------------------
# the contract end
# --------------------------------------------------------------------------


def _is_typed_dict(annotation: Any) -> bool:
    return (
        isinstance(annotation, type)
        and issubclass(annotation, dict)
        and hasattr(annotation, "__annotations__")
    )


def _is_structured(annotation: Any) -> bool:
    """Whether *annotation* names a shape a flat card cannot carry, at ANY depth.

    Recursive through unions, which is not a refinement but the whole rule: the common
    way a contract acquires a nested value is by making it optional
    (``inner: Inner | Unsaid``). Checking only the outermost annotation sees a union
    -- neither a TypedDict nor a list -- and lets the nested shape straight through,
    so the gate reports one flat key for a value the page can never render.
    """
    if _is_typed_dict(annotation) or typing.get_origin(annotation) in (list, dict, set, tuple):
        return True
    return any(_is_structured(arg) for arg in typing.get_args(annotation))


def contract_keys(contract: Any) -> set[str]:
    """Every key *contract* declares.

    Refuses a nested :class:`~typing.TypedDict` or a list of one, at any depth. A
    card's ``data`` is FLAT text -- the host binds one field name to one element's text
    -- so a nested shape has no way to reach the page, and accepting it here would let
    a contract declare structure the html could never read while the equality gate
    stayed green over the leaves.
    """
    hints = typing.get_type_hints(contract)
    if not hints:
        raise ExtractionRefused(f"{contract!r} declares no keys")
    nested = sorted(name for name, annotation in hints.items() if _is_structured(annotation))
    if nested:
        raise ExtractionRefused(
            f"card data is flat text, so a contract cannot nest: {nested}. "
            "Flatten the value into its own field, or render it as one text field."
        )
    return set(hints)


def _mentions_int(annotation: Any) -> bool:
    """Whether *annotation* is ``int`` or a union containing it."""
    if annotation is int:
        return True
    return int in typing.get_args(annotation)


def denominator_gaps(contract: Any) -> list[str]:
    """Int-typed keys with no denominator, i.e. invariant 2's violations.

    A count reaches the page either already shaped ``N/M`` -- in which case its type
    is ``str`` and this says nothing about it -- or as an ``int`` whose total is a
    sibling field named ``<key>_of``. A bare ``int`` with neither is a number a reader
    cannot size, and "7" on a status card reads as complete information.

    A denominator field is itself exempt, otherwise the rule would demand
    ``failed_of_of``.

    The sibling must itself mention ``int``. A present-but-unusable denominator is the
    same defect wearing the right name: ``failed: int`` beside ``failed_of: str`` passes
    a mere membership check while the page still has no number to divide by, so "7"
    renders exactly as sized as it did before.
    """
    hints = typing.get_type_hints(contract)
    gaps: list[str] = []
    for name, annotation in hints.items():
        if name.endswith(_DENOMINATOR_SUFFIX) or not _mentions_int(annotation):
            continue
        sibling = hints.get(f"{name}{_DENOMINATOR_SUFFIX}")
        if sibling is None or not _mentions_int(sibling):
            gaps.append(name)
    return gaps
