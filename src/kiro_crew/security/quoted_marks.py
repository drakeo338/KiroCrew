"""Marks the self-protection tokenizer glues onto quoted text before the quotes go.

bash reads a reserved word (``case``, ``esac``, ``in``, the command-position keepers)
only when no character of it is quoted, and a backtick inside single quotes or behind
a backslash is text rather than a command substitution.  The tokenizer drops the
quotes, so the argv walker (``shell_normalizer._SubstitutionDepth``) reading the
de-quoted tokens could not tell ``"case"`` from ``case`` or ``'a`b'`` from ``a`b``:
the first armed a phantom ``case`` pattern and the second opened a substitution no
later token closed, and in both an ssh destination behind the token was never read.

:func:`mask_quoted_reserved_words` runs one quote-state pass over the RAW text and
glues :data:`QUOTED_WORD_MARK` onto a quoted grammar word, and rewrites a single-quoted
or escaped backtick to :data:`QUOTED_TICK_MARK`; the walker then reads an ordinary word,
and every site that compares a token's text drops the marks again.  Both bytes sit
outside the ``\\x00``-``\\x06`` range ``argv_floor._QUOTED_SEP_SENTINELS`` uses for the
same job on the command separators.
"""

from __future__ import annotations

from typing import Callable

#: Reserved words after which bash still reads the NEXT word in command
#: position.  ``case`` is a reserved word ONLY in command position, so
#: ``if true; then case x in ...`` must arm the pattern rule while
#: ``echo case`` must not -- and the hand-through is INHERITED: ``then`` only
#: passes command position when it stands in command position itself
#: (``echo then case ...`` is three arguments).  Block ENDERS (``fi``,
#: ``done``, ``}``, ``esac``) are deliberately absent: bash refuses a keyword
#: directly after them (``fi case ...`` is a syntax error, measured), so not
#: arming there is exact.  Cross-pinned against
#: ``argv_floor._SHELL_RESERVED_WORDS`` by test, so the two keyword tables
#: cannot drift apart silently.  Defined here so the raw-text closer scan and the
#: walker read the same table (R28 GPT: ``time -p case`` is a case to both).
KEEPS_COMMAND_POSITION = frozenset(
    {"if", "then", "else", "elif", "while", "until", "do", "!", "{", "time", "coproc"}
)

#: Glued onto a token whose QUOTED spelling de-quotes to a walker grammar word.
QUOTED_WORD_MARK = "\x07"
#: Stands in for a single-quoted or escaped backtick.
QUOTED_TICK_MARK = "\x08"
#: Stands in for a QUOTED blank that is text of one word: ``case "x y" in "x y")``
#: names the WORD ``x y`` and the PATTERN ``x y``, and the tokenizer hands the walker
#: ``x y`` with no sign the blank was quoted, so the walker split it and lost the
#: compound (R22 GPT: the mint verb behind the closer minted).
QUOTED_BLANK_MARK = "\x0b"
#: Stand in for a QUOTED ``>`` / ``<``: text to bash, but the tokenizer drops the quote
#: and the argv walks read ``"a>b"`` as a redirection whose target swallows the next
#: word (``ssh -l "x 2>/dev/null; echo root" localhost`` was never host-checked, R23).
QUOTED_GT_MARK = "\x0c"
QUOTED_LT_MARK = "\x0e"
#: A quoted tab or newline is one word's text too (R25 GPT: ``case "x<TAB>y" in``).
QUOTED_TAB_MARK = "\x0f"
QUOTED_NL_MARK = "\x10"
#: What a quoted character is rewritten to, when it is rewritten at all.
_QUOTED_CHAR_MARKS = {
    " ": QUOTED_BLANK_MARK,
    "\t": QUOTED_TAB_MARK,
    "\n": QUOTED_NL_MARK,
    ">": QUOTED_GT_MARK,
    "<": QUOTED_LT_MARK,
}

_QUOTED_CHAR_TABLE = str.maketrans(_QUOTED_CHAR_MARKS)
#: The marks turned back into the text bash hands the program (the grammar-word mark
#: is dropped): every site that compares a token's text reads through this.
UNMARK_TABLE = str.maketrans(
    {QUOTED_WORD_MARK: None, QUOTED_TICK_MARK: "`", **{v: k for k, v in _QUOTED_CHAR_MARKS.items()}}
)
MARKS = frozenset(map(chr, UNMARK_TABLE))


def unmark(text: str) -> str:
    """*text* with the quoted-text marks turned back into the characters they stand for."""
    return text.translate(UNMARK_TABLE)


def _substitution_closer(text: str, start: int, closer: str) -> int:
    """Index of the ``)`` (or backtick) that closes the substitution opened before
    *start*, or -1.  A quote-aware scan: quoting restarted at *start*, so an inner
    ``"…"`` or ``'…'`` is a real quote (``"$(dirname "$0")/a"``, R25 Opus) and a ``)``
    inside it or inside a nested ``(`` is not the closer.  A ``)`` that closes a
    ``case`` PATTERN is not the closer either; the pass cannot parse that grammar, so
    a candidate is skipped while more ``case`` than ``esac`` words stand in COMMAND
    position so far (``"$(case y in y) :;; esac)"`` runs to its last ``)``; an
    argument spelled ``case`` counts for nothing, R26 GPT).  Command position is
    read as the walker reads it: a separator or a pattern's ``)`` opens it, and a
    keeper (:data:`KEEPS_COMMAND_POSITION`) or an option word holds it only when it
    holds it itself (``time -p case x in x)`` is a case, R28 GPT), and the NAME after
    ``function``/``coproc`` hands it on (``function f case x in x)``, R29 GPT).  When
    that reading finds no closer, the plain paren reading is taken instead.
    """
    found = _closer_scan(text, start, closer, case_aware=True)
    return found if found >= 0 else _closer_scan(text, start, closer, case_aware=False)


def _closer_scan(text: str, start: int, closer: str, *, case_aware: bool) -> int:
    n = len(text)
    depth = 0
    quote: str | None = None
    cases = esacs = 0
    word: list[str] = []
    command_position = True  # the word being read starts a command
    name_next = False  # the word being read is the NAME after ``function``/``coproc``
    k = start
    while k < n:
        ch = text[k]
        if quote is not None:
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and k + 1 < n:
                k += 1
            k += 1
            continue
        if ch in " \t\n;|&()<>`":
            w = "".join(word)
            word.clear()
            if w and name_next:
                # ``function f case …``: the name is read, and a compound follows
                # in command position (R29 GPT), as the walker reads it.
                name_next = False
                command_position = True
            elif w:
                if command_position:
                    cases += w == "case"
                    esacs += w == "esac"
                name_next = command_position and w in ("function", "coproc")
                command_position = command_position and (
                    w in KEEPS_COMMAND_POSITION or w.startswith("-")
                )
            if ch in ";|&\n()":
                command_position = True  # a separator or a pattern's ``)`` opens one
                name_next = False
        if ch == "\\" and k + 1 < n:
            k += 2
            continue
        if ch in ("'", '"'):
            quote = ch
        elif closer == "`" and ch == "`":
            return k
        elif ch == "(":
            depth += 1
        elif ch == ")" and closer == ")":
            if depth:
                depth -= 1
            elif not case_aware or cases <= esacs:
                return k
        elif ch not in " \t\n;|&<>`":
            word.append(ch)
        k += 1
    return -1


def mask_quoted_reserved_words(
    text: str,
    grammar_words: "frozenset[str]",
    word_break: "frozenset[str]",
    decode_ansi_c: "Callable[[str], str]",
) -> str:
    """Mark quoted grammar words and quoted backticks in the raw *text*.

    A word runs to the next UNQUOTED character of *word_break*; its de-quoted text
    is what bash would hand the command, and ``quoted`` records whether any character
    of it was inside quotes or behind a backslash.  A word that de-quotes to a member
    of *grammar_words* with ``quoted`` set gets :data:`QUOTED_WORD_MARK` appended after
    its last character: the tokenizer glues adjacent text into one token, so
    ``"case"`` arrives as ``case`` + mark and the walker reads an argument.

    Quote rules are bash's: a single quote takes no escapes; inside double quotes a
    backslash escapes only a double quote, a backslash, ``$`` or a backtick, and a
    backtick there still substitutes; outside quotes a backslash escapes the next
    character (a backslash-newline is a line continuation, neither text nor a quote).
    ``$'…'``/``$"…"`` open a quote with the ``$`` belonging to the opener, not to the
    word's text; the body of ``$'…'`` is compared DECODED (*decode_ansi_c*, the
    tokenizer's own decoder), since ``$'\\x63ase'`` is the word ``case`` to bash and to
    the tokenizer alike.  The function is the identity on every command with neither
    a quoted grammar word nor a single-quoted or escaped backtick.

    A blank inside quotes becomes :data:`QUOTED_BLANK_MARK` (text of one word); a
    substitution met inside double quotes restarts quoting, so ITS blanks stay the
    separators of that command list (``"$(case y in y) :;; esac)"`` is one token).
    """
    out: list[str] = []
    word: list[str] = []  # de-quoted text of the word being read
    quoted = False
    quote: str | None = None
    ansi = False  # the open quote is ``$'…'``: its body decodes
    # Quoting RESTARTS inside a command substitution: in ``"$('case' x in y)"`` the
    # single quotes are real quotes to bash (R23 GPT), so a ``$(`` or backtick met
    # inside double quotes pushes the enclosing quote here and reads its body unquoted;
    # the substitution's own closer (a ``)`` at its paren depth, or the backtick) pops it.
    # (enclosing quote, index of the closer that ends the restarted substitution): the
    # raw pass cannot parse the case grammar whose ``)`` is not a closer, so a ``$(``
    # runs to the LAST ``)`` before the enclosing quote closes; a backtick to the next.
    enclosing: list[tuple[str, int]] = []
    # An opener inside double quotes with NO closer ahead is one bash refuses the whole
    # line for, so every later opener is text as well and is never probed: probing each
    # of a run of ``$(`` to the end of the text is quadratic in the command's length,
    # and a 20 KB run held the synchronous gate for minutes (R27 Opus).
    refused = False
    i = 0
    n = len(text)

    def end_word() -> None:
        nonlocal quoted
        if quoted and "".join(word) in grammar_words:
            out.append(QUOTED_WORD_MARK)
        word.clear()
        quoted = False

    while i < n:
        ch = text[i]
        if quote is None:
            if enclosing and i == enclosing[-1][1]:
                # The restarted substitution closes: back inside the enclosing quote.
                end_word()
                out.append(ch)
                i += 1
                quote = enclosing.pop()[0]
                continue
            if ch in word_break:
                end_word()
                out.append(ch)
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                nxt = text[i + 1]
                out.append(ch)
                if nxt == "\n":
                    out.append(nxt)  # a line continuation: neither text nor a quote
                else:
                    out.append(QUOTED_TICK_MARK if nxt == "`" else _QUOTED_CHAR_MARKS.get(nxt, nxt))
                if nxt != "\n":
                    word.append(nxt)
                    quoted = True
                i += 2
                continue
            if ch == "$" and i + 1 < n and text[i + 1] in ("'", '"'):
                out.append(ch)  # ANSI-C / locale quoting: the ``$`` is the opener's
                i += 1
                ch = text[i]
                ansi = ch == "'"
            if ch in ("'", '"'):
                quote = ch
                quoted = True
                out.append(ch)
                i += 1
                if ansi:
                    # ``$'…'``: copy the raw body through (marking a backtick inside
                    # it, text to bash), and add the DECODED body to the word.
                    j = i
                    while j < n and text[j] != "'":
                        j += 2 if text[j] == "\\" and j + 1 < n else 1
                    decoded = decode_ansi_c(text[i:j])
                    # Emit the DECODED body re-escaped for the tokenizer's own decoder,
                    # with every backtick it decodes to marked: ``$'\\x60'`` is a
                    # backtick to bash and text, not a substitution (R21 GPT).
                    out.append(
                        decoded.replace("\\", "\\\\")
                        .replace("'", "\\'")
                        .replace("`", QUOTED_TICK_MARK)
                        .translate(_QUOTED_CHAR_TABLE)
                    )
                    word.extend(decoded)
                    i = j
                    quote = None if i >= n else quote
                    ansi = False
                    if i < n:
                        out.append(text[i])  # the closing quote
                        i += 1
                        quote = None
                continue
            word.append(ch)
            out.append(ch)
            i += 1
            continue
        if ch == quote:
            quote = None
            out.append(ch)
            i += 1
            continue
        if quote == '"' and not refused and (text.startswith("$(", i) or ch == "`"):
            # A substitution opens inside double quotes: its body is read UNQUOTED up
            # to its closer, then the quote resumes.
            end_word()
            j = i + (2 if ch == "$" else 1)
            closer = _substitution_closer(text, j, ")" if ch == "$" else "`")
            if closer < 0:
                # No closer ahead: bash would refuse the line.  Read the opener as text
                # so the enclosing quote still closes where it does (R25 Opus), and
                # every later opener as text without a probe (R27 Opus).
                refused = True
                word.extend(text[i:j])
                out.append(text[i:j])
                i = j
                continue
            out.append(text[i:j])
            enclosing.append((quote, closer))
            quote = None
            i = j
            continue
        if quote == '"' and ch == "\\" and i + 1 < n and text[i + 1] in '"\\$`':
            out.append(ch)
            out.append(
                QUOTED_TICK_MARK if text[i + 1] == "`" else text[i + 1]
            )  # an escaped one is text
            word.append(text[i + 1])
            i += 2
            continue
        word.append(ch)
        if ch == "`" and quote == "'":
            out.append(QUOTED_TICK_MARK)
        else:
            out.append(_QUOTED_CHAR_MARKS.get(ch, ch))
        i += 1
    end_word()
    return "".join(out)
