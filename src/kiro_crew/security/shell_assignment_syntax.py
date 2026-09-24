"""Shell assignment and operator syntax: the leaf readers the resolver is built on.

Control-operator and assignment spellings, the words that guard or scope an
assignment, the quote-state walk, and the glued-operator split.  Nothing here
tracks a binding: :mod:`.shell_normalizer` owns the resolver that does, and reads
these as its vocabulary.  Split out when that module reached the per-file line
cap of the package's monolith ratchet.
"""

from __future__ import annotations

import re
from itertools import product
from typing import Callable

# The start of a redirection, with any descriptor prefix; for use where a redirect
# ENDS an argument list rather than hiding a program.  Testing only the first
# character missed every descriptor-prefixed spelling (``2>``, ``&>``, ``{fd}>``,
# ``1>``), which is exactly where a redirection is most often written -- so the
# descriptor read as an ordinary refspec and the command after the redirect was
# absorbed as arguments.
_REDIRECT_START_RE = re.compile(r"(?:\d+|&|\*|\{[A-Za-z_][A-Za-z0-9_]*\})?(?:>{1,2}[&|!]?|<{1,3})")


# ``NAME=value``: a literal program name may reach its use only through the
# expansion, so neither the literal name nor the expansion alone looks dangerous.
# The assignment and the use are in the SAME command text, so the literal can be
# substituted back before any comparison.
_LOCAL_ASSIGN_RE = re.compile(r"\A([A-Za-z_][A-Za-z0-9_]*)=(.*)\Z", re.DOTALL)


# `NAME=value` prefix: `normalize_shell_command` keeps it as a single token, and
# the value is already $HOME-expanded by the time it is read.
#: ``NAME=value`` and ``NAME+=value``. The append form is a separate group so a
#: caller can add to what it already recorded instead of replacing it. Matching
#: only ``=`` means the whole ``NAME+=`` token fails to match, so the segment
#: reads as a command word rather than an assignment.
_SHELL_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(\+?)=(.*)$", re.DOTALL)


# A run of control operators: what separates one program from the next in a run
# that ``shlex`` handed over as a single word.
_CONTROL_OPERATOR_RE = re.compile(r"[;&|\n]+")
# The same split with the operators KEPT, so a script is walked segment by segment.
_CONTROL_OPERATOR_SPLIT_RE = re.compile(r"([;&|\n]+)")
# An assignment after ``||``/``&&`` may not run and one before ``|``/``&`` runs in a
# subshell: neither replaces the binding before it.  A run of assignments followed by
# a boundary (or nothing) persists; followed by a command word it is a prefix.
_CONDITIONAL_OPERATORS = frozenset({"||", "&&", "|", "|&"})
# The body of a compound command runs only when its test or pattern says so, so an
# assignment after one of these keywords (or after a ``case`` pattern, a word ending
# in ``)``) is guarded exactly as one after ``||``/``&&``: ``x=<cli>; if false; then
# x=echo; fi; $x <verb>`` runs the mint while the single reading took ``echo`` (found
# in review).  ``{`` is included: a group is a function body as often as a block.
_GUARD_WORDS = frozenset({"then", "do", "else", "elif", "{"})
# An assignment-shaped word after one of these is the shell's own binding; after any
# other command word it is that command's DATA (see ``_is_argument_assignment``).
_DECLARATION_BUILTINS = frozenset({"export", "declare", "typeset", "local", "readonly"})
_SUBSHELL_OPERATORS = frozenset({"|", "&", "|&"})
_COMMAND_BOUNDARY_TOKENS = frozenset({";", "||", "&&", "|", "|&", "&", ";;", ";&", ";;&"})
_SHELL_WORD_RE = re.compile(r"""(?:"[^"]*"|'[^']*'|\S)+""")
# A ``case`` pattern's closing ``)`` ends a word: ``x)$x <verb>`` is the pattern ``x)``
# and the command ``$x``, glued (found in review: the use stayed inside the pattern
# word and the mint was never formed).  No ``(``, quote, ``$`` or ``;``/``&`` before
# the ``)``: a substitution, a function definition or a glued separator is not a
# pattern, and a glued OPERATOR (``esac);``) is no word.  A literal ``a)b`` argument
# split so only widens the reading.
_CASE_PATTERN_GLUE_RE = re.compile(r"""\A([^\s()"'$;&]*\))([^\s;&|]\S*)\Z""")
# A function definition's ``{`` may be glued to its ``()`` (``f(){``): it is the
# ``{`` that opens the body, and a reassignment in the body runs when the function
# is called (``x=echo; bash -c 'f(){ x=<cli>; }; f; $x <verb>'`` minted while the
# body's assignment hid behind the glued word; found in review).
_FUNCTION_DEF_GLUE_RE = re.compile(r"\A(\S*\(\))\{\Z")


def _split_glued_word(word: str) -> "list[str]":
    """*word* with a ``case`` pattern's ``)`` or a function definition's ``{`` split off."""
    glued = _CASE_PATTERN_GLUE_RE.match(word)
    if glued:
        return [glued.group(1), glued.group(2)]
    function = _FUNCTION_DEF_GLUE_RE.match(word)
    if function:
        return [function.group(1), "{"]
    return [word]


def _shell_words(segment: str) -> "list[str]":
    """*segment*'s words, a word glued to a ``case`` pattern or a function's ``{`` split."""
    return [piece for word in _SHELL_WORD_RE.findall(segment) for piece in _split_glued_word(word)]


# The whitespace ``shlex`` splits on: one INSIDE a token is proof the token was quoted.
_SHLEX_WHITESPACE_RE = re.compile(r"[ \t\r\n]")
# ``$IFS`` expands to the word separators themselves, so an unquoted ``$x${IFS}<verb>``
# is two words to bash (the raw rules already read it so; the resolver's expansion
# of ``$x`` did not, found in review).  Case-insensitive: the text is case-folded.
_IFS_USE_RE = re.compile(r"\$\{IFS\}|\$IFS(?![A-Za-z0-9_])", re.IGNORECASE)


def _is_guard_token(token: str) -> bool:
    """True when an assignment right after *token* may not run (see ``_GUARD_WORDS``)."""
    return token in _CONDITIONAL_OPERATORS or token in _GUARD_WORDS or token.endswith(")")


# A compound command's body runs as ONE unit: every statement between the opener and
# its closer is as guarded as the first.  Testing only the adjacent token let one
# extra statement re-open the mint (``x=<cli>; if false; then y=1; x=echo; fi; $x
# <verb>``: ``x=echo`` follows ``;``, not ``then``; found in review).  ``if``/``fi``
# rather than ``then``: an ``elif`` chain has one ``fi`` for several ``then``.
_COMPOUND_OPENERS = frozenset({"if", "while", "until", "for", "select", "case", "{"})
_COMPOUND_CLOSERS = frozenset({"fi", "done", "esac", "}"})
_COMPOUND_KEYWORDS = _COMPOUND_OPENERS | _COMPOUND_CLOSERS
_BRACES = frozenset({"{", "}"})


def _compound_body_flags(
    tokens: "list[str]", in_command_position: "Callable[[int], bool]", quoted_is_data: bool
) -> "list[bool]":
    """``flags[idx]``: ``tokens[idx]`` sits inside a compound command's body.

    A keyword counts only in command position: the first word of a token that is
    itself in command position, or an opener right after a guard word (``else if``,
    ``then {``) or after a control operator glued inside a token.  A CLOSER glued
    inside a token never counts: ``shlex`` hands ``echo 'a;fi'`` over as the bare
    token ``a;fi``, indistinguishable from a real separator, and closing the body on
    it read the reassignment after it as unguarded (found in review) -- an opener
    read from such a token only widens the reading, a closer would narrow it.
    With *quoted_is_data* a whitespace-bearing token is a quoted word (``echo "if
    so"``) and counts for nothing; the whole-script walk passes its segments with
    ``False``, since there every segment is a command.  Depth never goes below zero.
    """
    flags: list[bool] = []
    depth = 0
    for idx, token in enumerate(tokens):
        if quoted_is_data and _SHLEX_WHITESPACE_RE.search(token):
            flags.append(depth > 0)
            continue
        pieces = _CONTROL_OPERATOR_SPLIT_RE.split(token)
        first = _shell_words(pieces[0])
        # A guard word is followed by a command too (``then if``, ``do {``), as a
        # separate token when the source spaced them.  The position walk (linear in
        # the assignment run before the token) is taken only for a keyword.
        keyword = bool(first) and (first[0] in _COMPOUND_KEYWORDS or first[0] in _GUARD_WORDS)
        # A brace is a word of its own wherever it stands (``f(){``, ``function f {``),
        # so it is read without the position walk; ``{`` as an argument is rare and
        # only widens the reading.
        opens = keyword and (
            first[0] in _BRACES
            or (idx > 0 and (tokens[idx - 1].split() or [""])[-1] in _GUARD_WORDS)
            or in_command_position(idx)
        )
        if first and first[0] in _COMPOUND_CLOSERS and depth and opens:
            depth -= 1
        flags.append(depth > 0)
        for at, piece in enumerate(pieces):
            if _CONTROL_OPERATOR_RE.fullmatch(piece):
                continue
            words = _shell_words(piece)
            if not words or (at == 0 and not opens):
                continue
            for pos, word in enumerate(words):
                if word in _COMPOUND_OPENERS and (
                    pos == 0 or word == "{" or words[pos - 1] in _GUARD_WORDS
                ):
                    depth += 1
    return flags


def _guarded_body(segment: str) -> "tuple[bool, str]":
    """``(guarded, body)``: *segment* with a leading guard word (or the words up to a
    ``case`` pattern) removed when an assignment follows it, else the segment as is."""
    words = _shell_words(segment)
    for pos, word in enumerate(words[:-1]):
        if _is_guard_token(word) and _SHELL_ASSIGN_RE.match(words[pos + 1]):
            return True, " ".join(words[pos + 1 :])
    return False, segment


def _is_command_word(token: str) -> bool:
    """True when a prefix before *token* is scoped to it: not a boundary, redirection
    or comment.  Builtins are command words too: outside POSIX mode bash scopes a
    prefix to every builtin (``x=echo export y`` leaves ``x`` alone), and that is
    the reading that refuses; ``sh`` persisting it before a SPECIAL builtin only
    widens the refusal.  Regular builtins (``local``, ``declare``) never persist."""
    return not (
        token in _COMMAND_BOUNDARY_TOKENS
        or token.startswith("#")
        or _REDIRECT_START_RE.match(token) is not None
    )


def _leading_assignments(segment: str) -> "list[tuple[str, str, bool]]":
    """``(name, value, appends)`` for a segment's LEADING run of assignments.

    Both spellings open a command's prefix (``y=1 x=<cli>``, ``A+=foo x=<cli>``): an
    append in the run is still an assignment, so the command word comes after it.
    """
    pairs: list[tuple[str, str, bool]] = []
    for word in _shell_words(segment):
        assign = _SHELL_ASSIGN_RE.match(word)
        if not assign:
            break
        pairs.append((assign.group(1), assign.group(3).strip("\"'"), bool(assign.group(2))))
    return pairs


def _token_in_command_position(tokens: "list[str]", idx: int) -> bool:
    """True when ``tokens[idx]`` is the command word: what follows a boundary (or the
    start) and that command's leading assignments.  A boundary is a separator token
    or a token ENDING in one (``shlex`` glues ``true;``).  An APPEND is a leading
    assignment too (``A+=foo $x <verb>`` runs ``$x``): read with the plain spelling
    only, the append hid the command position and a multiword value stayed one word.
    """
    look = idx
    while look and _SHELL_ASSIGN_RE.match(tokens[look - 1]):
        look -= 1
    return look == 0 or _CONTROL_OPERATOR_RE.fullmatch(tokens[look - 1][-1:]) is not None


def _is_argument_assignment(tokens: "list[str]", idx: int) -> bool:
    """True when the assignment-shaped ``tokens[idx]`` is an ARGUMENT: it follows a
    command word other than a declaration builtin, so the shell hands it to that
    command as data (``echo x=echo``, ``make CFLAGS=-O2``) -- and the PIECES of a
    quoted operand (see :func:`_split_glued_operators`) land here too, because the
    operand's own separators put them after its command word.  Whether the command
    runs them (``eval``) is what the resolver reads both ways."""
    if _token_in_command_position(tokens, idx):
        return False
    look = idx
    while look and _SHELL_ASSIGN_RE.match(tokens[look - 1]):
        look -= 1
    return tokens[look - 1] not in _DECLARATION_BUILTINS


def _is_command_scoped_assignment(segment: str) -> bool:
    """True when a segment is ``NAME=value ... command`` (``X=foo true``), not a bare run."""
    rest = _shell_words(segment)[len(_leading_assignments(segment)) :]
    return bool(rest) and _is_command_word(rest[0])


def _quote_state_after(text: str, quote: str) -> str:
    """The quote (``'``, ``"`` or none) still OPEN after *text*, entered with *quote* open.

    A separator inside a quote is data, so a segment that opens inside one is not a
    command and cannot assign (``printf "%s" "; x=echo"``).  An unclosed quote stays
    open to the end: every later segment is then read as data, which only widens the
    outer binding's reach -- the refusal direction.
    """
    skip = False
    for ch in text:
        if skip:
            skip = False
        elif ch == "\\" and quote != "'":
            skip = True
        elif quote:
            quote = "" if ch == quote else quote
        elif ch in "\"'":
            quote = ch
    return quote


_TRAILING_OPERATOR_RE = re.compile(r"[;&|\n]+\Z")


def _trailing_operator(token: str) -> str:
    """The control-operator run *token* ends in (``|&`` of ``"$v token"|&``), or ``""``."""
    match = _TRAILING_OPERATOR_RE.search(token)
    return match.group(0) if match else ""


def _split_glued_operators(tokens: "list[str]") -> "list[str]":
    """Split tokens on control operators glued to their neighbours.

    ``shlex`` splits on whitespace only, so ``X=<name>;$X`` arrives as one token and an
    assignment glued to the command that uses it is invisible to both.  Splitting keeps
    the operator itself as a token so argv-boundary logic still sees it.

    A token that CONTAINS ``shlex`` whitespace was quoted, and a quoted word is ONE
    argument however many ``;`` it carries (``bash -c '<name>=<cli>; $<name> <verb>'``).
    Split into pieces ALONE, the payload walk -- which takes the ONE token after the
    carrier -- saw only ``<name>=<cli>``.  So such a token is yielded WHOLE first
    (the walk re-tokenizes it) and then its pieces, because the whitespace may sit
    inside a quoted VALUE of a top-level glued run (``X="a b";Y=<cli>;$Y <verb>``).
    """
    out: list[str] = []
    for token in tokens:
        # A word glued to a ``case`` pattern's ``)`` or a function's ``{`` is split
        # off first (see :func:`_split_glued_word`); the rest reads as any token.
        *glued, token = _split_glued_word(token)
        out.extend(glued)
        # ONLY split a token that begins with an assignment.  Splitting any token
        # carrying a separator would destroy a QUOTED target -- ``shlex`` has already
        # removed the quotes, so ``pkill -f '[;]*<name>'`` arrives as the single token
        # ``[;]*<name>`` and is indistinguishable from a real separator at this point.
        # The reported evasion is specifically an assignment glued to its use, so that
        # is the only shape split here -- in both its spellings: ``q+=$p;`` left
        # glued kept the append out of the reading (found in review).
        if not (_LOCAL_ASSIGN_RE.match(token) or _SHELL_ASSIGN_RE.match(token)) or not (
            _CONTROL_OPERATOR_RE.search(token)
        ):
            out.append(token)
            continue
        quoted_whole = bool(_SHLEX_WHITESPACE_RE.search(token))
        if quoted_whole:
            out.append(token)  # quoted whole (see above): the carrier's operand first
        # The operator run is kept as spelled, so the resolver's guard sees ``||``.
        # In a quoted whole SCRIPT a separator INSIDE a quote it still carries is data
        # (``printf ";x=echo"`` is no assignment), so it stays glued to its word; an
        # UNCLOSED quote keeps the plain split, since ``shlex`` may have consumed the
        # escape that balanced it.  A token WITHOUT ``shlex`` whitespace was never a
        # script: ``shlex`` has already stripped its quoting level, so a quote it still
        # carries was ESCAPED (``q=\";x=<cli>;\";$x``) and is literal data, while the
        # separators beside it are real -- the plain split reads them.
        # A glued run is collected and joined ONCE when it ends: appending to a
        # list element re-copies the run per separator (quadratic on a long
        # quoted token).
        pieces: list[str] = []
        run: list[str] = []
        quote = ""
        balanced = quoted_whole and not _quote_state_after(token, "")
        for piece in _CONTROL_OPERATOR_SPLIT_RE.split(token):
            opens_quoted, quote = bool(quote), _quote_state_after(piece, quote)
            if balanced and opens_quoted and run:
                run.append(piece)
                continue
            if run:
                pieces.append("".join(run))
            run = [piece]
        if run:
            pieces.append("".join(run))
        for piece in pieces:
            if _CONTROL_OPERATOR_RE.fullmatch(piece):
                out.append(piece.strip() or ";")
            elif piece:
                out.append(piece)
    return out


_VALUE_VAR_USE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


#: The self-target-capable programs a guarded operand must not hide behind: the cli
#: (checked by ``is_self_program``) and the remote-shell verbs that can open a channel
#: back to this host.  A guarded operand of any of these past the cap is fail-closed.
_SSH_LIKE_PROGRAMS = frozenset({"ssh", "scp", "sftp", "rsync"})


def _over_cap_group_is_fail_closed(
    tokens: "list[str]",
    names: "set[str]",
    is_protected_program: "Callable[[str], bool]",
    is_self_program: "Callable[[str], bool]",
    program_basename: "Callable[[str], str]",
    nested_shell_programs: "frozenset[str]",
    nested_shell_verbs: "frozenset[str]",
) -> bool:
    """Whether an over-cap guarded group must read as the fail-closed reading.

    Program-aware fold: an over-cap group folds to allowed only when NO simple command
    that uses a group name could, under ANY combination of its guarded values, invoke a
    dangerous program -- the cli, a kill program, an ssh-like verb reaching this host, a
    nested shell (``bash``/``eval``/...), or ``git``.  Each guarded name has BOTH values
    considered (its guard may or may not run), so a word assembled from adjacent
    expansions with opposite guard outcomes (``$p$q`` -> ``kirocrew``) and a value
    naming a nested shell (``g=bash; $g -c ...``) are both seen; the cli hidden behind an
    exec-wrapper is seen because it is one of the words.  An ordinary many-flag build or
    deploy script -- inert values fed to an inert program -- folds however many names
    meet in it.  An ambient ``$VAR`` the command never binds stays unresolved and is not
    dangerous, exactly as the top-level reading treats it.
    """
    values_by_name: "dict[str, list[str]]" = {}
    live: "dict[str, str]" = {}
    split = _split_glued_operators(tokens)
    for token in split:
        match = _SHELL_ASSIGN_RE.match(token)
        if not match:
            continue
        name, append, value = match.group(1), match.group(2), match.group(3)
        if append:  # ``NAME+=tail`` concatenates onto the live binding, as the shell does
            value = live.get(name, "") + value
        live[name] = value
        values_by_name.setdefault(name, []).append(value)

    def dangerous(word: str) -> bool:
        base = program_basename(word)
        return (
            is_protected_program(word)
            or is_self_program(word)
            or base in _SSH_LIKE_PROGRAMS
            or base in nested_shell_programs
            or base in nested_shell_verbs
            or base == "git"
        )

    def any_candidate_dangerous(word: str) -> bool:
        refs = list(
            dict.fromkeys(m.group(1) or m.group(2) for m in _VALUE_VAR_USE_RE.finditer(word))
        )
        options = [values_by_name.get(ref) or ["$" + ref] for ref in refs]
        combos = 1
        for opt in options:
            combos *= len(opt)
        if combos > 256:  # too many guard combinations to enumerate -- fail closed
            return True
        for pick in product(*options) if options else [()]:
            chosen = dict(zip(refs, pick))
            resolved = _VALUE_VAR_USE_RE.sub(
                lambda m: chosen.get(m.group(1) or m.group(2), m.group(0)), word
            )
            if "$" not in resolved and dangerous(resolved):
                return True
        return False

    # A group value that could resolve to a dangerous program.
    for name in names:
        for value in values_by_name.get(name, []):
            if any_candidate_dangerous(value):
                return True
    # A simple command that uses a group name AND whose words could assemble a dangerous
    # program under some combination of the guarded values.
    segments: "list[list[str]]" = []
    segment: "list[str]" = []
    for token in split:
        if _CONTROL_OPERATOR_RE.fullmatch(token):
            if segment:
                segments.append(segment)
            segment = []
            continue
        segment.append(token)
        if token and token[-1] in ";&|\n":
            segments.append(segment)
            segment = []
    if segment:
        segments.append(segment)
    for segment in segments:
        if not any(
            (m.group(1) or m.group(2)) in names
            for tok in segment
            for m in _VALUE_VAR_USE_RE.finditer(tok)
        ):
            continue
        if any(any_candidate_dangerous(tok) for tok in segment):
            return True
    return False
