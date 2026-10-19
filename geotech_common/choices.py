"""
What a module chose: the ``choices`` a result reports (backlog GM-115).

The owner, 2026-10-16: "make sure the agent is communicating its
engineering judgements to the user. User's judgement is the one that
matters." A result that picked a method by soil type, applied a default or
searched automatically has made choices the user never saw. A module's
result carries them as a ``choices`` list, one entry per choice:

* ``what``: the quantity or method ("side resistance method");
* ``chosen``: what was used ("Nordlund, 0-10 m");
* ``how``: ``"given"`` (in the inputs), ``"default"`` (the module's default
  value), ``"module"`` (the module decided, e.g. ``method="auto"``
  resolved by soil type) or ``"assumed"`` (in the inputs, but a calc
  package's call said the user did not give it: a method the agent chose,
  or an input it stated as an assumption; GM-167);
* ``basis``: the reason, cited in the reference's own terms, where one
  exists;
* ``alternatives``: the other options the module offers.

``basis`` and ``alternatives`` are left out when there is none: the agent
reads every tool result, so the list stays short. The text is plain words;
an input or result key the agent may need follows in brackets
("Boussinesq (stress_method 'boussinesq')"), and a printed table leaves
those brackets out (:func:`plain`).

A module cannot tell a default it applied from the same value given
explicitly, so the agent layer says which inputs a call named (GM-118):
``call_agent`` runs the method inside :func:`call_inputs`, and
:func:`how_given` with the input's ``name`` answers from that, so a typed
value equal to the default is "given". Outside such a call, and for values
nested inside an input (a layer's field), :func:`how_given` compares the
value with the default, and a nested value names the input it came in
(``within``), so it reads "assumed" with that input. A module never writes
"given" itself: an entry is "given" only by :func:`how_given`, so a calc
package's call that says an input is not the user's reaches every choice
made from it (GM-180). The agent layer also adds ``defaults_used`` to a
result (the declared defaults a call left in force and that apply to it);
a name :func:`how_given` called "default" is always among them, so the two
never contradict each other.

Pure data; no third-party import.
"""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, FrozenSet, Iterable, Iterator, List, Optional, Set

GIVEN = "given"
DEFAULT = "default"
MODULE = "module"
ASSUMED = "assumed"
HOW = (GIVEN, DEFAULT, MODULE, ASSUMED)

#: How each ``how`` reads in a summary or a calc package.
HOW_TEXT = {GIVEN: "given", DEFAULT: "default", MODULE: "module's choice",
            ASSUMED: "assumed"}


def _plain(value: Any) -> Any:
    """A numpy scalar as a plain Python value; anything else unchanged."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def choice(what: str, chosen: Any, how: str, basis: str = "",
           alternatives: Iterable = ()) -> dict:
    """One entry of a result's ``choices`` list.

    Parameters
    ----------
    what : str
        The quantity or method chosen, as the references name it.
    chosen : str, number or bool
        What was used.
    how : str
        ``"given"``, ``"default"``, ``"module"`` or ``"assumed"``.
    basis : str, optional
        The reason, with its reference where one governs.
    alternatives : iterable of str, optional
        The other options the module offers.
    """
    if how not in HOW:
        raise ValueError(f"how must be one of {HOW}, got {how!r}")
    entry = {"what": what, "chosen": _plain(chosen), "how": how}
    if basis:
        entry["basis"] = basis
    alts = [a for a in (alternatives or ()) if a != chosen]
    if alts:
        entry["alternatives"] = alts
    return entry


class CallInputs:
    """The inputs of one ``call_agent`` call (GM-118): the names it gave
    (``given``), the declared defaults it left in force (``defaults``), the
    names it gave that are not the user's (``assumed``: a calc package's
    stated assumptions and the methods it was not told the user gave,
    GM-167), and the names :func:`how_given` has called "default" so far
    (``reported``)."""

    __slots__ = ("given", "defaults", "assumed", "reported")

    def __init__(self, given: Iterable[str], defaults: Iterable[str],
                 assumed: Iterable[str] = ()):
        self.given: FrozenSet[str] = frozenset(given)
        self.defaults: FrozenSet[str] = frozenset(defaults)
        self.assumed: FrozenSet[str] = frozenset(assumed)
        self.reported: Set[str] = set()


_CALL: ContextVar[Optional[CallInputs]] = ContextVar(
    "geotech_call_inputs", default=None)


@contextmanager
def call_inputs(given: Iterable[str], defaults: Iterable[str] = (),
                assumed: Iterable[str] = ()) -> Iterator[CallInputs]:
    """Run a method call with the names it was given, the declared
    defaults it left in force and the given names that are not the user's
    (``assumed``) known to :func:`how_given`; yields the
    :class:`CallInputs`, whose ``reported`` the caller reads after."""
    record = CallInputs(given, defaults, assumed)
    token = _CALL.set(record)
    try:
        yield record
    finally:
        _CALL.reset(token)


def how_given(value: Any, default: Any, name: Optional[str] = None,
              within: Optional[str] = None) -> str:
    """``"given"`` or ``"default"``: whether the input was given.

    With ``name`` (the input as the tool declares it) inside
    :func:`call_inputs`: ``"given"`` when the call named it (``"assumed"``
    when the call also said it is not the user's, GM-167), ``"default"``
    when it left that declared default in force. Otherwise, and for a value
    nested inside an input: ``"default"`` when ``value`` equals the
    module's ``default``, else ``"given"``; with ``within`` (the input the
    nested value came in, such as ``"layers"`` for a layer's field),
    ``"assumed"`` in place of ``"given"`` when the call said that input is
    not the user's (GM-180). A required input has no default: pass
    ``None``."""
    record = _CALL.get()
    if name is not None and record is not None:
        if name in record.given:
            return ASSUMED if name in record.assumed else GIVEN
        if name in record.defaults:
            record.reported.add(name)
            return DEFAULT
    try:
        same = value == default
        same = bool(same)
    except (TypeError, ValueError):
        same = False
    if same:
        return DEFAULT
    if within is not None and record is not None \
            and within in record.given and within in record.assumed:
        return ASSUMED
    return GIVEN


def choices_lines(choices: Optional[List[dict]],
                  indent: str = "  ") -> List[str]:
    """The short "Choices" section of a ``summary()``: one line per entry,
    ``what: chosen (how)``; nothing when there are no choices."""
    if not choices:
        return []
    lines = ["", f"{indent}Choices:"]
    for c in choices:
        lines.append(f"{indent}  - {plain(c['what'])}: {plain(c['chosen'])} "
                     f"({HOW_TEXT.get(c['how'], c['how'])})")
    return lines


def choices_from(obj: Any) -> List[dict]:
    """The ``choices`` of a result object or a result dict; ``[]`` when
    there are none."""
    if isinstance(obj, dict):
        out = obj.get("choices")
    else:
        out = getattr(obj, "choices", None)
    return list(out) if isinstance(out, (list, tuple)) else []


#: A bracket that only names inputs or result keys for the agent
#: ("(stress_method 'boussinesq')", "(layer_results)", "(Hdr)"): a printed
#: table leaves it out, so the engineer reads plain words.
_CODE_ITEM = (r"(?:[A-Za-z0-9]*_[A-Za-z0-9_]*(?:\s*=\s*[\w.]+|\s+'[^']*')?"
              r"|[A-Za-z]\w*\s+'[^']*'|'[^']*'|Hdr|cv|KA|Es|candidates)")
_CODE_BRACKET = re.compile(rf"\s*\((?:{_CODE_ITEM})(?:,\s*(?:{_CODE_ITEM}))*\)")


def plain(text: Any) -> str:
    """``text`` with the brackets that only name inputs or result keys
    taken out, for a printed table."""
    return _CODE_BRACKET.sub("", str(text))


def choice_rows(choices: Optional[List[dict]]) -> List[list]:
    """Rows of a printed table of ``choices``: what, chosen, how, basis and
    the alternatives joined by "; ", in plain words (:func:`plain`)."""
    rows = []
    for c in choices or []:
        rows.append([plain(c.get("what", "")), plain(c.get("chosen", "")),
                     HOW_TEXT.get(c.get("how"), str(c.get("how", ""))),
                     plain(c.get("basis", "")),
                     "; ".join(plain(a) for a in c.get("alternatives") or [])])
    return rows


__all__ = ["GIVEN", "DEFAULT", "MODULE", "ASSUMED", "HOW", "HOW_TEXT",
           "choice",
           "how_given", "call_inputs", "CallInputs", "choices_lines",
           "choices_from", "plain", "choice_rows"]
