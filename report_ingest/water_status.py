"""Groundwater 'not measured' kept apart from 'not encountered' (BACKLOG RI-106).

A boring drilled with water or mud and no well has NO groundwater reading:
the hole is full of drilling fluid, so whatever level the fluid stands at
says nothing about the ground. That is not the same as water below the
bottom of the hole, which is what 'not encountered' means, and a record that
filed one as the other told a designer the site was dry. The owner's
checklist (OWN-31) says it plainly: "a boring drilled with water added and
no well has no reading, which does not mean the water is below the bottom of
the hole."

The record holds it as a water row with ``when='not_measured'`` and no depth
(:data:`report_ingest.model.WaterWhen`). This module is everything that
works the same whichever reader filled the record:

* :func:`says_not_measured` -- does a piece of printed text say the water
  level was not measured ("water level not measured", "groundwater not
  recorded, mud rotary", "obscured by drilling fluid")?  Pure words, no
  form's own layout.
* :func:`fluid_method` -- does a printed drilling method put water or mud in
  the hole (mud rotary, wash boring, drilling fluid, water added)?
* :func:`settle_water` -- one investigation's water rows made to say what
  the log printed: a row whose own note says "not measured" is a
  ``not_measured`` row, whatever the reader called it; a hole with no
  reading whose remarks or header say "not measured" gains such a row; and
  a BORING with no water row at all, drilled by a method that fills the
  hole with fluid, gains one too, marked ``derived`` and said in its note.
  The reconciler calls it on every investigation.

What a MODEL is told sits behind :data:`WATER_NOT_MEASURED_ENV` (off by
default): :data:`LOG_READER_PARAGRAPH` for today's log reader and
:data:`MODEL_FIRST_SENTENCE` for the model-first log instruction. Off, both
prompts are byte for byte what they were.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, List, Optional

__all__ = [
    "WATER_NOT_MEASURED_ENV", "water_not_measured_enabled",
    "use_water_not_measured", "LOG_READER_PARAGRAPH", "MODEL_FIRST_SENTENCE",
    "says_not_measured", "says_not_encountered", "fluid_method",
    "settle_water", "NOT_MEASURED",
]

#: The ``WaterLevel.when`` value for a hole whose level was not measured.
NOT_MEASURED = "not_measured"

# ---------------------------------------------------------------------------
# the switch (model-facing text only)
# ---------------------------------------------------------------------------

#: Off by default: tells both log readers about 'not_measured'.
WATER_NOT_MEASURED_ENV = "GEOTECH_INGEST_WATER_NOT_MEASURED"
_FORCED: ContextVar[Optional[bool]] = ContextVar(
    "ingest_water_not_measured", default=None)


def water_not_measured_enabled() -> bool:
    """Whether the log readers are told about 'not_measured' (default:
    off)."""
    forced = _FORCED.get()
    if forced is not None:
        return forced
    return (os.environ.get(WATER_NOT_MEASURED_ENV) or "").strip().lower() in (
        "1", "on", "true", "yes")


@contextmanager
def use_water_not_measured(on: bool = True) -> Iterator[None]:
    """Turn the 'not measured' prompt text on (or off) inside the block."""
    token = _FORCED.set(bool(on))
    try:
        yield
    finally:
        _FORCED.reset(token)


#: Added after the water bullet of today's log reader's system prompt when
#: the switch is on. Written for any form: it names the printed words, not a
#: firm's layout.
LOG_READER_PARAGRAPH = """\
- NOT MEASURED IS NOT NOT ENCOUNTERED. Where the log says the water level
  was not measured, not recorded or not obtained, or could not be read
  because the hole was full of drilling fluid (mud rotary, wash boring,
  water or mud added), give a water entry with when 'not_measured' and a
  null depth, and copy the printed words into its note. Use
  'not_encountered' only where the log says no water was found.
"""

#: Added to the model-first log instruction, after "the water levels", when
#: the switch is on.
MODEL_FIRST_SENTENCE = (
    "Where the log says the water level was not measured (not recorded, not "
    "obtained, or hidden by drilling fluid in a mud rotary or wash boring), "
    "give a water entry with when 'not_measured' and no depth, with the "
    "printed words in its note; 'not_encountered' is only for a log that "
    "says no water was found.")


# ---------------------------------------------------------------------------
# the printed words
# ---------------------------------------------------------------------------

#: A word for the water level.
_WATER = re.compile(
    r"\b(?:ground\s*-?\s*water|water(?:\s+(?:level|table|depth))?|"
    r"g\.?\s*w\.?\s*l\.?|w\.?\s*l\.?|gwt|niveau\s+d'eau|nivel\s+fre[aá]tico|"
    r"nappe)\b", re.I)

#: Words saying a level was NOT taken. "Not observed" is left out on
#: purpose: logs print it for "none seen", which is not encountered.
_NOT_TAKEN = re.compile(
    r"\b(?:not|never|un-?able\s+to\s+be|could\s+not\s+be|cannot\s+be|"
    r"can't\s+be|was\s+not|were\s+not)\s+(?:be\s+)?"
    r"(?:measured|recorded|obtained|determined|monitored|taken|established|"
    r"read|checked|gauged)\b"
    r"|\bun-?able\s+to\s+(?:measure|record|obtain|determine|read)\b"
    r"|\bno\s+(?:water[-\s]+level\s+|groundwater\s+|gwl?\s+)?"
    r"(?:readings?|measurements?)\b"
    r"|\b(?:obscured|masked|hidden)\s+by\s+(?:the\s+)?(?:use\s+of\s+)?"
    r"(?:drilling\s+)?(?:mud|fluid|slurry|water|bentonite)\b"
    r"|\bnot\s+(?:possible|able)\s+to\s+(?:measure|record|read)\b"
    r"|\bn\.?\s*/\s*m\b\.?|\bn\.\s*m\.",
    re.I)

#: "Not encountered" and its usual spellings; a row saying it is not
#: re-filed as "not measured" on a stray word.
_NOT_FOUND = re.compile(
    r"not\s+encountered|none\s+encountered|no\s+(?:free\s+)?(?:ground)?water"
    r"\s+(?:was\s+)?(?:encountered|found|observed)|\bdry\b|\bn\.?\s*/\s*e\b",
    re.I)

#: A drilling method that puts water or mud in the hole.
_FLUID_METHOD = re.compile(
    r"\bmud\b|\bwash(?:ed)?[-\s]*(?:boring|bore|rotary|drill\w*|out)\b|"
    r"\brotary[-\s]+wash\b|\bdrilling\s+(?:mud|fluid)\b|\bslurry\b|"
    r"\bbentonite\b|\bpolymer\b|\bwater\s+(?:added|flush\w*|circulat\w*)\b|"
    r"\b(?:wet|fluid|mud)\s+rotary\b|\blodo\b|\bboue\b", re.I)

_SENTENCE = re.compile(r"[^.;\n]+(?:[.;\n]|$)")


def _sentences(text: str) -> List[str]:
    return [s.strip(" .;\n\t") for s in _SENTENCE.findall(str(text or ""))
            if s.strip(" .;\n\t")]


def says_not_measured(text: Any, water_field: bool = False) -> str:
    """The printed sentence saying the water level was not measured, or
    ``""``.

    ``water_field`` says the text is already about water (a groundwater
    header field, a water row's note), so the not-measured words alone are
    enough there; elsewhere (remarks, other header fields) the sentence must
    also name the water. A sentence that says "not encountered" is not one.
    """
    for sentence in _sentences(text):
        if not _NOT_TAKEN.search(sentence) or _NOT_FOUND.search(sentence):
            continue
        if water_field or _WATER.search(sentence):
            return sentence
    return ""


def says_not_encountered(text: Any) -> bool:
    """Does the text say no water was found?"""
    return bool(_NOT_FOUND.search(str(text or "")))


def fluid_method(text: Any) -> bool:
    """Does a printed drilling method put water or mud in the hole?"""
    return bool(_FLUID_METHOD.search(str(text or "")))


# ---------------------------------------------------------------------------
# one investigation
# ---------------------------------------------------------------------------

#: Kinds a fluid method can leave with no reading. A pit is dug in the open
#: and a sounding is not drilled.
_DRILLED_KINDS = ("boring", "other")

#: A well or piezometer put in the hole: its level is read in the pipe.
_WELL = re.compile(
    r"\b(?:piezometer|standpipe|(?:monitoring|observation)\s+well|"
    r"well\s+(?:install\w*|screen\w*|construct\w*))\b", re.I)

_WATER_KEY = re.compile(r"water|gwl|\bwl\b|nappe|fre[aá]tico", re.I)


def _is_reading(level: Any) -> bool:
    return level.when not in ("not_encountered", NOT_MEASURED) \
        and level.depth is not None


def settle_water(inv: Any) -> List[str]:
    """Make ``inv``'s water rows say what its log printed about 'not
    measured' (module docstring). Changes ``inv`` in place and returns one
    line per change, for a QA note; empty when nothing changed.

    A hole with a reading is left as it is: a level was measured, at some
    point, and the reading is what a designer needs.
    """
    from report_ingest.model import Provenance, WaterLevel

    name = inv.investigation_id or "(no identifier)"
    said: List[str] = []
    for level in inv.water:
        if level.depth is not None or level.when not in (
                "unknown", "not_encountered"):
            continue
        sentence = says_not_measured(level.note, water_field=True)
        if sentence:
            was = level.when
            level.when = NOT_MEASURED
            said.append(f"{name}: a water row read as {was!r} prints "
                        f"{sentence!r}; filed as not measured")
    if any(_is_reading(w) or w.when == NOT_MEASURED for w in inv.water):
        return said
    page = inv.pages[0] if inv.pages else 0
    texts = [inv.remarks] + [f"{k}: {v}" for k, v in
                             sorted((inv.fields or {}).items())]
    sentence = ""
    for key, value in sorted((inv.fields or {}).items()):
        if _WATER_KEY.search(str(key)):
            sentence = says_not_measured(value, water_field=True)
            if sentence:
                break
    if not sentence:
        for text in texts:
            sentence = says_not_measured(text)
            if sentence:
                break
    if sentence and not any(w.when == "not_encountered" for w in inv.water):
        inv.water.append(WaterLevel(
            when=NOT_MEASURED, note=sentence,
            prov=Provenance(page=page, method="derived", confidence=0.6,
                            note="the log prints that the water level was "
                                 "not measured")))
        said.append(f"{name}: the log prints {sentence!r}; a 'not measured' "
                    f"water row added")
        return said
    method = getattr(getattr(inv, "drilling", None), "method", "") or ""
    if not inv.water and inv.kind in _DRILLED_KINDS and fluid_method(method) \
            and not any(_WELL.search(str(t or "")) for t in texts):
        inv.water.append(WaterLevel(
            when=NOT_MEASURED,
            note=(f"no water reading printed; drilled by {method.strip()!r}, "
                  f"which fills the hole with fluid"),
            prov=Provenance(page=page, method="derived", confidence=0.5,
                            note="no water reading on the log, and a "
                                 "drilling method that puts water or mud in "
                                 "the hole")))
        said.append(f"{name}: no water reading printed and drilled by "
                    f"{method.strip()!r}; a 'not measured' water row added "
                    f"(the level of the drilling fluid says nothing of the "
                    f"ground)")
    return said
