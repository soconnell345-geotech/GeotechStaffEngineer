"""Two checks on the narrative answer once it is read, for both readers.

REPEATED LIST ITEMS (BACKLOG RI-178). A reader that sees one recommendation
in the summary, the body and a calculation lists it three times, and one
that names a building in full and then by its abbreviation lists it twice;
today's reader, which reads a long narrative in parts and unions their
lists, does both. :func:`drop_repeats` keeps the first of each and drops
the rest:

* an item the same as another, case and punctuation folded;
* a short name that is another item's bracketed abbreviation ("NOB" beside
  "New Office Building (NOB)");
* a bearing sentence that gives the same pressures, number and unit, as an
  earlier one for the same foundation (footings, a mat, piles, shafts, a
  slab), or that names no foundation: a mat and its footings may be given
  one pressure and are two items;
* in the two hole lists, a hole already listed under another spelling
  (LB1 and LB-1, :func:`report_ingest.reconciler.hole_key`), and a hole
  name with text after it ("B-1 (offset)") when that hole is listed on its
  own and the text names no other hole.

Each drop is a row in the reader's ``unresolved``, so the record says what
went. The builder recounts the structures afterwards (structureCount is the
number of items in structureList).

A SUMMARY OVER ITS LIMIT (BACKLOG RI-179) goes back once to be shortened:
:func:`shorten_summaries` asks one call for every over-long summary of the
reading, keeps a shortened one that is inside its limit, and notes on the
summary's own "past the limit" row what happened. A call that fails or is
refused leaves the summaries as written; it never costs the reading.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from report_ingest.model import (
    GENERAL_FIELDS, SUMMARY_FIELDS, SUMMARY_WORD_LIMITS,
)

__all__ = [
    "PROSE_LISTS", "HOLE_LISTS", "SHORTEN_SYSTEM", "drop_repeats",
    "over_limit", "shorten_summaries",
]

#: The list answers whose items are prose, compared as folded words.
PROSE_LISTS: Tuple[str, ...] = (
    "structureList", "recommendedFoundations", "bearingCapacity",
)
#: The two list answers whose items are hole names.
HOLE_LISTS: Tuple[str, ...] = ("boringDictionary", "testPitDictionary")

#: A bracketed part of an item: "(NOB)".
_RE_BRACKET = re.compile(r"\(([^()]+)\)")
#: A pressure, number and unit, in a bearing sentence.
_RE_PRESSURE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(psf|ksf|tsf|kpa|mpa|kn/m2|kn/m²|"
    r"kips?\s*/\s*(?:ft2|ft²|sq\.?\s*ft)|pounds?[-\s]per[-\s]square[-\s]"
    r"foot)\b", re.IGNORECASE)
#: The foundation a bearing sentence is for, by the words that name it: a
#: mat and a footing given the same pressure are two items.
_FOUNDATION_WORDS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in (
        ("footing", r"\bfootings?\b|\bshallow\s+foundations?\b"),
        ("mat", r"\bmats?\b|\brafts?\b"),
        ("pile", r"\bpil(?:e|es|ing|ings)\b"),
        ("shaft", r"\bshafts?\b|\bcaissons?\b|\bpiers?\b"),
        ("slab", r"\bslabs?\b"),
    ))
#: Words naming one unit of pressure, folded onto its short name.
_UNIT_NAMES = {"poundspersquarefoot": "psf", "poundpersquarefoot": "psf",
               "kipft2": "ksf", "kipsft2": "ksf", "kipft²": "ksf",
               "kipsft²": "ksf", "kipsqft": "ksf", "kipssqft": "ksf",
               "knm2": "kpa", "knm²": "kpa"}
#: A hole name at the head of an item, and what follows it.
_RE_HEAD_HOLE = re.compile(r"\s*([A-Za-z]{1,6}[-\s]?\d+[A-Za-z]?)\b(.+)$")
#: A hole name in the text after one: hyphenated (B-2, Perc-2) or capitals
#: run on to a number (LB2). "offset 10 ft" is neither.
_RE_ANOTHER_HOLE = re.compile(
    r"(?<![A-Za-z])(?:[A-Za-z]{1,6}-\d+|[A-Z]{1,6}\d+)")


def _fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _pressures(text: str) -> frozenset:
    out = set()
    for number, unit in _RE_PRESSURE.findall(str(text or "")):
        unit = re.sub(r"[\s.\-/]+", "", unit.lower())
        out.add((float(number.replace(",", "")),
                 _UNIT_NAMES.get(unit, unit)))
    return frozenset(out)


def _foundations(text: str) -> frozenset:
    return frozenset(name for name, pattern in _FOUNDATION_WORDS
                     if pattern.search(str(text or "")))


def _abbreviations(text: str) -> List[str]:
    return [_fold(part) for part in _RE_BRACKET.findall(str(text or ""))
            if _fold(part)]


def _prose_repeat(name: str, items: Sequence[str], index: int
                  ) -> Optional[str]:
    """Why ``items[index]`` repeats another item, or None."""
    item, earlier = items[index], items[:index]
    folded = _fold(item)
    for other in earlier:
        if _fold(other) == folded:
            return f"the same as '{other}'"
    for at, other in enumerate(items):
        if at != index and folded in _abbreviations(other):
            return f"the abbreviation '{other}' gives"
    if name == "bearingCapacity":
        mine, for_ = _pressures(item), _foundations(item)
        for other in earlier:
            if (mine and _pressures(other) == mine
                    and (not for_ or _foundations(other) == for_)):
                return f"the same pressures as '{other}'"
    return None


def _hole_repeats(items: Sequence[str]) -> Dict[int, str]:
    """``index -> why`` for each hole-list item that repeats another."""
    from report_ingest.reconciler import hole_key

    bare: Dict[str, str] = {}
    for item in items:
        if not _RE_HEAD_HOLE.match(item):
            bare.setdefault(hole_key(item), item)
    out: Dict[int, str] = {}
    seen: Dict[str, str] = {}
    for index, item in enumerate(items):
        head = _RE_HEAD_HOLE.match(item)
        if head is None:
            key = hole_key(item)
            if key in seen:
                out[index] = f"hole '{seen[key]}' is listed already"
            else:
                seen[key] = item
            continue
        key = hole_key(head.group(1))
        if key in bare and not _RE_ANOTHER_HOLE.search(head.group(2)):
            out[index] = f"hole '{bare[key]}' is listed already"
    return out


def drop_repeats(values: Dict[str, Any]
                 ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``values`` with each list's repeated items dropped, and one
    ``unresolved`` row for each item that went. The first of a repeat is
    the one kept. ``values`` is the builder's ``field -> answer``; it is
    not changed in place."""
    out = dict(values)
    rows: List[Dict[str, Any]] = []
    for name in PROSE_LISTS + HOLE_LISTS:
        items = out.get(name)
        if not isinstance(items, list) or len(items) < 2:
            continue
        if name in HOLE_LISTS:
            why_of = _hole_repeats(items)
        else:
            why_of = {}
            for index in range(len(items)):
                why = _prose_repeat(name, items, index)
                if why:
                    why_of[index] = why
        if not why_of:
            continue
        out[name] = [item for index, item in enumerate(items)
                     if index not in why_of]
        for index in sorted(why_of):
            rows.append({"what": name,
                         "why": f"repeats an item: {why_of[index]}; dropped",
                         "value": str(items[index])[:200]})
    return out, rows


# ---------------------------------------------------------------------------
# a summary over its limit
# ---------------------------------------------------------------------------

SHORTEN_SYSTEM = """\
You are shortening summaries of one geotechnical report that came out over
their word limits. Each is given with its limit. Rewrite each one inside its
limit: keep its facts and its numbers, add nothing, and drop the least
important detail first. Answer only the summaries you are given and leave
the others null.
"""


def over_limit(general: Any, hazards: Any) -> Dict[str, Tuple[str, int, int]]:
    """``field -> (text, words, limit)`` for each summary past its limit."""
    out: Dict[str, Tuple[str, int, int]] = {}
    for name in SUMMARY_FIELDS:
        holder = general if name in GENERAL_FIELDS else hazards
        text = " ".join(str(getattr(holder, name, None) or "").split())
        limit = SUMMARY_WORD_LIMITS[name]
        if len(text.split()) > limit:
            out[name] = (text, len(text.split()), limit)
    return out


def _note(unresolved: List[Dict[str, Any]], name: str, words: int,
          limit: int, what_happened: str) -> None:
    """Say on the summary's "past the limit" row what became of it."""
    head = f"{words} words, past the {limit}-word limit"
    for row in unresolved:
        if row.get("what") == name and str(row.get("why", "")).startswith(
                head):
            row["why"] = f"{head}; {what_happened}"
            return
    unresolved.append({"what": name, "why": f"{head}; {what_happened}"})


def shorten_summaries(general: Any, hazards: Any, engine: Any,
                      unresolved: List[Dict[str, Any]],
                      charge: Optional[Callable[[Any], None]] = None) -> int:
    """One call to shorten every summary past its limit; the calls made.

    A shortened summary inside its limit replaces the one written; one still
    over it, or missing, leaves the summary as written. ``charge`` is given
    the reply, for the caller's cost. A call that raises (a ceiling of calls
    reached, a server error) leaves every summary as written and is noted,
    so the reading is never lost to it.
    """
    from report_ingest.engine import text_block, user
    from report_ingest.narrative_reader import SummaryReading

    over = over_limit(general, hazards)
    if not over:
        return 0
    lines: List[str] = []
    for name, (text, words, limit) in over.items():
        lines += [f"{name} ({words} words; its limit is {limit}):", text, ""]
    lines.append("Write the shortened summaries now.")
    try:
        reply = engine.complete([user(text_block("\n".join(lines)))],
                                system=SHORTEN_SYSTEM,
                                output_format=SummaryReading)
    except Exception as exc:                     # never costs the reading
        for name, (_text, words, limit) in over.items():
            _note(unresolved, name, words, limit,
                  f"the call to shorten it did not answer "
                  f"({type(exc).__name__}); kept as written")
        return 0
    if charge is not None:
        charge(reply)
    parsed = None if getattr(reply, "cut_off", False) else reply.parsed
    for name, (_text, words, limit) in over.items():
        new = " ".join(str(getattr(parsed, name, None) or "").split())
        if new and len(new.split()) <= limit:
            holder = general if name in GENERAL_FIELDS else hazards
            setattr(holder, name, new)
            _note(unresolved, name, words, limit,
                  f"shortened once, to {len(new.split())} words")
        else:
            _note(unresolved, name, words, limit,
                  "one call to shorten it did not bring it inside; kept as "
                  "written")
    return 1
