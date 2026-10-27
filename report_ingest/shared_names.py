"""Two holes under one printed name (BACKLOG RI-135).

A report can print two holes under one name: a second series of logs bound
after the first that numbers its holes from 1 again, an earlier firm's
investigation reproduced in an appendix, a phase drilled years later. Once
each log page is read as its own hole (RI-117), the record holds both, and
until this pass nothing said which was which: a laboratory sheet naming
"B-1" went to the first B-1 the record held, the narrative's count took
them for one hole, and the DIGGS file wrote every result naming "B-1" into
the last.

ONE HOLE READ AS SEVERAL LOGS IS NOT TWO HOLES. The reconciler folds a
continuation sheet into the log before it only where it is on the very next
page and is a boring's or a pit's (``reconciler._fold_continuations``). A
later sheet one page further on (the page between was not read as a log), a
sounding's record carried on to the next page, or a sheet whose number was
misread is left as a log of its own under the same name. Before anything
else, the logs of a name are therefore gathered into holes
(:func:`hole_parts`): a log printed after another of its name, of the same
kind, in the same report section, whose depths do not overlap the other's,
is a later sheet of that hole where one of these says so:

* it prints sheet k of n, k above 1, of a set the earlier log's own sheet
  number does not rule out;
* its title is the earlier log's with only "continued" ("cont.", "cont'd")
  added;
* its depths carry on from where the earlier log's leave off, deeper (for a
  sounding, its readings resume where the earlier ones stop), and nothing
  printed says it is another hole.

Logs whose depths overlap (two holes, each logged down from the ground) are
two holes, and so are logs that, with no printed sheet number or title
saying they are later sheets, print a first sheet or a sheet of another
set, or differ in what their pages print (the facts below). Depths that only
carry on do not outweigh these: a second hole whose first sample was taken
below the first hole's last is not its later sheet. Two holes are kept apart
only where something printed says so: logs nothing tells apart either way
(a page the readers could read nothing off) are taken as one hole, as
before. Logs of one name taken as one hole's sheets are kept as the logs
they were read as, counted as one hole, and one QA entry, ``open``, says
they may be one hole's sheets the fold did not join. A laboratory result
naming that name is linked to the log holding its sample (where two logs
hold a sample of the name it prints, the one whose depth agrees too), else
to the first, as before (:func:`report_ingest.reconciler.reconcile`); the
link names that log by its pages (``linked_log_pages``), and the DIGGS file
writes the result in that log's feature.

THE PRINTED NAME IS NEVER CHANGED. Each hole keeps the identifier its log
printed. Where two or more holes remain under one name, this pass adds a
QUALIFIER kept beside it (``Investigation.qualifier``; every log of one hole
carries its hole's), read off what the holes' own pages print, in this
order, the first that tells every one of them from the others:

* the bound report its pages sit in (``record.bound_documents``; in the
  ingest graph a report's own record is reconciled before its bound reports
  are listed on it, and each bound report is reconciled in a record of its
  own, so there this fact is never known: it tells holes apart only in a
  record reconciled again after the run);
* the report section they sit under: the appendix tab before the first page
  (planlens' outline);
* the title the log prints naming the hole ("Log of Boring QX-1", "Stratum
  chart, test hole 1"), which a series prints on every log: its numbers
  masked, two series' titles differ, one log printed twice does not;
* the project, job, contract, client, site or phase its header prints;
* the firm (the drilling contractor, or a firm, company, consultant or
  engineer field);
* the kind of exploration (a boring, a test pit);
* the date it was drilled, compared only where no reading of the day and the
  month makes the two one day;
* where it was drilled: its coordinates, or its station and offset;
* its ground elevation.

A hole's total depth is not among them: a log cut off where its next sheet
was not read prints a shallower depth than the hole's.

A fact counts only where every hole of the name prints it and no two print
it alike: a value one log left blank tells nothing. Where nothing printed
tells them apart, they are still kept as two holes -- pages of two places in
a report are never merged into one hole by their name alone -- each
qualified by the pages it is printed on, and the QA entry says that one may
be the other printed again.

Each name gets one QA entry, ``open``: a reviewer should see that the
report prints two holes under one name, and what tells them apart.

THE LABORATORY LINK DOES NOT GUESS BETWEEN THEM. A sheet or a summary row
naming such a name is linked to one of the holes only where it prints what
tells them apart: its pages under that hole's report section or bound
report, the hole's project or firm in its own fields or label, the date the
hole was drilled as its sample date (where the date can be read only one
way), or the hole's station or coordinates (:meth:`Twins.pick`). An
elevation a sheet prints is not used: what it is the elevation of is not
defined. Otherwise the result is linked to neither, and said
(:func:`report_ingest.reconciler.reconcile`).

LAYOUT JUDGMENT: the words that mark a title naming a hole, the field names
read as a project or a firm, and the tolerances below are read off log forms
and the trade's habits, not taken from any standard.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import (
    Any, Dict, FrozenSet, List, Optional, Sequence, Tuple,
)

__all__ = [
    "FACT_ORDER", "Fact", "PagePrint", "Twins", "name_groups", "hole_facts",
    "hole_parts", "tell_apart", "hole_identity", "pages_span",
]

#: The facts that may tell two holes of one name apart, first first.
FACT_ORDER: Tuple[str, ...] = (
    "bound report", "report section", "title", "project", "firm", "kind",
    "date", "location", "ground elevation")

#: The facts a laboratory sheet or a summary row may print to be linked to
#: one hole of a shared name (:meth:`Twins.pick`).
LAB_FACTS: Tuple[str, ...] = (
    "bound report", "report section", "project", "firm", "date", "location")

#: Two titles whose numbers are masked are one series' when they are at
#: least this alike (difflib's ratio): a scan reads one title a letter
#: differently from sheet to sheet. LAYOUT JUDGMENT.
TITLE_ALIKE = 0.8
#: Two printed values (a project, a firm) are the same when at least this
#: alike once case, spacing and punctuation are dropped. LAYOUT JUDGMENT.
VALUE_ALIKE = 0.85
#: Two ground elevations closer than this, in metres, are one value printed
#: twice (a datum's rounding).
LEVEL_TOL_M = 0.05
#: Two coordinates closer than this are one place: a hundred-thousandth of a
#: degree where both read as degrees, half a unit (a metre or a foot) where
#: they read as grid coordinates.
DEGREE_TOL = 1e-5
GRID_TOL = 0.5
#: The longest title line read as a log's title, in characters.
TITLE_MAX_CHARS = 90
#: How much of a printed value a qualifier shows, in characters.
SHOWN_CHARS = 60

#: A word that makes a line a title naming a hole, lower case. LAYOUT
#: JUDGMENT, in the corpus's languages.
_TITLE_WORD = re.compile(
    r"(?<![a-z])(?:log|logs|bor(?:e|ing|ehole)s?|holes?|pits?|trench(?:es)?|"
    r"profile|sondages?|sondeos?|forages?|calicatas?|perforaci[oó]n|"
    r"bohrung|schurf)(?![a-z])", re.I)
#: A header field named for the project a log belongs to (case, spacing and
#: punctuation dropped from the field's name).
_PROJECT_KEY = re.compile(r"project|job|contract(?!or)|client|site|phase|"
                          r"stage")
#: A header field named for a firm.
_FIRM_KEY = re.compile(r"firm|company|consultant|contractor|engineer|"
                       r"drilledby|preparedby")
#: A field a laboratory sheet names a sample's or a hole's date by.
_SAMPLED_KEY = re.compile(r"sampl|drill|boring|bore|hole|collect")
_STATION_KEY = re.compile(r"station|chainage|^sta$")
_NORTH_KEY = re.compile(r"northing|^north$|^n$|^y$|latitude|^lat$")
_EAST_KEY = re.compile(r"easting|^east$|^e$|^x$|longitude|^lon$|^long$")
#: The word a continuation sheet's title adds: "(continued)", "cont.",
#: "(cont'd)", "contd". LAYOUT JUDGMENT.
_CONTINUED = re.compile(
    r"[(\[]?\s*\b(?:continued|cont['’]?d|cont)\b\.?\s*[)\]]?", re.I)
#: A month printed as a word, by its first three letters. LAYOUT JUDGMENT,
#: English only: a date whose month is another language's word is read by
#: its numbers alone.
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _clip(text: Any, n: int = SHOWN_CHARS) -> str:
    text = _flat(text)
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _plain(text: Any) -> str:
    """Lower case, letters and digits only: for comparing printed values."""
    return re.sub(r"[^a-z0-9]", "", _flat(text).lower())


def _alike(a: str, b: str, at: float) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= at


def pages_span(pages: Sequence[int]) -> str:
    """``[3, 4, 5, 9]`` as "pages 3-5, 9", ``[3]`` as "page 3"."""
    pages = sorted({int(p) for p in pages})
    if not pages:
        return "no page"
    runs: List[List[int]] = []
    for page in pages:
        if runs and page == runs[-1][-1] + 1:
            runs[-1].append(page)
        else:
            runs.append([page])
    shown = [f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs]
    return ("pages " if len(pages) > 1 else "page ") + ", ".join(shown)


def hole_identity(inv: Any) -> str:
    """What makes a hole ONE hole when holes are counted: its name with
    leading zeros folded (:func:`report_ingest.reconciler.hole_key`), and
    the qualifier this pass gave it, where it shares its name with another
    hole. ``""`` for a hole that printed no name."""
    from report_ingest.reconciler import hole_key

    key = hole_key(getattr(inv, "investigation_id", ""))
    if not key:
        return ""
    qualifier = getattr(inv, "qualifier", "") or ""
    return f"{key}|{qualifier}" if qualifier else key


# ---------------------------------------------------------------------------
# what a page prints: its report section, and the title naming its hole
# ---------------------------------------------------------------------------

class PagePrint:
    """What the pages print about where they sit and what they are titled.

    ``doc`` is the planlens document (its pages' text lines) and
    ``outline`` planlens' :func:`~planlens.document.roles.document_outline`
    of it; either may be None, and then the facts it gives are unknown.
    """

    def __init__(self, doc: Any = None, outline: Any = None) -> None:
        self.doc = doc
        marks: List[Tuple[int, str]] = []
        for mark in getattr(outline, "dividers", None) or ():
            if getattr(mark, "page", None) is not None:
                marks.append((int(mark.page), _flat(mark.text)))
        if not marks:
            for entry in (outline.appendices if outline is not None
                          and hasattr(outline, "appendices") else ()):
                if getattr(entry, "page", None) is not None:
                    marks.append((int(entry.page), _flat(
                        " ".join(x for x in (f"appendix {entry.number}"
                                             if entry.number else "",
                                             entry.title) if x))))
        self.marks = sorted(set(marks))
        self._lines: Dict[int, List[str]] = {}

    def section(self, page: int) -> Optional[Tuple[int, str]]:
        """``(the page of the tab, the tab's words)`` for the report
        section ``page`` sits in: the last appendix tab at or before it;
        ``(-1, ...)`` for a page before the first tab. None when the report
        prints no tab."""
        if not self.marks:
            return None
        found = None
        for at, text in self.marks:
            if at <= int(page):
                found = (at, text)
        return found or (-1, "the report before its first appendix tab")

    def lines(self, page: int) -> List[str]:
        """The page's text lines, top to bottom; none when the page will
        not read."""
        page = int(page)
        if page not in self._lines:
            out: List[str] = []
            if self.doc is not None:
                try:
                    content = self.doc.page(page)
                    rows = [ln for ln in content.lines
                            if _flat(getattr(ln, "text", ""))]
                    rows.sort(key=lambda ln: (float(ln.bbox[1]),
                                              float(ln.bbox[0])))
                    out = [_flat(ln.text) for ln in rows]
                except Exception:              # a page that will not read
                    out = []
            self._lines[page] = out
        return self._lines[page]

    def titles(self, pages: Sequence[int], name: str) -> List[str]:
        """The lines of a hole's first page that name it as a title does: a
        word for a hole or a log beside the hole's own number ("Log of
        Boring QX-1", "STRATUM CHART, TEST HOLE 1", "Boring No.: QX-1"), no
        longer than :data:`TITLE_MAX_CHARS`. Empty where the name has no
        number or the page is not to hand."""
        digits = re.findall(r"\d+", str(name or ""))
        if not digits or not pages:
            return []
        number = int(digits[-1])
        out: List[str] = []
        for text in self.lines(min(int(p) for p in pages)):
            if len(text) > TITLE_MAX_CHARS or not _TITLE_WORD.search(text):
                continue
            if any(int(d) == number for d in re.findall(r"\d+", text)):
                out.append(text)
        return out


def _masked(title: str) -> str:
    """A title with its numbers masked and a continuation sheet's
    "(continued)" dropped, letters only: one series' titles are then one
    text whatever hole or sheet each names."""
    title = _CONTINUED.sub(" ", title.lower())
    return re.sub(r"[^a-z#]", "", re.sub(r"\d+", "#", title))


# ---------------------------------------------------------------------------
# the facts each hole's pages print
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    """One thing a hole's pages print that may tell it from another hole of
    its name. ``shown`` is the words a qualifier and a QA entry use, with
    the value as printed; ``key`` is what two holes' facts are compared
    by."""

    what: str
    shown: str
    key: Any


def _si_m(quantity: Any) -> Optional[float]:
    """A length or an elevation in metres; None when there is none or it
    will not convert."""
    if quantity is None:
        return None
    try:
        got = quantity.to_si(kind="length")
    except Exception:                          # a unit with no conversion
        return None
    if got is None or got.unit != "m":
        return None
    return float(got.value)


def _printed(quantity: Any) -> str:
    unit = str(getattr(quantity, "unit", "") or "").strip()
    return f"{float(quantity.value):g}" + (f" {unit}" if unit else "")


#: One day a printed date may be: ``(the year's last two digits, the
#: month, the day)``, each None where the print does not give it.
Day = Tuple[Optional[int], Optional[int], Optional[int]]


def _date_key(text: str) -> Optional[FrozenSet[Day]]:
    """The days a printed date may be, so "10/29/47" and "10/29/2047" are
    one day.

    The order of the day and the month is not guessed. Where the month is a
    word ("29 Oct 2047"), a four-digit year leads ("2047-10-29"), or one of
    the first two numbers is above 12, the date is one day; where either of
    the first two could be the month ("3/8/47") it is both days. A year
    alone, or a month and a year, leaves the rest None. None for a date that
    prints no number ("n/a"), or numbers that make no day.
    """
    low = str(text or "").lower()
    month = next((n for word, n in _MONTHS.items()
                  if re.search(rf"(?<![a-z]){word}", low)), None)
    numbers = [int(n) for n in re.findall(r"\d+", low)]
    if not numbers:
        return None
    days: set = set()
    if month is not None:
        years = [n for n in numbers if n > 31]
        if years:
            rest = [n for n in numbers if n <= 31]
            year, day = years[0], (rest[0] if rest else None)
        elif len(numbers) >= 2:
            day, year = numbers[0], numbers[-1]
        else:
            day, year = numbers[0], None
        days.add((year % 100 if year is not None else None, month, day))
    elif numbers[0] >= 1000:
        tail: List[Optional[int]] = list(numbers[1:3]) + [None, None]
        days.add((numbers[0] % 100, tail[0], tail[1]))
    elif len(numbers) >= 3:
        a, b, year = numbers[0], numbers[1], numbers[2] % 100
        days.update(((year, a, b), (year, b, a)))
    elif len(numbers) == 2 and numbers[1] >= 1000:
        days.add((numbers[1] % 100, numbers[0], None))
    elif len(numbers) == 1 and numbers[0] >= 1000:
        days.add((numbers[0] % 100, None, None))
    days = {d for d in days if (d[1] is None or 1 <= d[1] <= 12)
            and (d[2] is None or 1 <= d[2] <= 31)}
    return frozenset(days) or None


def _maybe_one_day(a: Day, b: Day) -> bool:
    """May these two be one day: alike in every part both give?"""
    return all(x == y for x, y in zip(a, b)
               if x is not None and y is not None)


def _one_day(a: Optional[FrozenSet[Day]],
             b: Optional[FrozenSet[Day]]) -> bool:
    """Are these two printed dates certainly one day: each read only one
    way, in full, and the same?"""
    if not a or not b or len(a) != 1 or len(b) != 1:
        return False
    (x,), (y,) = a, b
    return None not in x and x == y


def _field_facts(fields: Dict[str, str], pattern: Any
                 ) -> Dict[str, Tuple[str, str]]:
    """``plain key -> (the key as printed, the value)`` for the fields
    whose name matches ``pattern``."""
    out: Dict[str, Tuple[str, str]] = {}
    for key, value in (fields or {}).items():
        plain = re.sub(r"[^a-z]", "", str(key).lower())
        if plain and pattern.search(plain) and _plain(value):
            out.setdefault(plain, (str(key).strip().rstrip(":"),
                                   _flat(value)))
    return out


def hole_facts(inv: Any, record: Any = None,
               page_print: Optional[PagePrint] = None) -> Dict[str, Any]:
    """Every fact of :data:`FACT_ORDER` this hole's pages print. Values
    are a :class:`Fact`, except ``project`` and ``firm``, which are
    ``{plain field name: Fact}`` (a header may print several)."""
    out: Dict[str, Any] = {}
    pages = [int(p) for p in (getattr(inv, "pages", None) or ())]
    bound = [row for row in (getattr(record, "bound_documents", None) or ())
             if pages and all(row.first_page <= p <= row.last_page
                              for p in pages)]
    if bound or any(getattr(record, "bound_documents", None) or ()):
        row = bound[0] if bound else None
        out["bound report"] = Fact(
            "bound report",
            f"in bound report {row.bound_id}" + (
                f" ({_clip(row.title)})" if row.title else "")
            if row is not None else "in the report itself, not a bound one",
            row.bound_id if row is not None else "")
    if page_print is not None and pages:
        section = page_print.section(min(pages))
        if section is not None:
            out["report section"] = Fact(
                "report section",
                f"under '{_clip(section[1])}'" if section[0] >= 0
                else "before the first appendix tab", section[0])
        titles = page_print.titles(pages, inv.investigation_id)
        if titles:
            out["title"] = Fact("title", f"titled '{_clip(titles[0])}'",
                                frozenset(_masked(t) for t in titles))
    fields = dict(getattr(inv, "fields", None) or {})
    project = {key: Fact("project", f"{label}: {_clip(value)}",
                         _plain(value))
               for key, (label, value)
               in _field_facts(fields, _PROJECT_KEY).items()}
    if project:
        out["project"] = project
    firm = {key: Fact("firm", f"{label}: {_clip(value)}", _plain(value))
            for key, (label, value)
            in _field_facts(fields, _FIRM_KEY).items()}
    contractor = _flat(getattr(getattr(inv, "drilling", None), "contractor",
                               ""))
    if contractor:
        firm.setdefault("contractor", Fact(
            "firm", f"drilled by {_clip(contractor)}", _plain(contractor)))
    if firm:
        out["firm"] = firm
    kind = str(getattr(inv, "kind", "") or "")
    if kind:
        out["kind"] = Fact("kind", f"a {kind.replace('_', ' ')}", kind)
    for slot, word in (("date_started", "started"),
                       ("date_finished", "finished")):
        text = _flat(getattr(inv, slot, ""))
        if text and _date_key(text) is not None:
            out["date"] = Fact("date", f"{word} {_clip(text)}",
                               (word, _date_key(text)))
            break
    x, y = getattr(inv, "x", None), getattr(inv, "y", None)
    if x is not None and y is not None:
        out["location"] = Fact("location", f"at {x:g}, {y:g}",
                               ("xy", float(x), float(y)))
    elif _flat(getattr(inv, "station", "")):
        station = _flat(inv.station)
        offset = _flat(getattr(inv, "offset", ""))
        out["location"] = Fact(
            "location", f"at station {_clip(station)}"
            + (f", offset {_clip(offset)}" if offset else ""),
            ("station", _plain(station), _plain(offset)))
    level = _si_m(getattr(inv, "elevation", None))
    if level is not None:
        out["ground elevation"] = Fact(
            "ground elevation", f"ground elevation {_printed(inv.elevation)}",
            level)
    return out


def _apart(a: Fact, b: Fact) -> bool:
    """Do these two facts of one kind tell two holes apart? A place printed
    as coordinates on one log and as a station on the other does not: the
    two say different things, not different places. Nor does a date that,
    read one way or the other ("3/8/47", "8/3/47"), may be the other's
    day."""
    if a.what == "title":
        return not any(_alike(x, y, TITLE_ALIKE)
                       for x in a.key for y in b.key)
    if a.what == "ground elevation":
        return abs(float(a.key) - float(b.key)) > LEVEL_TOL_M
    if a.what == "location":
        if a.key[0] != b.key[0]:
            return False
        if a.key[0] == "xy":
            values = a.key[1:] + b.key[1:]
            tol = DEGREE_TOL if all(abs(v) <= 360.0 for v in values) \
                else GRID_TOL
            return max(abs(a.key[1] - b.key[1]),
                       abs(a.key[2] - b.key[2])) > tol
        return a.key != b.key
    if a.what in ("project", "firm"):
        return not _alike(a.key, b.key, VALUE_ALIKE)
    if a.what == "date":
        # a start beside a finish says nothing
        return a.key[0] == b.key[0] and not any(
            _maybe_one_day(x, y) for x in a.key[1] for y in b.key[1])
    return a.key != b.key


def _all_apart(facts: Sequence[Optional[Fact]]) -> bool:
    """Does every hole print this fact, and no two alike?"""
    if any(f is None for f in facts):
        return False
    return all(_apart(a, b) for i, a in enumerate(facts)   # type: ignore
               for b in facts[i + 1:])


def _telling(facts: Sequence[Dict[str, Any]], what: str
             ) -> Optional[List[Fact]]:
    """The facts of kind ``what``, one per hole, that tell every hole of
    the group from the others; None where that kind does not."""
    if what in ("project", "firm"):
        keys = set.intersection(*[set(f.get(what) or {}) for f in facts])
        for key in sorted(keys):
            chosen = [f[what][key] for f in facts]
            if _all_apart(chosen):
                return chosen
        return None
    chosen = [f.get(what) for f in facts]
    return chosen if _all_apart(chosen) else None   # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# the groups, and the pass
# ---------------------------------------------------------------------------

def name_groups(investigations: Sequence[Any]) -> List[List[Any]]:
    """The holes printed under one name, two or more to a name, in the
    order the record holds them: one name is one name written alike
    (:func:`report_ingest.reconciler.fold_id`), so "B-01" and "B-1" are not
    a group here (the reconciler keeps those apart already)."""
    from report_ingest.reconciler import fold_id

    groups: Dict[str, List[Any]] = {}
    for inv in investigations:
        key = fold_id(getattr(inv, "investigation_id", ""))
        if key:
            groups.setdefault(key, []).append(inv)
    return [group for group in groups.values() if len(group) > 1]


# ---------------------------------------------------------------------------
# one hole read as several logs
# ---------------------------------------------------------------------------

def _readings_range(inv: Any) -> Optional[Tuple[float, float]]:
    """(shallowest, deepest) of a sounding's readings, in metres: a cone's
    points where it is plotted against depth, a probe's drives."""
    from report_ingest.reconciler import depth_m

    depths: List[Any] = []
    cpt = getattr(inv, "cpt", None)
    if cpt is not None and getattr(cpt, "vertical_axis", "depth") == "depth":
        depths += [p.depth for p in cpt.points]
    dcp = getattr(inv, "dcp", None)
    if dcp is not None:
        depths += [p.depth for p in dcp.points]
        depths += [p.depth_top for p in dcp.points if p.depth_top is not None]
    found = [d for d in (depth_m(q) for q in depths)
             if d is not None and d >= 0]
    return (min(found), max(found)) if found else None


def _reaches(inv: Any) -> Dict[str, Tuple[float, float]]:
    """How deep a log's rows reach, ``{"rows": (top, bottom)}`` from its
    samples, driven records or layers
    (:func:`report_ingest.reconciler._depth_range`), and ``{"readings":
    ...}`` from a sounding's readings."""
    from report_ingest.reconciler import _depth_range

    out: Dict[str, Tuple[float, float]] = {}
    rows = _depth_range(inv)
    if rows is not None:
        out["rows"] = rows
    readings = _readings_range(inv)
    if readings is not None:
        out["readings"] = readings
    return out


def _span(parts: Sequence[Any], what: str) -> Optional[Tuple[float, float]]:
    got = [r for r in (_reaches(p).get(what) for p in parts) if r]
    return (min(r[0] for r in got), max(r[1] for r in got)) if got else None


def _bound_of(page: int, bound: Sequence[Tuple[int, int, str]]) -> str:
    return next((bid for first, last, bid in bound if first <= page <= last),
                "")


def _continues_part(earlier: Sequence[Any], later: Any,
                    page_print: Optional[PagePrint] = None,
                    record: Any = None) -> Tuple[str, str]:
    """Is ``later``, a log printed under the name of the logs ``earlier``
    (one hole's, page order), a later sheet of that hole the fold did not
    join, or another hole? ``(verdict, why)``, the verdict one of:

    * ``"two"``: another hole. It is of another kind or sits in another
      report section or bound report; or its depths overlap theirs (two
      holes, each logged down from the ground); or, with no printed sheet
      number or title below saying it is a later sheet, something printed
      says it is another hole: it prints a first sheet or a sheet of
      another set, or its pages print a project, firm, date, place, ground
      elevation or title that differs from the first earlier log's. That
      outweighs depths that only carry on: a second hole whose first
      samples were taken below the first hole's last is not its later
      sheet.
    * ``"continues"``: a later sheet, starting after the earlier logs end,
      where it prints sheet k of n, k above 1, of a set the last earlier
      log's own sheet number does not rule out, or its title is the first
      earlier log's with only "continued" added; or where its depths carry
      on from where theirs leave off, deeper (a sounding's readings resume
      where theirs stop), and nothing printed says it is another hole.
      ``why`` says which.
    * ``"unknown"``: nothing its pages print says either (a page the
      readers could read nothing off).
    """
    from report_ingest.reconciler import (
        CONTINUATION_OVERLAP, _sheet_number,
    )

    head, last = earlier[0], earlier[-1]
    if getattr(later, "kind", "") != getattr(head, "kind", ""):
        return "two", ""
    before = [int(p) for inv in earlier for p in (inv.pages or ())]
    pages = [int(p) for p in (later.pages or ())]
    if before and pages:
        if page_print is not None and page_print.section(min(before)) != \
                page_print.section(min(pages)):
            return "two", ""
        bound = [(int(row.first_page), int(row.last_page), row.bound_id)
                 for row in (getattr(record, "bound_documents", None) or ())]
        if bound and _bound_of(min(before), bound) != _bound_of(min(pages),
                                                                bound):
            return "two", ""
    after = bool(before and pages and min(pages) > max(before))
    deeper: List[str] = []           # its depths carry on from theirs
    for what, words in (("rows", ("its depths", "leave off")),
                        ("readings", ("its readings", "stop"))):
        here, there = _span(earlier, what), _reaches(later).get(what)
        if here is None or there is None:
            continue
        reach = CONTINUATION_OVERLAP * (here[1] - here[0])
        if there[0] < here[1] - reach:
            return "two", ""        # overlapping: two holes from the ground
        if after and there[1] > here[1]:
            deeper.append(f"{words[0]} ({there[0]:.2f}-{there[1]:.2f} m) "
                          f"carry on from where the earlier log's "
                          f"({here[0]:.2f}-{here[1]:.2f} m) {words[1]}")
        break
    mine, theirs = (_sheet_number(getattr(last, "sheet", "")),
                    _sheet_number(getattr(later, "sheet", "")))
    # A first sheet, or a sheet of another set, starts another hole's.
    other_set = theirs is not None and (theirs[0] == 1 or (
        mine is not None and (mine[1] != theirs[1] or mine[0] >= theirs[0])))
    printed: List[str] = []          # its sheet number or title says so
    if after and theirs is not None and not other_set:
        printed.append(f"it prints sheet {theirs[0]} of {theirs[1]}")
    if after and page_print is not None:
        first = {_masked(t) for t in page_print.titles(
            head.pages, head.investigation_id)}
        for title in page_print.titles(pages, later.investigation_id):
            word = _CONTINUED.search(title)
            if word is None:
                continue
            if any(_alike(_masked(title), t, TITLE_ALIKE) for t in first):
                printed.append(f"its title is the earlier log's with only "
                               f"'{_flat(word.group(0))}' added")
                break
    if printed:
        return "continues", "; ".join(deeper + printed)
    # Something printed that says another hole outweighs depths that only
    # carry on: a second hole whose first sample is deeper than the first
    # hole's last is not its later sheet.
    if other_set:
        return "two", ""
    facts = [hole_facts(head, record, page_print),
             hole_facts(later, record, page_print)]
    if any(_telling(facts, what) is not None for what in FACT_ORDER):
        return "two", ""
    if deeper:
        return "continues", "; ".join(deeper)
    return "unknown", (f"nothing its pages print says whether it is a later "
                       f"sheet of the log before it or another hole")


def hole_parts(group: Sequence[Any], page_print: Optional[PagePrint] = None,
               record: Any = None
               ) -> Tuple[List[List[Any]], Dict[int, str], set]:
    """The logs printed under one name gathered into holes: ``(the holes,
    each its logs in page order; id(log) -> why it is taken as a later sheet
    of its hole; the id() of those taken so only because nothing says they
    are another hole)``.

    A log joins the latest hole of its name it continues
    (:func:`_continues_part`); else the latest one nothing tells it apart
    from -- two holes are kept apart only where something printed says so,
    their depths overlapping or their pages printing a different series --
    else it starts a hole of its own. Nothing is merged: each log stays the
    log it was read as."""
    ordered = sorted(group, key=lambda inv: (min(inv.pages) if inv.pages
                                             else 10 ** 9))
    holes: List[List[Any]] = []
    why: Dict[int, str] = {}
    unsure: set = set()
    for inv in ordered:
        latest = sorted(holes, key=lambda h: -max(
            (int(p) for x in h for p in x.pages), default=-1))
        verdicts = [(hole, _continues_part(hole, inv, page_print, record))
                    for hole in latest]
        chosen = next(((hole, said) for hole, (verdict, said) in verdicts
                       if verdict == "continues"), None)
        if chosen is None:
            chosen = next(((hole, said) for hole, (verdict, said) in verdicts
                           if verdict == "unknown"), None)
            if chosen is not None:
                unsure.add(id(inv))
        if chosen is None:
            holes.append([inv])
            continue
        chosen[0].append(inv)
        why[id(inv)] = chosen[1]
    return holes, why, unsure


class Twins:
    """The logs of a record that share a name, gathered into holes
    (:func:`hole_parts`), each hole qualified where two or more remain, and
    how a laboratory sheet is placed among them. Built by
    :func:`tell_apart`."""

    def __init__(self, page_print: Optional[PagePrint] = None,
                 bound: Sequence[Any] = ()) -> None:
        self.page_print = page_print
        #: The record's bound reports: ``(first page, last page, id)``.
        self.bound = [(int(row.first_page), int(row.last_page), row.bound_id)
                      for row in bound]
        #: ``fold_id(name) -> the logs``, two or more.
        self.groups: Dict[str, List[Any]] = {}
        #: ``fold_id(name) -> the holes``, each its logs in page order.
        self.holes: Dict[str, List[List[Any]]] = {}
        #: ``id(log) -> the hole it is a log of``.
        self._hole: Dict[int, List[Any]] = {}
        #: ``id(a hole's first log) -> the hole's facts`` (:func:`hole_facts`).
        self.facts: Dict[int, Dict[str, Any]] = {}

    def __bool__(self) -> bool:
        return bool(self.groups)

    def group(self, inv: Any) -> List[Any]:
        """The logs sharing ``inv``'s name, ``inv`` among them; empty when
        its name is its own."""
        from report_ingest.reconciler import fold_id

        group = self.groups.get(fold_id(getattr(inv, "investigation_id", "")))
        return group if group and any(inv is g for g in group) else []

    def holes_of(self, inv: Any) -> List[List[Any]]:
        """The holes printed under ``inv``'s name, each its logs; empty when
        its name is its own. One hole where every log of the name reads as
        one hole's sheets."""
        from report_ingest.reconciler import fold_id

        if not self.group(inv):
            return []
        return self.holes.get(
            fold_id(getattr(inv, "investigation_id", "")), [])

    def parts(self, inv: Any) -> List[Any]:
        """The logs of ``inv``'s hole, page order; ``[inv]`` when its name
        is its own."""
        return self._hole.get(id(inv), [inv])

    def pick(self, inv: Any, test: Any, pages: Sequence[int]
             ) -> Tuple[Optional[Any], List[str]]:
        """``(the hole's first log, what the sheet printed that says so)``
        for a sheet or a summary row the link placed in ``inv``, a log
        sharing its name with another hole's; ``(None, [])`` where the sheet
        prints nothing that tells which hole (or points at two). Where every
        log of the name is one hole's, that hole's first log and no reason.
        ``test`` is the sheet or the row, ``pages`` the pages it is printed
        on."""
        holes = self.holes_of(inv)
        if not holes:
            return inv, []
        if len(holes) == 1:
            return holes[0][0], []
        facts = [self.facts.get(id(hole[0]), {}) for hole in holes]
        votes: Dict[int, List[str]] = {}

        def vote(index: int, why: str) -> None:
            votes.setdefault(index, []).append(why)

        texts = _sheet_texts(test)
        for what in LAB_FACTS:
            telling = _telling(facts, what)
            if telling is None:
                continue
            if what in ("bound report", "report section"):
                hits = self._section_hits(what, telling, pages)
                for i in hits:
                    vote(i, f"its pages are {telling[i].shown}")
            elif what in ("project", "firm"):
                for i, fact in enumerate(telling):
                    if _value_printed(fact.key, texts):
                        vote(i, f"it prints {fact.shown}")
            elif what == "date":
                said = _sample_dates(test)
                for i, fact in enumerate(telling):
                    if any(_one_day(fact.key[1], day) for day in said):
                        vote(i, f"its sample date is the hole's, "
                                f"{fact.shown}")
            elif what == "location":
                for i, fact in enumerate(telling):
                    if _location_printed(fact, test):
                        vote(i, f"it prints the hole's place, {fact.shown}")
        if len(votes) != 1:
            return None, []
        (index, why), = votes.items()
        return holes[index][0], why

    def _section_hits(self, what: str, telling: Sequence[Fact],
                      pages: Sequence[int]) -> List[int]:
        """The holes whose report section (or bound report) the sheet's
        pages sit in, all of them."""
        if not pages:
            return []
        if what == "report section":
            if self.page_print is None:
                return []
            sections = {self.page_print.section(int(p)) for p in pages}
            if len(sections) != 1 or None in sections:
                return []
            (section,) = sections
            return [i for i, f in enumerate(telling) if f.key == section[0]]
        held = {next((bid for first, last, bid in self.bound
                      if first <= int(p) <= last), "") for p in pages}
        if len(held) != 1 or not self.bound:
            return []
        (bid,) = held
        return [i for i, f in enumerate(telling) if f.key == bid]


def _sheet_texts(test: Any) -> List[str]:
    """What a sheet or a row prints about whose it is: its fields' values,
    its laboratory, its labels, its note, plain."""
    out = [_plain(value) for _name, value in _printed_fields(test)]
    out += [_plain(getattr(test, name, "")) for name in
            ("lab", "investigation_id", "sample_id", "note", "description")]
    return [t for t in out if t]


def _value_printed(value: str, texts: Sequence[str]) -> bool:
    """Is a hole's project or firm among what a sheet prints: one inside
    the other, the shorter at least five letters long."""
    if len(value) < 5:
        return False
    return any(value in text or (len(text) >= 5 and text in value)
               for text in texts)


def _sample_dates(test: Any) -> List[Any]:
    """The dates a sheet prints under a field naming a sample's or a hole's
    date ("Date sampled", "Boring date"), as :func:`_date_key` keys."""
    out = []
    for key, value in _printed_fields(test):
        plain = re.sub(r"[^a-z]", "", str(key).lower())
        if "date" in plain and _SAMPLED_KEY.search(plain):
            day = _date_key(str(value))
            if day is not None:
                out.append(day)
    return out


def _location_printed(fact: Fact, test: Any) -> bool:
    """Does a sheet print a hole's station, or its coordinates?"""
    fields = {re.sub(r"[^a-z]", "", str(k).lower()): str(v)
              for k, v in _printed_fields(test)}
    if fact.key[0] == "station":
        return bool(fact.key[1]) and any(
            _STATION_KEY.search(k) and _plain(v) == fact.key[1]
            for k, v in fields.items())

    def number(pattern: Any) -> Optional[float]:
        for key, value in fields.items():
            if pattern.search(key):
                found = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
                if found:
                    return float(found.group(0))
        return None

    x, y = number(_EAST_KEY), number(_NORTH_KEY)
    if x is None or y is None:
        return False
    return not _apart(fact, Fact("location", "", ("xy", x, y)))


def _printed_fields(test: Any) -> List[Tuple[str, str]]:
    """``(name, value)`` for every field a sheet prints beside its results,
    and every column a summary row carries that has no field of its own."""
    out = [(str(k), str(v)) for k, v in
           (getattr(test, "fields", None) or {}).items()]
    out += [(str(k), str(v)) for k, v in (getattr(test, "other", None) or ())]
    return out


def _hole_pages(hole: Sequence[Any]) -> List[int]:
    return sorted({int(p) for inv in hole for p in (inv.pages or ())})


def tell_apart(record: Any, *, doc: Any = None, outline: Any = None
               ) -> Tuple[Twins, List[Any]]:
    """Gather the logs of ``record`` printed under one name into holes,
    qualify each hole where two or more remain, and say so: ``(the twins,
    the QA entries)``.

    The logs of a name are first gathered into holes (:func:`hole_parts`).
    Where they make one hole, no log is qualified, and one QA entry, ranked
    ``open``, says they may be one hole's sheets the fold did not join.
    Where they make two or more, every log of each hole gets ``qualifier``
    set to the first fact of :data:`FACT_ORDER` that tells every hole of the
    name from the others (as the pages print it), or to the pages the hole
    is printed on where none does, and one QA entry, ranked ``open``, says
    so. A log whose name is its own keeps an empty qualifier. ``doc`` and
    ``outline`` give the report sections and the titles; without them those
    two are unknown. Run again on the same record, it gives the same
    qualifiers.
    """
    from report_ingest.model import QAEntry
    from report_ingest.reconciler import fold_id

    page_print = PagePrint(doc, outline) if (doc is not None
                                             or outline is not None) else None
    twins = Twins(page_print, getattr(record, "bound_documents", None) or ())
    qa: List[Any] = []
    for inv in getattr(record, "investigations", None) or ():
        if getattr(inv, "qualifier", ""):
            inv.qualifier = ""
    for group in name_groups(getattr(record, "investigations", None) or ()):
        fold = fold_id(group[0].investigation_id)
        twins.groups[fold] = group
        holes, later, unsure = hole_parts(group, page_print, record)
        twins.holes[fold] = holes
        for hole in holes:
            for inv in hole:
                twins._hole[id(inv)] = hole
        name = group[0].investigation_id
        facts = [hole_facts(hole[0], record, page_print) for hole in holes]
        for hole, found in zip(holes, facts):
            twins.facts[id(hole[0])] = found
        if len(holes) == 1:
            qa.append(_one_hole_entry(name, holes[0], later, unsure, group,
                                      record, page_print))
            continue
        telling = [(what, _telling(facts, what)) for what in FACT_ORDER]
        telling = [(what, got) for what, got in telling if got is not None]
        spans = [pages_span(_hole_pages(hole)) for hole in holes]
        if telling:
            what, chosen = telling[0]
            qualifiers = [fact.shown for fact in chosen]
        else:
            what, chosen = "", []
            qualifiers = (spans if len(set(spans)) == len(spans) and all(
                _hole_pages(hole) for hole in holes)
                else [f"the {_ordinal(n + 1)} of {len(holes)} printed"
                      for n in range(len(holes))])
        if len(set(qualifiers)) < len(qualifiers):
            # Two values cut to one shown text: the pages say which is which.
            qualifiers = [f"{q}, {span}" for q, span in zip(qualifiers, spans)]
        for hole, qualifier in zip(holes, qualifiers):
            for inv in hole:
                inv.qualifier = qualifier
        places = "; ".join(spans)
        count = _count(len(holes))
        if what:
            also = [w for w, _got in telling[1:]]
            detail = (f"{count} holes are printed under the name {name!r} "
                      f"({places}); they are holes under one printed name, "
                      f"kept as {count} holes, told apart by their {what}, "
                      f"which is kept beside the printed name")
            if also:
                detail += f"; they differ in their {_listed(also)} too"
        else:
            detail = (f"{count} holes are printed under the name {name!r} "
                      f"({places}); they are holes under one printed name, "
                      f"and nothing their pages print tells them apart (no "
                      f"report section, title, project, firm, kind, date, "
                      f"place or ground elevation differs, and their depths "
                      f"overlap or say nothing): they are kept as {count} "
                      f"holes, each named by the pages it is printed on, and "
                      f"one may be the other printed again")
        cut = [(hole, qualifier) for hole, qualifier in zip(holes, qualifiers)
               if len(hole) > 1]
        for hole, qualifier in cut:
            detail += (f"; {name} ({qualifier}) is printed on "
                       f"{_count(len(hole))} logs taken as one hole's "
                       f"sheets the fold did not join ("
                       + "; ".join(f"{pages_span(inv.pages)}: "
                                   f"{later[id(inv)]}"
                                   for inv in hole[1:]) + ")")
        qa.append(QAEntry(
            kind="note", where=f"investigations.{name}", rank="open",
            detail=detail,
            values=[f"{name} ({qualifier}): {span}"
                    for qualifier, span in zip(qualifiers, spans)],
            pages=sorted({int(p) for inv in group for p in inv.pages})))
    return twins, qa


def _one_hole_entry(name: str, hole: Sequence[Any], later: Dict[int, str],
                    unsure: set, group: Sequence[Any], record: Any,
                    page_print: Optional[PagePrint]) -> Any:
    """The QA entry for the logs of a name taken as one hole's sheets:
    open, as a person should confirm they are one hole; what each later
    log prints that says so, or that nothing does; and what their pages
    print differently, where anything does."""
    from report_ingest.model import QAEntry

    count = _count(len(hole))
    each = [hole_facts(inv, record, page_print) for inv in hole]
    differ = [what for what in FACT_ORDER
              if _telling(each, what) is not None]
    places = "; ".join(pages_span(inv.pages) for inv in hole)
    if any(id(inv) in unsure for inv in hole):
        head = (f"{name!r} is printed on {count} logs ({places}) that may be "
                f"one hole's sheets the fold did not join, or two holes: "
                f"neither their depths nor anything else their pages print "
                f"tells them apart. ")
        tail = (f" They are kept as the {count} logs they were read as and "
                f"counted as one hole, as before; a laboratory result naming "
                f"{name!r} is linked to the log holding its sample (by the "
                f"sample's name or its depth), else to the first. A person "
                f"should say whether they are one hole or more")
    else:
        head = (f"{name!r} is printed on {count} logs ({places}) that read as "
                f"one hole's sheets the fold did not join. ")
        tail = (f" They are kept as the {count} logs they were read as, "
                f"counted as one hole; a laboratory result naming {name!r} is "
                f"linked to the log holding its sample (by the sample's name "
                f"or its depth), else to the first. A person should confirm "
                f"they are one hole")
    detail = (head + " ".join(f"On {pages_span(inv.pages)}, "
                              f"{later[id(inv)]}." for inv in hole[1:])
              + tail)
    if differ:
        detail += (f"; their pages print a different {_listed(differ)}, so "
                   f"they may yet be two holes")
    return QAEntry(
        kind="note", where=f"investigations.{name}", rank="open",
        detail=detail,
        values=[f"{name}: {pages_span(inv.pages)}" + (
            f" ({later[id(inv)]})" if id(inv) in later else "")
            for inv in hole],
        pages=sorted({int(p) for inv in group for p in inv.pages}))


def _count(n: int) -> str:
    return {2: "two", 3: "three", 4: "four"}.get(n, str(n))


def _ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth"}.get(
        n, f"{n}th")


def _listed(words: Sequence[str]) -> str:
    words = list(words)
    return words[0] if len(words) == 1 else \
        ", ".join(words[:-1]) + " and " + words[-1]
