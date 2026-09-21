"""Recognising the FORM a boring log was printed on, before any model runs.

THE OWNER'S OBSERVATION, 2026-09-20: *"take a look at [two firms'] logs and
how they're so standard. You can probably build some simple rules that would
almost always catch those kinds of logs (but keep in mind the [second firm's]
template has shifted slightly over the years). Maybe ... a subclassification
tag ... (kinda a flag that says 'hey this looks like a <firm> log') that can
get fed with the other items for a vote."*

That is exactly right, and it is worth having for two separate reasons.

**As a voter.** A page whose footer prints one firm's gINT report name and
whose title block prints that firm's own field labels is a boring log of that
firm, and no picture of it and no page-role rule is as sure of that as the
printed form is. The match goes beside the rules' label and the vision
label on the page's line, with a confidence, and the vote stage carries it.

**As a key to the grid.** ``log_grid`` names a column by classifying its
printed header against a general vocabulary, and on a form that prints
``DATA`` over three stacked values -- the sample, the blow record and the
recovery, one under another -- there is nothing in the word ``DATA`` to
classify. A fingerprint that says *on this form the column headed DATA
carries the sample id, the blows and the recovery* puts those names on that
column, and the grid's floor then seeds values it would otherwise leave on
the page.

WHAT IS SHIPPED AND WHAT IS NOT. The machinery here is generic and knows no
firm. The FINGERPRINTS are data in a JSON file that travels with the private
truth folder and is never committed:

* ``<truth_dir>/../templates.json`` -- found automatically beside the truth
  folder the harness and the cluster stage already carry, or
* an explicit ``templates_path=`` on any of the entry points.

:data:`EXAMPLE_PATH` ships beside this module with two INVENTED firms, to
document the shape. With no templates file anywhere, every function here is a
no-op and nothing downstream changes.

THE COST IS MILLISECONDS. One page's text, folded once, compared to a handful
of short phrases with rapidfuzz. No model call, no render, no network.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Fingerprint", "TemplateMatch", "EXAMPLE_PATH", "MATCH_THRESHOLD",
    "MAX_EVIDENCE", "PHRASE_RATIO", "COLUMN_RATIO", "GROUP_WEIGHTS",
    "SLOT_NAMES",
    "load_templates", "templates_beside", "use_templates", "active_templates",
    "resolve_templates",
    "recognise", "recognise_pages", "column_names", "ledger_note",
    "annotate_ledger", "page_text",
]

#: The example file, shipped with the package: two INVENTED firms documenting
#: the shape a real fingerprint takes. It is never loaded automatically.
EXAMPLE_PATH = Path(__file__).with_name("templates.json.EXAMPLE")

#: A phrase counts as printed on the page at this partial ratio or better.
#: High, because the phrases are the form's own words and a form prints them
#: the same way every time; the slack is for a hyphen, a stray space and the
#: letter an optical pass got wrong.
PHRASE_RATIO = 88.0
#: A column header matches a ``column_map`` entry at this ratio or better.
#: Lower than :data:`PHRASE_RATIO` because ``log_grid`` joins the header band
#: into one string and may carry a stray neighbour word into it.
COLUMN_RATIO = 80.0
#: What each group of phrases is worth. The FOOTER is the strongest evidence
#: a form leaves -- a gINT report name or a data-template file name is
#: printed by the template itself and by nothing else -- the title block next,
#: and the column headers last, since two firms' forms share many of them.
GROUP_WEIGHTS: Dict[str, float] = {"footer": 0.5, "title": 0.3,
                                   "columns": 0.2}
#: How many matched phrases a match carries with it. Enough that a per-group
#: tally off the list is the true one for any fingerprint a person would
#: write, and bounded so a match stays a small object.
MAX_EVIDENCE = 24

#: Below this the page is not claimed for any template. Measured on the
#: corpus, 2026-09-20: every log a fingerprint describes scores 0.97 or
#: better, because the footer stamp is there and the footer is half the
#: score; the one page off the logs that any fingerprint reached scored 0.59
#: (a laboratory sheet from the same gINT project, carrying the title block's
#: words and none of the footer). The line is drawn between the two, nearer
#: the false one, because a page whose footer did not survive into text is
#: not a page to claim on the title block alone.
MATCH_THRESHOLD = 0.65

#: What a ``column_map`` key means to the grid. The owner writes the slot in
#: the record's words; the grid names columns in its own. Unknown keys are
#: passed through unchanged, so a fingerprint may name a grid column class
#: directly.
SLOT_NAMES: Dict[str, Tuple[str, ...]] = {
    "blows": ("blows",),
    "n": ("n_value",),
    "n_value": ("n_value",),
    "recovery": ("recovery",),
    "rec": ("recovery",),
    "rqd": ("rqd",),
    "sample": ("sample_id",),
    "sample_id": ("sample_id",),
    "sample_number": ("sample_id",),
    "sample_type": ("sample_type",),
    "type": ("sample_type",),
    "index": ("water_content", "liquid_limit", "plastic_limit",
              "plasticity_index", "fines"),
    "tests": ("tests",),
    "water_content": ("water_content",),
    "moisture": ("water_content",),
    "dry_unit_weight": ("dry_unit_weight",),
    "liquid_limit": ("liquid_limit",),
    "plastic_limit": ("plastic_limit",),
    "plasticity_index": ("plasticity_index",),
    "fines": ("fines",),
    "passing_200": ("fines",),
    "qu": ("qu",),
    "pocket_pen": ("pocket_pen",),
    "uscs": ("uscs",),
    "graphic": ("graphic",),
    "description": ("description",),
    "depth": ("depth",),
    "elevation": ("elevation",),
    "remarks": ("remarks",),
}


# ---------------------------------------------------------------------------
# the fingerprint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fingerprint:
    """One printed FORM: what it says about itself, and what its columns are.

    ``family`` is the thing a voter wants -- who printed this log -- and
    several fingerprints share one, because a firm's form drifts over the
    years: a new gINT library, a coordinate line the old one did not have, a
    column that changed its heading. ``years`` is free text for a person
    reading the file.
    """

    name: str
    family: str
    years: str = ""
    title_phrases: Tuple[str, ...] = ()
    column_headers: Tuple[str, ...] = ()
    footer_phrases: Tuple[str, ...] = ()
    layout: Dict[str, Any] = field(default_factory=dict)
    column_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, blob: Dict[str, Any]) -> "Fingerprint":
        name = str(blob.get("name") or "").strip()
        family = str(blob.get("family") or "").strip()
        if not name or not family:
            raise ValueError("a fingerprint needs a 'name' and a 'family'")

        def phrases(key: str) -> Tuple[str, ...]:
            out: List[str] = []
            for item in (blob.get(key) or ()):
                text = " ".join(str(item).split())
                if text:
                    out.append(text)
            return tuple(out)

        column_map = {str(k): " ".join(str(v).split())
                      for k, v in (blob.get("column_map") or {}).items()
                      if str(v or "").strip()}
        return cls(name=name, family=family,
                   years=str(blob.get("years") or "").strip(),
                   title_phrases=phrases("title_phrases"),
                   column_headers=phrases("column_headers"),
                   footer_phrases=phrases("footer_phrases"),
                   layout=dict(blob.get("layout") or {}),
                   column_map=column_map)

    def groups(self) -> Dict[str, Tuple[str, ...]]:
        """The phrase groups this fingerprint actually declares."""
        out = {"footer": self.footer_phrases, "title": self.title_phrases,
               "columns": self.column_headers}
        return {name: phrases for name, phrases in out.items() if phrases}


@dataclass(frozen=True)
class TemplateMatch:
    """One page, recognised as one form.

    ``confidence`` is the weighted fraction of the fingerprint's own phrases
    the page prints, 0 to 1. ``margin`` is how far that is above the best
    fingerprint of a DIFFERENT family, which is the number that says whether
    the FAMILY -- the thing a voter is told -- was in doubt; a second
    fingerprint of the same family scoring nearly as well is the template
    drifting, not a disagreement.
    """

    name: str
    family: str
    confidence: float
    evidence: Tuple[str, ...] = ()
    column_map: Dict[str, str] = field(default_factory=dict)
    margin: float = 1.0
    runner_up: str = ""
    page: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "family": self.family,
                "confidence": round(float(self.confidence), 3),
                "margin": round(float(self.margin), 3),
                "runner_up": self.runner_up,
                "evidence": list(self.evidence),
                "page": self.page}

    def __str__(self) -> str:
        return f"template {self.family} ({self.confidence:.2f})"


# ---------------------------------------------------------------------------
# loading the private file
# ---------------------------------------------------------------------------

def load_templates(path: Any) -> List[Fingerprint]:
    """Every fingerprint in one JSON file.

    The file is either a list of fingerprints or ``{"templates": [...]}``, so
    a person can keep notes beside them. A missing file is an empty list, not
    an error: the whole feature is optional.
    """
    if path is None:
        return []
    here = Path(path)
    if not here.is_file():
        return []
    blob = json.loads(here.read_text(encoding="utf-8"))
    rows = blob.get("templates") if isinstance(blob, dict) else blob
    return [Fingerprint.from_dict(row) for row in (rows or [])]


def templates_beside(truth_dir: Any) -> List[Fingerprint]:
    """The fingerprints that travel with a truth folder.

    ``<truth_dir>/../templates.json`` -- so the file sits in the private
    truth folder beside ``logs/`` and ``narrative/`` and is carried wherever
    the truth folder is carried, including onto the cluster.
    """
    if truth_dir is None:
        return []
    return load_templates(Path(truth_dir).parent / "templates.json")


#: The fingerprints this process is working with. Empty until something sets
#: them, which is what makes every entry point a no-op by default.
_ACTIVE: List[Fingerprint] = []


def use_templates(source: Any) -> List[Fingerprint]:
    """Set the fingerprints for this process and return them.

    ``source`` is a path to a JSON file, an iterable of
    :class:`Fingerprint`, or ``None`` to clear. Called once by whatever knows
    where the private file is -- the harness, the cluster stage, an
    application that ships its own -- and everything else just asks.
    """
    global _ACTIVE
    if source is None:
        _ACTIVE = []
    elif isinstance(source, (str, Path)):
        _ACTIVE = load_templates(source)
    else:
        _ACTIVE = [item if isinstance(item, Fingerprint)
                   else Fingerprint.from_dict(item) for item in source]
    return list(_ACTIVE)


def active_templates() -> List[Fingerprint]:
    """The fingerprints in force, which is usually none."""
    return list(_ACTIVE)


def resolve_templates(templates: Any = None,
                      templates_path: Any = None) -> List[Fingerprint]:
    """The fingerprints a caller means, whichever way it said it.

    A path, an iterable of :class:`Fingerprint` or of plain dicts, or
    ``None`` for whatever :func:`use_templates` put in force -- which is
    nothing unless something set it. The public spelling of what every entry
    point here does with its own two arguments, for a caller (the ingest
    graph) that has one of them and wants the same answer.
    """
    return _resolve(templates, templates_path)


def _resolve(templates: Any, templates_path: Any) -> List[Fingerprint]:
    if templates_path is not None:
        return load_templates(templates_path)
    if templates is None:
        return active_templates()
    if isinstance(templates, (str, Path)):
        return load_templates(templates)
    return [item if isinstance(item, Fingerprint)
            else Fingerprint.from_dict(item) for item in templates]


# ---------------------------------------------------------------------------
# matching text
# ---------------------------------------------------------------------------

_RE_SPACE = re.compile(r"\s+")


def _fold(text: Any) -> str:
    """One string as it is compared: lower case, one space between words."""
    return _RE_SPACE.sub(" ", str(text or "")).strip().lower()


def _ratio(needle: str, hay: str) -> float:
    """rapidfuzz's partial ratio, or containment without it.

    rapidfuzz arrives with planlens, so it is normally here. The fallback is
    deliberately blunt rather than clever: a missing package must show up as a
    different NUMBER, never as a quietly different definition of "printed on
    this page".
    """
    if not needle or not hay:
        return 0.0
    try:
        from rapidfuzz import fuzz
    except ImportError:                      # pragma: no cover - fallback
        return 100.0 if needle in hay else 0.0
    return float(fuzz.partial_ratio(needle, hay))


def page_text(doc: Any, page: int) -> str:
    """Everything one page prints, folded, or ``""`` when it will not read."""
    try:
        content = doc.page(int(page))
    except Exception:                        # a page that will not open is
        return ""                            # simply not recognised
    return _fold(" ".join(str(line.text or "") for line in content.lines))


def _grid_headers(grid: Any, page: Optional[int] = None) -> List[str]:
    """The headers ``log_grid`` read, folded, for the columns group."""
    out: List[str] = []
    for column in getattr(grid, "columns", ()) or ():
        if page is not None and getattr(column, "page", page) != page:
            continue
        header = _fold(getattr(column, "header", ""))
        if header:
            out.append(header)
    return out


def _score(fingerprint: Fingerprint, text: str, headers: Sequence[str]
           ) -> Tuple[float, List[str]]:
    """One fingerprint against one page: the confidence and the evidence."""
    weight_used = 0.0
    earned = 0.0
    evidence: List[str] = []
    for group, phrases in fingerprint.groups().items():
        weight = GROUP_WEIGHTS.get(group, 0.2)
        weight_used += weight
        hits = 0
        for phrase in phrases:
            folded = _fold(phrase)
            best = _ratio(folded, text)
            if group == "columns" and headers:
                # A column header the grid has already isolated is a cleaner
                # comparison than the whole page's text, where the header's
                # words sit among everything else the form prints.
                best = max(best, max(_ratio(folded, h) for h in headers))
            if best >= PHRASE_RATIO:
                hits += 1
                if len(evidence) < MAX_EVIDENCE:
                    evidence.append(f'{group} "{phrase}" ({best:.0f})')
        earned += weight * (hits / len(phrases))
    if not weight_used:
        return 0.0, []
    return earned / weight_used, evidence


def recognise(doc: Any, page: int, grid: Any = None, *,
              templates: Any = None, templates_path: Any = None,
              threshold: float = MATCH_THRESHOLD) -> Optional[TemplateMatch]:
    """Which printed form is this page, if any?

    ``grid`` is ``log_grid``'s reading of the page when the caller already has
    it; it only sharpens the column-header group and is never required.
    ``None`` comes back for a page below ``threshold``, for a page with no
    text and whenever there are no fingerprints -- which is the default state
    and the reason nothing changes without the private file.
    """
    rows = _resolve(templates, templates_path)
    if not rows:
        return None
    text = page_text(doc, page)
    if not text:
        return None
    headers = _grid_headers(grid, int(page)) if grid is not None else []

    scored: List[Tuple[float, List[str], Fingerprint]] = []
    for fingerprint in rows:
        confidence, evidence = _score(fingerprint, text, headers)
        scored.append((confidence, evidence, fingerprint))
    scored.sort(key=lambda row: row[0], reverse=True)
    confidence, evidence, best = scored[0]
    if confidence < threshold:
        return None
    other = next((row for row in scored[1:] if row[2].family != best.family),
                 None)
    return TemplateMatch(
        name=best.name, family=best.family, confidence=round(confidence, 4),
        evidence=tuple(evidence), column_map=dict(best.column_map),
        margin=round(confidence - (other[0] if other else 0.0), 4),
        runner_up=other[2].name if other else "", page=int(page))


def recognise_pages(doc: Any, pages: Iterable[int], *, templates: Any = None,
                    templates_path: Any = None,
                    threshold: float = MATCH_THRESHOLD
                    ) -> Dict[int, TemplateMatch]:
    """:func:`recognise` over several pages; pages that matched only."""
    rows = _resolve(templates, templates_path)
    if not rows:
        return {}
    out: Dict[int, TemplateMatch] = {}
    for page in pages:
        match = recognise(doc, int(page), templates=rows, threshold=threshold)
        if match is not None:
            out[int(page)] = match
    return out


# ---------------------------------------------------------------------------
# what a match is worth to the grid
# ---------------------------------------------------------------------------

def _names_for(slot: str) -> Tuple[str, ...]:
    return SLOT_NAMES.get(str(slot).strip().lower(), (str(slot).strip(),))


def column_names(grid: Any, match: Optional[TemplateMatch]
                 ) -> Dict[str, Tuple[str, ...]]:
    """``column_id -> the names this form says that column carries``.

    Each ``column_map`` entry names a printed header; the grid's column whose
    own header matches it best, at :data:`COLUMN_RATIO` or better, gets the
    slot's names. The grid's OWN names are kept beside them: a generic name
    the grid got right is not worth losing to a fingerprint, and the floor
    treats the names as a set. Empty when nothing matched, which leaves the
    floor exactly as it was.
    """
    if match is None or not match.column_map:
        return {}
    columns = list(getattr(grid, "columns", ()) or ())
    if not columns:
        return {}
    out: Dict[str, List[str]] = {}
    for slot, header_phrase in match.column_map.items():
        folded = _fold(header_phrase)
        if not folded:
            continue
        best_id = ""
        best_score = COLUMN_RATIO
        for column in columns:
            header = _fold(getattr(column, "header", ""))
            if not header:
                continue
            score = _ratio(folded, header)
            if score >= best_score:
                best_score, best_id = score, column.id
        if not best_id:
            continue
        here = out.setdefault(best_id, [])
        for name in _names_for(slot):
            if name and name not in here:
                here.append(name)
    if not out:
        return {}
    by_id = {column.id: column for column in columns}
    merged: Dict[str, Tuple[str, ...]] = {}
    for column_id, names in out.items():
        own = tuple(getattr(by_id.get(column_id), "names", ()) or ())
        merged[column_id] = tuple(names) + tuple(n for n in own
                                                 if n not in names)
    return merged


def ledger_note(match: Optional[TemplateMatch]) -> str:
    """The line a ledger or a vote row carries for one page, or ``""``.

    ``template <family> (0.92)`` -- the family, because that is the claim
    ("this looks like a <firm> log"), and the confidence, because a voter
    that cannot be argued with is not a voter.
    """
    if match is None:
        return ""
    return f"template {match.family} ({match.confidence:.2f})"


def annotate_ledger(lines: Sequence[str],
                    matches: Dict[int, TemplateMatch]) -> List[str]:
    """planlens' page ledger with a template note on the pages that matched.

    A ledger line begins ``pNNN ``; the note is appended to the line for that
    page and every other line is returned untouched. Given no matches the
    lines come back as they went in.
    """
    if not matches:
        return list(lines)
    out: List[str] = []
    for line in lines:
        text = str(line)
        head = text.split(" ", 1)[0]
        page: Optional[int] = None
        if head.startswith("p") and head[1:].isdigit():
            page = int(head[1:])
        match = matches.get(page) if page is not None else None
        out.append(f"{text} {ledger_note(match)}" if match else text)
    return out
