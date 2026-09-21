"""A report bound inside another report, and where its pages begin and end.

WHY THIS EXISTS. A geotechnical report is very often not one report. An
earlier firm's whole investigation is reproduced as an appendix; a bridging
report is bound into the design-build report that answers it; a data report
sits inside a design report. The corpus run of 2026-09-20 (ledger run 7) has
triage calling 19 of 38 reports ``multi_document``, with one to four bound
documents in each, and the hand narrative notes for two of them say the same
thing in the owner's own words: the borings and test pits in that appendix
belong to the EARLIER investigation, not to the report they are bound into.

Until this module the pipeline recorded those pages and did not read them --
one ``QAEntry(kind="skipped")`` per bound document saying the pages exist.
The alternative that was never on the table is reading them into the SAME
record: an earlier firm's B-1 and this report's B-1 in one list of
investigations, a 2009 water level beside a 2026 one, a count of borings that
is the sum of two investigations and the truth about neither.

SO A BOUND DOCUMENT BECOMES ITS OWN RECORD. The parent keeps those pages
labelled ``appended_report`` and LISTS the child; the child is built from the
same pages by the same pipeline -- its own identity, its own labels, its own
work items, its own readers, its own reconcile, its own exports -- and
nothing is attributed across the boundary. A boring in the child is the
child's. The parent's narrative may cite it as a previous investigation, and
the parent's narrative reader is told the page ranges so it can.

WHERE THE PAGE RANGES COME FROM. Two sources, and they are not the same kind
of evidence:

* **triage** (:mod:`report_ingest.triage`) answers ``bound_together`` -- a
  model that has read the whole ledger, the outline and the front matter
  saying "pages 112-184 are their own document". It sees the CONTENTS LIST,
  so it catches a bound report the pages themselves are shy about;
* **planlens' rules** label a page ``appended_report`` when it sits inside a
  document the section builder found nested in this one. Structural, cheap,
  and the label class the rules demonstrably own -- 494 in-sample pages a
  vision pass never once emits.

They usually agree. Where they do not, this takes the UNION and says so in a
QA entry, because a page wrongly included in the child is recoverable (it is
in the child's record, on the child's own page numbers) and a page wrongly
left in the parent is an earlier investigation's boring attributed to this
report, which is the error the whole exercise is about.

AND A SHORT RUN IS NOT A DOCUMENT. Under :data:`MIN_BOUND_PAGES` pages the
run stays in the parent: a single reproduced log sheet, a two-page letter
from a previous consultant and a one-page figure lifted from an old report
are appended PAGES, and a record, a summary, a library page and a DIGGS file
for each of them would be four files saying nothing.

PAGE NUMBERS ARE THE PARENT FILE'S THROUGHOUT. There is one PDF, and a
reviewer sent to check a value opens it at the page the record names. So a
child record's provenance, item pages and page labels are all 0-based indexes
into the SAME file, and the child's own extent is recorded once, on
``ReportRecord.parent``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from report_ingest.model import DOCUMENT_TYPE_VALUES, QAEntry

__all__ = [
    "MIN_BOUND_PAGES", "BOUND_KINDS", "APPENDED_LABEL", "BoundRange",
    "bound_ranges", "label_runs", "triage_runs", "parse_range",
    "compact_pages", "BoundIdentity", "BoundIdentityResult",
    "identify_bound", "identity_prompt", "narrative_block",
    "BOUND_IDENTITY_SYSTEM",
]

#: A run of pages shorter than this is not a bound DOCUMENT -- it is appended
#: pages, and it stays in the parent. Four is the smallest thing that is
#: recognisably a report reproduced whole: a cover, something said, a tab and
#: a sheet behind it. Three pages of somebody else's letterhead is a letter.
MIN_BOUND_PAGES = 4

#: What a separately-bound part of a file can be. Declared HERE and imported
#: by :mod:`report_ingest.triage`, which answers with it: the module that
#: makes a bound document into a record owns the vocabulary for what one is,
#: and two copies of an enumeration are two enumerations.
BOUND_KINDS: Tuple[str, ...] = (
    "volume", "appended_prior_report", "data_report", "other",
)

#: The page label planlens gives every page of a document nested in this one.
APPENDED_LABEL = "appended_report"

#: How many pages of the child are shown to the identity call. Its cover, its
#: letter and its contents are at the front of it, which is the whole of what
#: the call is for.
IDENTITY_PAGES = 4
#: And how much of their text.
IDENTITY_CHARS = 9000


# ---------------------------------------------------------------------------
# page ranges
# ---------------------------------------------------------------------------

def parse_range(spec: Any, n_pages: int) -> List[int]:
    """``"112-184"`` or ``"0-93,210-240"`` as a sorted list of page indexes.

    CLIPPED, never raised. This reads a MODEL's answer: a range running past
    the end of the document is a model that counted wrong about a file it was
    shown the ledger of, and the recoverable part of its answer is worth more
    than an exception. Anything unparseable is an empty list, and the caller
    turns that into a QA entry rather than a failure.
    """
    if spec is None:
        return []
    if isinstance(spec, int):
        pages = [spec]
    elif isinstance(spec, str):
        pages = []
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            match = re.fullmatch(r"(\d+)(?:[-–](\d+))?", part)
            if match is None:
                continue
            first = int(match.group(1))
            last = int(match.group(2)) if match.group(2) else first
            if last < first:
                first, last = last, first
            pages.extend(range(first, last + 1))
    else:
        try:
            pages = [int(p) for p in spec]
        except (TypeError, ValueError):
            return []
    kept = sorted({p for p in pages if 0 <= p < int(n_pages)})
    return kept


def compact_pages(pages: Sequence[int]) -> str:
    """``[15,16,17,18]`` as ``"15-18"``; the spelling the record stores."""
    rows = sorted({int(p) for p in pages})
    if not rows:
        return ""
    out: List[str] = []
    start = previous = rows[0]
    for page in rows[1:]:
        if page == previous + 1:
            previous = page
            continue
        out.append(f"{start}-{previous}" if previous > start else f"{start}")
        start = previous = page
    out.append(f"{start}-{previous}" if previous > start else f"{start}")
    return ",".join(out)


@dataclass
class BoundRange:
    """One report bound inside another, before anything of it is read.

    ``pages`` are 0-based indexes into the PARENT file, contiguous and in
    order. ``said_by`` names which of the two sources claimed this range --
    ``("triage",)``, ``("planlens",)`` or both -- so a reviewer of the record
    can tell a range both agreed on from one only the model saw.
    """

    bound_id: str
    pages: List[int]
    kind: str = "appended_prior_report"
    title: str = ""
    said_by: Tuple[str, ...] = ()
    note: str = ""

    @property
    def first_page(self) -> int:
        return int(self.pages[0])

    @property
    def last_page(self) -> int:
        return int(self.pages[-1])

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def spec(self) -> str:
        """The page range as the record spells it."""
        return compact_pages(self.pages)

    def to_dict(self) -> Dict[str, Any]:
        return {"bound_id": self.bound_id, "pages": self.spec,
                "first_page": self.first_page, "last_page": self.last_page,
                "n_pages": self.n_pages, "kind": self.kind,
                "title": self.title, "said_by": list(self.said_by),
                "note": self.note}


def _runs(pages: Sequence[int]) -> List[List[int]]:
    """Contiguous runs out of a page list, in order."""
    out: List[List[int]] = []
    for page in sorted({int(p) for p in pages}):
        if out and page == out[-1][-1] + 1:
            out[-1].append(page)
        else:
            out.append([page])
    return out


def label_runs(labels: Dict[int, str]) -> List[List[int]]:
    """The ``appended_report`` runs in the record's own page labels.

    The labels rather than the rules' raw roles on purpose: what the record
    SAYS a page is has been through the vote and the review, and a bound
    document the review moved out of ``appended_report`` is no longer one as
    far as everything else downstream is concerned.
    """
    return _runs([int(page) for page, label in (labels or {}).items()
                  if label == APPENDED_LABEL])


def triage_runs(profile: Any, n_pages: int) -> List[Dict[str, Any]]:
    """Triage's ``bound_together``, parsed into runs of parent pages.

    One entry may name a split range (``"0-93,210-240"`` is a volume whose
    two halves are not adjacent); each contiguous run of it becomes its own
    row, carrying the entry's kind and title, because each is separately a
    thing with a beginning and an end.
    """
    out: List[Dict[str, Any]] = []
    for entry in (getattr(profile, "bound_together", None) or ()):
        if not isinstance(entry, dict):
            entry = {"kind": getattr(entry, "kind", ""),
                     "pages": getattr(entry, "pages", ""),
                     "title": getattr(entry, "title", "")}
        pages = parse_range(entry.get("pages"), n_pages)
        for run in _runs(pages):
            out.append({"pages": run,
                        "kind": str(entry.get("kind") or "other"),
                        "title": str(entry.get("title") or "")})
    return out


def bound_ranges(profile: Any, labels: Dict[int, str], n_pages: int,
                 *, min_pages: Optional[int] = None
                 ) -> Tuple[List[BoundRange], List[QAEntry]]:
    """The reports bound inside this one, and what a reviewer is told.

    The union of what triage said and what the page labels say, run by run:
    two runs that touch or overlap are ONE bound document and their union is
    its extent. A run shorter than ``min_pages`` is dropped with a note --
    appended pages, not a document -- and a range the two sources disagreed
    about carries a QA entry naming both, because the union silently taking
    the larger answer is the kind of decision a record should print.

    ``min_pages`` defaults to :data:`MIN_BOUND_PAGES`, read HERE rather than
    bound as a default argument, so that setting the module's constant
    actually changes the floor.
    """
    min_pages = MIN_BOUND_PAGES if min_pages is None else int(min_pages)
    from_triage = triage_runs(profile, n_pages)
    from_labels = label_runs(labels)

    claims: List[Dict[str, Any]] = [
        {"pages": set(row["pages"]), "kind": row["kind"],
         "title": row["title"], "said_by": {"triage"},
         "triage": list(row["pages"]), "planlens": []}
        for row in from_triage]
    for run in from_labels:
        here = set(run)
        touching = [c for c in claims if _touches(c["pages"], here)]
        if not touching:
            claims.append({"pages": here, "kind": "appended_prior_report",
                           "title": "", "said_by": {"planlens"},
                           "triage": [], "planlens": list(run)})
            continue
        for claim in touching:
            claim["pages"] |= here
            claim["said_by"].add("planlens")
            claim["planlens"] = sorted(set(claim["planlens"]) | here)
    claims = _merge_touching(claims)

    qa: List[QAEntry] = []
    ranges: List[BoundRange] = []
    for claim in sorted(claims, key=lambda c: min(c["pages"])):
        pages = sorted(claim["pages"])
        for run in _runs(pages):
            if len(run) < int(min_pages):
                qa.append(QAEntry(
                    kind="note", where="bound.short_run",
                    detail=(f"{len(run)} page(s) read as part of a report "
                            f"bound into this one, which is under the "
                            f"{min_pages}-page floor for a document; they "
                            f"stay in this record as appended pages"),
                    pages=list(run)))
                continue
            said_by = tuple(sorted(claim["said_by"]))
            note = _disagreement_note(claim, run)
            if note:
                qa.append(QAEntry(
                    kind="note", where="bound.extent",
                    detail=note,
                    values=[f"triage {compact_pages(claim['triage']) or 'none'}",
                            f"planlens "
                            f"{compact_pages(claim['planlens']) or 'none'}"],
                    pages=[run[0], run[-1]]))
            ranges.append(BoundRange(
                bound_id=f"bound{len(ranges) + 1}",
                pages=list(run), kind=claim["kind"] or "other",
                title=str(claim["title"] or ""), said_by=said_by, note=note))
    return ranges, qa


def _touches(a: set, b: set) -> bool:
    """Do two page sets overlap or sit next to each other?"""
    if a & b:
        return True
    return (min(a) - 1 <= max(b) <= max(a) + 1
            or min(b) - 1 <= max(a) <= max(b) + 1)


def _merge_touching(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold claims that touch into one, until none do."""
    out: List[Dict[str, Any]] = []
    for claim in sorted(claims, key=lambda c: min(c["pages"])):
        if out and _touches(out[-1]["pages"], claim["pages"]):
            merged = out[-1]
            merged["pages"] |= claim["pages"]
            merged["said_by"] |= claim["said_by"]
            merged["triage"] = sorted(set(merged["triage"])
                                      | set(claim["triage"]))
            merged["planlens"] = sorted(set(merged["planlens"])
                                        | set(claim["planlens"]))
            merged["title"] = merged["title"] or claim["title"]
            if merged["kind"] in ("", "other"):
                merged["kind"] = claim["kind"]
            continue
        out.append(claim)
    return out


def _disagreement_note(claim: Dict[str, Any], run: Sequence[int]) -> str:
    """What to say when the two sources did not draw the same boundary."""
    triage = set(claim["triage"]) & set(run)
    planlens = set(claim["planlens"]) & set(run)
    if not triage or not planlens or triage == planlens:
        return ""
    return (f"triage and planlens' rules drew this bound document's edges "
            f"differently; the record takes the union, pages "
            f"{compact_pages(run)}, so that no page of it is left attributed "
            f"to the report it is bound into")


# ---------------------------------------------------------------------------
# the child's identity: one small call over its own front matter
# ---------------------------------------------------------------------------

class BoundIdentity(BaseModel):
    """What the bound report says it IS, off its own cover and letter.

    The whole of the triage a child gets. It is not the parent's triage
    shrunk: a bound document does not need a workflow (the page labels decide
    what runs in it) or a scan fraction (it is pages of a file already
    measured). What nothing else can supply is its identity -- a title, a
    firm and a date that are ITS and not the parent's -- and that is what a
    reader of the library needs to tell the two records apart.
    """

    title: str = Field(
        default="",
        description="the title this bound report prints for itself, as "
                    "printed; empty when it prints none")
    firm: str = Field(
        default="",
        description="the firm or agency that wrote it, as printed; empty "
                    "when it does not say")
    date: str = Field(
        default="",
        description="the date it prints for itself, as printed; empty when "
                    "it does not say")
    document_type: str = Field(
        default="other",
        description="one of: " + "; ".join(DOCUMENT_TYPE_VALUES))
    kind: str = Field(
        default="appended_prior_report",
        description="how it is bound in: " + ", ".join(BOUND_KINDS))
    same_site: str = Field(
        default="unclear",
        description="yes, no or unclear - is it about the same site as the "
                    "report it is bound into")


BOUND_IDENTITY_SYSTEM = """\
You are looking at the FIRST FEW PAGES of a report that was bound inside
another report as an appendix or a volume. You are not reading it and you are
not summarising it. You are answering one question: what is this document,
in its own words.

Everything you answer must be PRINTED ON THESE PAGES. The title is the title
on its cover or its letterhead, not a description you write. The firm is the
firm whose name is on it. The date is the date it prints. Where a page does
not say, the answer is the empty string -- never the parent report's title,
firm or date, which you have not been shown and must not guess at.

'same_site' is yes only when these pages name the same site or project as the
report they are bound into; 'unclear' is the honest answer for pages that
name a site without saying whether it is the same one.
"""


def identity_prompt(doc: Any, pages: Sequence[int],
                    max_pages: int = IDENTITY_PAGES,
                    max_chars: int = IDENTITY_CHARS) -> str:
    """The front of a bound document, verbatim, for the identity call."""
    wanted = [int(p) for p in sorted(pages)][:max(1, int(max_pages))]
    out: List[str] = [
        f"A report bound inside another report, at pages "
        f"{compact_pages(pages)} of the file. Its first "
        f"{len(wanted)} page(s), verbatim:"]
    used = 0
    for page in wanted:
        try:
            lines = [line.text for line in doc.page(page, tables=False).lines
                     if (line.text or "").strip()]
        except Exception:                        # a page that will not read
            continue
        block = f"=== page {page} ===\n" + "\n".join(lines)
        if used + len(block) > max_chars:
            out.append(block[:max(0, max_chars - used)])
            out.append("\n[cut at the size limit]")
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)


@dataclass
class BoundIdentityResult:
    """What the identity call answered, and what it cost."""

    identity: BoundIdentity = field(default_factory=BoundIdentity)
    model: str = ""
    cost: Dict[str, Any] = field(default_factory=dict)
    warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"identity": self.identity.model_dump(),
                "model": self.model, "cost": dict(self.cost),
                "warning": self.warning}


def identify_bound(doc: Any, pages: Sequence[int], engine: Any,
                   *, title_hint: str = "") -> BoundIdentityResult:
    """One structured call over a bound document's own front matter.

    Never raises. A call that returns nothing structured, or an engine that
    fails outright, leaves the identity empty with a warning on the result:
    a child record whose title is the hint triage printed is worth more than
    no child record at all.
    """
    from report_ingest.engine import text_block, user

    body = identity_prompt(doc, pages)
    if title_hint:
        body += (f"\n\nThe report it is bound into lists this document as "
                 f"{title_hint!r}. Use the pages, not that, unless the pages "
                 f"say nothing.")
    try:
        reply = engine.complete([user(text_block(body))],
                                system=BOUND_IDENTITY_SYSTEM,
                                output_format=BoundIdentity)
    except Exception as exc:                     # one child, not the report
        return BoundIdentityResult(
            warning=f"the bound document's identity call failed "
                    f"({type(exc).__name__}: {exc})")
    found = reply.parsed
    if found is None:
        return BoundIdentityResult(
            model=reply.model or getattr(engine, "name", ""),
            cost=_cost(reply),
            warning=f"the bound document's identity call returned no "
                    f"structured answer (stop_reason "
                    f"{reply.stop_reason!r})")
    return BoundIdentityResult(identity=found,
                               model=reply.model or getattr(engine, "name", ""),
                               cost=_cost(reply))


def _cost(reply: Any) -> Dict[str, Any]:
    return {"calls": 1,
            "input_tokens": reply.usage.input_tokens,
            "output_tokens": reply.usage.output_tokens,
            "cache_read_tokens": reply.usage.cache_read_tokens,
            "seconds": round(reply.seconds, 1),
            "dollars": round(reply.usage.dollars(reply.model), 4)}


def narrative_block(rows: Sequence[Dict[str, Any]]) -> str:
    """What the PARENT's narrative reader is told about its bound documents.

    Short and factual, and it says what to do with the knowledge rather than
    only stating it: the counts and the identifier lists this reader answers
    are about the report in front of it, and an earlier investigation bound
    in behind a tab is a PREVIOUS investigation, which is its own question.
    """
    kept = [row for row in rows or () if row.get("pages")]
    if not kept:
        return ""
    out = ["REPORTS BOUND INSIDE THIS ONE. These page ranges are other "
           "reports reproduced whole inside this file. They are being read "
           "SEPARATELY, into their own records."]
    for row in kept:
        title = str(row.get("title") or "").strip()
        firm = str(row.get("firm") or "").strip()
        date = str(row.get("date") or "").strip()
        said = " - ".join(x for x in (title, firm, date) if x)
        out.append(f"  pages {row['pages']}: "
                   + (said or "a report bound in, title not printed"))
    out.append(
        "So: do NOT count their borings, test pits or laboratory tests as "
        "this report's, and do not put their identifiers in this report's "
        "boringDictionary or testPitDictionary. Each of them IS a previous "
        "investigation of this site and counts towards "
        "previousInvestigationCount, together with any others the prose "
        "names. If the narrative says this report was written for a client "
        "other than the project owner, that is still outsideProject; a "
        "report bound in as an appendix does not by itself make this one an "
        "outside report.")
    return "\n".join(out)
