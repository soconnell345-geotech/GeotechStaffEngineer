"""What the record keeps for the owner's question list (BACKLOG RI-102).

Each value is taken from something the pipeline has already read; nothing
here asks a model. The graph calls these when it builds a record, and
:mod:`report_ingest.question_list` calls :func:`field_work` on a record
saved before the field, so that row answers the same either way.

* :func:`printed_pages`: the page number printed on each page, off
  planlens' page map.
* :func:`appendices`: the list of appendices the document prints for
  itself, off planlens' outline.
* :func:`caption_labels`: the caption labels the narrative's caption count
  saw, which until RI-102 were counted and dropped.
* :func:`field_work`: the years and the firms the report's own logs print
  (a firm, company or consultant header field; not the driller).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = ["printed_pages", "appendices", "caption_labels", "field_work",
           "OWN_LOG_KINDS"]

#: The bound reports whose pages hold THIS investigation's logs: a volume of
#: the report itself. A log on a prior report's, a data report's or another
#: document's pages is not this report's own (as the question list places
#: them, :data:`report_ingest.question_list._LOG_SCOPE`).
OWN_LOG_KINDS = ("volume",)

#: A four-digit year in a printed date. Two-digit years are not read: "3/8/47"
#: does not say its century.
_RE_YEAR = re.compile(r"(?<!\d)(19\d\d|20\d\d)(?!\d)")

#: A log header field named for a firm (the field's name, lower case and
#: letters only). Narrower than the one that tells holes apart
#: (:mod:`report_ingest.shared_names`): "engineer" and "drilled by" name a
#: person as often as a firm.
_RE_FIRM_KEY = re.compile(r"firm|company|consultant")
#: A header field that names who drilled or dug, or a person: not the firm
#: that did the investigation ("Drilling Company", "Contractor Foreman").
_RE_NOT_FIRM_KEY = re.compile(r"drill|excavat|foreman|operator|contractor")
#: A header value that says nothing.
_NO_VALUE = {"na", "n/a", "none", "-", "--", "unknown", "nil"}


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def printed_pages(summaries: Iterable[Any],
                  pages: Optional[Iterable[int]] = None) -> Dict[int, str]:
    """``{0-based page: the number printed on it}`` off planlens' page map
    (each summary's ``printed_page``); ``pages`` keeps only those pages (a
    bound report's own). A page with no printed number is left out."""
    keep = None if pages is None else {int(p) for p in pages}
    out: Dict[int, str] = {}
    for summary in summaries:
        page = int(summary.page)
        number = getattr(summary, "printed_page", None)
        if number in (None, "") or (keep is not None and page not in keep):
            continue
        out[page] = str(number)
    return out


def appendices(outline: Any) -> List[Any]:
    """The outline's list of appendices as :class:`AppendixEntry` rows, in
    printed order; empty for no outline or no list."""
    from report_ingest.model import AppendixEntry

    if outline is None or not hasattr(outline, "of_kind"):
        return []
    return [AppendixEntry(number=_flat(e.number), title=_flat(e.title),
                          printed_page=_flat(e.printed_page),
                          page=None if e.page is None else int(e.page))
            for e in outline.of_kind("appendix")]


#: count_captions' keys, and the names the record keeps them under.
_CAPTION_KEYS = (("table_labels", "tables"), ("figure_labels", "figures"),
                 ("appendix_figure_labels", "appendix_figures"))


def caption_labels(counted: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """The labels :func:`report_ingest.narrative_reader.count_captions`
    returned, under the record's names; a kind with none is left out."""
    out: Dict[str, List[str]] = {}
    for key, name in _CAPTION_KEYS:
        labels = [str(x) for x in ((counted or {}).get(key) or ())]
        if labels:
            out[name] = labels
    return out


def _own_logs(record: Any) -> List[Any]:
    """The record's logs that are its own investigation's: not on the pages
    of a bound report that is not one of its volumes."""
    out = []
    for inv in record.investigations:
        first = inv.pages[0] if inv.pages else None
        doc = next((d for d in record.bound_documents
                    if first is not None
                    and d.first_page <= first <= d.last_page), None)
        if doc is None or doc.kind in OWN_LOG_KINDS:
            out.append(inv)
    return out


def _firms(inv: Any) -> List[str]:
    """The firms a log's header fields name: a firm, company or consultant
    field, never the driller's (``drilling.contractor`` included) or a
    person's, and never a value that says nothing ("NA")."""
    said = []
    for key, value in (inv.fields or {}).items():
        key = re.sub(r"[^a-z]", "", str(key).lower())
        if _RE_FIRM_KEY.search(key) and not _RE_NOT_FIRM_KEY.search(key):
            said.append(_flat(value))
    return [s for s in said if s and s.lower() not in _NO_VALUE]


def field_work(record: Any, logs: Optional[Sequence[Any]] = None
               ) -> Optional[Any]:
    """When and by whom this record's own investigation was done, as its
    logs print it: a :class:`FieldWork`, or None when no log prints a date
    or a firm. ``logs`` overrides which logs are read (default: the
    record's own, :func:`_own_logs`)."""
    from report_ingest.model import FieldWork

    logs = _own_logs(record) if logs is None else list(logs)
    dates: List[str] = []
    firms: List[str] = []
    years: set = set()
    pages: set = set()
    holes = 0
    for inv in logs:
        said_dates = [d for d in (_flat(inv.date_started),
                                  _flat(inv.date_finished)) if d]
        said_firms = _firms(inv)
        if not said_dates and not said_firms:
            continue
        holes += 1
        pages.update(int(p) for p in inv.pages[:1])
        for date in said_dates:
            years.update(int(y) for y in _RE_YEAR.findall(date))
            if date not in dates:
                dates.append(date)
        for firm in said_firms:
            if firm.lower() not in {f.lower() for f in firms}:
                firms.append(firm)
    if not holes:
        return None
    return FieldWork(years=sorted(years), dates=dates, firms=firms,
                     holes=holes, pages=sorted(pages))
