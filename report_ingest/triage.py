"""Pass 0b: one look at the whole document before anything is read.

planlens' rules label every page from what the pages print about themselves.
They are a first draft, and the out-of-sample measurement says so: 0.91
accuracy across the fourteen reports they were built on, 0.79 on the next
report anybody opens. The reason is structural rather than fixable by more
rules -- a page only makes sense in the context of the rest of the report,
and a rule sees one page.

So before a reader runs, two model passes look at the document as a whole.
This is the first: ONE structured call that answers what kind of document
this is, whether it is one document or several bound together, what it is
missing, and which workflow it should go down. The second pass
(:mod:`report_ingest.label_review`) then reviews the labels page by page.

WHAT THE MODEL IS GIVEN, AND WHAT IT IS NOT ASKED. The call carries the
whole per-page ledger (one line per page: kind, rule label, confidence, the
rule that fired, heading, running header, printed page number, segment, text
reliability, whether Azure Document Intelligence read it), the outline the
document prints about itself, the text of the front matter through the end
of the contents, and the first contact sheet as a picture. The numbers that
Python can count -- the scanned fraction, the unreliable-text fraction, where
the printed page numbering restarts -- are computed here and handed over as
FACTS. A model asked to count 455 ledger lines will get it nearly right,
which is worse than useless in a scorecard.

The profile it returns chooses the workflow; an odd report is flagged
instead of being forced down the standard path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from report_ingest.bound import BOUND_KINDS
from report_ingest.engine import Engine, text_block, user
from report_ingest.model import DOCUMENT_TYPE_VALUES

__all__ = [
    "DOCUMENT_TYPES", "WORKFLOWS", "TOC_AGREEMENTS", "BOUND_KINDS",
    "DocumentFacts", "BoundDocument", "TriageFindings", "DocumentProfile",
    "document_facts", "front_matter_text", "outline_text", "triage",
    "TRIAGE_SYSTEM",
]

#: The owner's enumeration, kept verbatim so past query outputs stay
#: comparable (plan section 3). It lives in :mod:`report_ingest.model` beside
#: the field that answers with it, and is imported here rather than retyped:
#: two copies of an enumeration are two enumerations, and the one this pass
#: answers with has to be the one the record stores.
DOCUMENT_TYPES: Tuple[str, ...] = DOCUMENT_TYPE_VALUES

#: Which readers run, and whether a person is needed first.
WORKFLOWS: Tuple[str, ...] = (
    "standard", "appendix_only", "partial", "multi_document", "scanned",
    "needs_person",
)

#: How well the contents list matches what was found on the pages.
TOC_AGREEMENTS: Tuple[str, ...] = ("matched", "partial", "none", "no_toc")

#: What a separately-bound part of the file can be. It lives in
#: :mod:`report_ingest.bound`, beside the code that turns one into its own
#: record, and is imported here rather than retyped so that what triage may
#: ANSWER and what the pipeline can BUILD are one list.

#: A page whose text layer is this fraction unreliable, or which has no text
#: at all, cannot be read as text. Both are already decided by planlens
#: (``text_reliable``); this module only counts them.
MAX_FRONT_MATTER_CHARS = 12000
#: How far into a document the front matter may run before we stop looking
#: for the end of the contents. A cover, a letter and a contents list in a
#: geotechnical report are done well inside this.
FRONT_MATTER_SEARCH = 20
#: Pages per contact sheet. planlens' own default, and the number a model
#: can still tell apart at 140 px a thumbnail.
CONTACT_SHEET_PAGES = 48


# -- deterministic facts ----------------------------------------------------

@dataclass(frozen=True)
class DocumentFacts:
    """What Python can count, so the model is never asked to.

    Every field here is a measurement of the open document, not a judgement.
    They are given to the model as facts and copied into the profile
    unchanged, so a scorecard can trust them.
    """

    n_pages: int
    scan_fraction: float
    unreliable_text_fraction: float
    page_numbering_restarts: List[int]
    di_pages: int
    blank_pages: int
    kind_counts: Dict[str, int]
    role_counts: Dict[str, int]
    low_confidence_pages: List[int]
    no_text_pages: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_pages": self.n_pages,
            "scan_fraction": round(self.scan_fraction, 3),
            "unreliable_text_fraction": round(self.unreliable_text_fraction, 3),
            "page_numbering_restarts": list(self.page_numbering_restarts),
            "di_pages": self.di_pages,
            "blank_pages": self.blank_pages,
            "kind_counts": dict(self.kind_counts),
            "role_counts": dict(self.role_counts),
            "n_low_confidence_pages": len(self.low_confidence_pages),
            "n_no_text_pages": len(self.no_text_pages),
        }

    def as_prompt(self) -> str:
        """The facts as the model sees them: counted, not to be recounted."""
        lines = [
            f"pages: {self.n_pages}",
            f"scanned fraction: {self.scan_fraction:.3f}",
            f"unreliable-text fraction: {self.unreliable_text_fraction:.3f}",
            f"pages read by Azure Document Intelligence: {self.di_pages}",
            f"blank pages: {self.blank_pages}",
            "printed page numbering restarts at page indexes: "
            + (", ".join(str(p) for p in self.page_numbering_restarts)
               or "none"),
            "page kinds: " + ", ".join(
                f"{k} {v}" for k, v in sorted(self.kind_counts.items())),
            "rule labels: " + ", ".join(
                f"{k} {v}" for k, v in sorted(self.role_counts.items())),
            f"pages the rules labelled at low confidence: "
            f"{len(self.low_confidence_pages)}",
            f"pages with no readable text at all: {len(self.no_text_pages)}",
        ]
        return "\n".join(lines)


def document_facts(doc, roles: Optional[Sequence[Any]] = None,
                   low_confidence: float = 0.65) -> DocumentFacts:
    """Count what can be counted about an open document.

    ``roles`` is :func:`planlens.document.roles.page_roles` when the caller
    already has it; recomputing page roles reads every page again.
    """
    from planlens.document.model import SOURCE_AZURE_DI

    summaries = list(doc.page_map())
    n = len(summaries)
    if roles is None:
        from planlens.document.roles import page_roles
        roles = page_roles(doc)
    role_of = {r.page: r for r in roles}

    kind_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    scanned = blank = di = unreliable = 0
    no_text: List[int] = []
    low: List[int] = []
    restarts: List[int] = []
    last_printed: Optional[int] = None

    for s in summaries:
        kind_counts[s.kind] = kind_counts.get(s.kind, 0) + 1
        if s.kind == "scanned":
            scanned += 1
        if s.kind == "blank":
            blank += 1
        if s.evidence.get("text_source") == SOURCE_AZURE_DI:
            di += 1
        if not s.text_reliable:
            unreliable += 1
        if not s.text_reliable or s.n_text_chars == 0:
            no_text.append(s.page)
        r = role_of.get(s.page)
        if r is not None:
            role_counts[r.role] = role_counts.get(r.role, 0) + 1
            if r.confidence <= low_confidence:
                low.append(s.page)
        if s.printed_page is not None:
            if last_printed is not None and s.printed_page < last_printed:
                restarts.append(s.page)
            last_printed = s.printed_page

    return DocumentFacts(
        n_pages=n,
        scan_fraction=(scanned / n) if n else 0.0,
        unreliable_text_fraction=(unreliable / n) if n else 0.0,
        page_numbering_restarts=restarts,
        di_pages=di,
        blank_pages=blank,
        kind_counts=kind_counts,
        role_counts=role_counts,
        low_confidence_pages=low,
        no_text_pages=no_text,
    )


def front_matter_text(doc, roles: Optional[Sequence[Any]] = None,
                      max_chars: int = MAX_FRONT_MATTER_CHARS) -> str:
    """The cover and the contents, verbatim, through the end of the contents.

    The end of the contents is where the document stops describing itself and
    starts saying things: the OPENING RUN of pages the rules called a cover, a
    letter, a contents list, a divider or blank, ending at the first page that
    is none of those. It has to be the opening run rather than the last such
    page in the first twenty, because a report's appendix tabs are dividers
    too and one of them can sit on page 6.

    A document whose first page is already narrative has no front matter, so
    the first three pages stand in -- something has to carry the title.
    """
    if roles is None:
        from planlens.document.roles import page_roles
        roles = page_roles(doc)
    role_of = {r.page: r.role for r in roles}
    front = ("cover", "letter", "toc", "divider", "other")
    last = -1
    for page in range(min(FRONT_MATTER_SEARCH, doc.n_pages)):
        if role_of.get(page) not in front and doc.summary(page).kind != "blank":
            break
        last = page
    if last < 0:
        last = min(2, doc.n_pages - 1)

    out: List[str] = []
    used = 0
    for page in range(last + 1):
        lines = [ln.text for ln in doc.page(page, tables=False).lines
                 if (ln.text or "").strip()]
        block = f"=== page {page} ({role_of.get(page, 'other')}) ===\n" \
                + "\n".join(lines)
        if used + len(block) > max_chars:
            out.append(block[:max(0, max_chars - used)])
            out.append("\n[front matter cut at the size limit]")
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)


# -- what the model returns -------------------------------------------------

class BoundDocument(BaseModel):
    """One separately-bound part of the file."""

    kind: str = Field(description="one of: " + ", ".join(BOUND_KINDS))
    pages: str = Field(
        description="0-based PDF page range, e.g. '112-184' or '0-93,210-240'")
    title: Optional[str] = Field(
        description="the title the part prints for itself, or null")


class TriageFindings(BaseModel):
    """The judgements, and only the judgements, the model is asked for.

    The counted facts are not here: they are measured in Python by
    :func:`document_facts` and assembled into the profile beside these.
    """

    document_type: str = Field(
        description="one of: " + "; ".join(DOCUMENT_TYPES))
    bound_together: List[BoundDocument] = Field(
        description="parts bound into this file that are their own document "
                    "- volumes, an appended prior report, a data report "
                    "inside a design report. Empty when it is one document.")
    has_narrative: bool = Field(
        description="does the file contain the report's own prose findings "
                    "and recommendations")
    has_logs: bool = Field(
        description="does it contain boring, test pit, CPT or DCP logs")
    has_lab: bool = Field(description="does it contain laboratory test data")
    has_calcs: bool = Field(
        description="does it contain calculation printouts or worksheets")
    languages: List[str] = Field(
        description="languages printed anywhere in the file, English names, "
                    "most common first")
    toc_agreement: str = Field(
        description="one of: " + ", ".join(TOC_AGREEMENTS)
        + " - how well the contents list matches what the pages turned out "
          "to be")
    anomalies: List[str] = Field(
        description="anything a reader should be warned about, one short "
                    "phrase each; empty list when there is nothing")
    workflow: str = Field(
        description="one of: " + ", ".join(WORKFLOWS))
    rationale: str = Field(
        description="why this workflow, in 80 words or fewer")


@dataclass
class DocumentProfile:
    """What the document is, as pass 0b decided it.

    The judgement fields come from the model; ``scan_fraction``,
    ``unreliable_text_fraction`` and ``page_numbering_restarts`` are
    measurements copied from :class:`DocumentFacts`.
    """

    document_type: str
    workflow: str
    bound_together: List[Dict[str, Any]]
    has_narrative: bool
    has_logs: bool
    has_lab: bool
    has_calcs: bool
    languages: List[str]
    toc_agreement: str
    anomalies: List[str]
    rationale: str
    scan_fraction: float
    unreliable_text_fraction: float
    page_numbering_restarts: List[int]
    facts: Optional[DocumentFacts] = None
    model: str = ""
    cost: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "document_type": self.document_type,
            "workflow": self.workflow,
            "bound_together": [dict(b) for b in self.bound_together],
            "has_narrative": self.has_narrative,
            "has_logs": self.has_logs,
            "has_lab": self.has_lab,
            "has_calcs": self.has_calcs,
            "languages": list(self.languages),
            "toc_agreement": self.toc_agreement,
            "anomalies": list(self.anomalies),
            "rationale": self.rationale,
            "scan_fraction": round(self.scan_fraction, 3),
            "unreliable_text_fraction": round(self.unreliable_text_fraction, 3),
            "page_numbering_restarts": list(self.page_numbering_restarts),
            "model": self.model,
            "cost": dict(self.cost),
        }
        if self.facts is not None:
            out["facts"] = self.facts.to_dict()
        return out

    def summary_row(self) -> Dict[str, Any]:
        """The one line a scorecard table carries for this report."""
        return {
            "document_type": self.document_type,
            "workflow": self.workflow,
            "bound_together": len(self.bound_together),
            "toc_agreement": self.toc_agreement,
            "scan_fraction": round(self.scan_fraction, 3),
        }


TRIAGE_SYSTEM = """\
You are triaging one engineering report PDF before any of it is read in
detail. You are given everything the document says about ITSELF, never a
summary of it: one ledger line per page, the contents and figure/table/
appendix lists it prints, the text of its front matter, a contact sheet of
its first pages as a picture, and a block of facts already counted from the
file.

Your job is to say what this file IS, so the right readers run on it and an
odd file is flagged instead of forced down the standard path.

How to read the ledger. Each line is one page:

    p007 mixed  boring_log  0.90 [page-title] "LOG OF BORING" hdr="..."
         pp=1/3 seg=4 chars=550 text_ok=Y di=N item_4

kind is the page's shape; then the rule label and how much the rule that
fired is worth; [tag] names the rule; the quoted string is the page's own
largest heading; hdr is its running header; pp is the page number PRINTED on
the page; seg is the constituent document planlens assigned it; chars is how
much text it has; text_ok=N means its text layer is garbage or absent; di=Y
means Azure Document Intelligence read it. A label of INHERITED came from
the appendix tab rather than the page, and 'other' with candidates means the
rules could not settle it. Those are the rules' weak spots, not yours to fix
here -- the next pass reviews them page by page.

Rules for your answer.

- The counted facts are counts, not opinions. Do not recount pages.
- bound_together is for a part of the file that is its OWN document: a
  second volume, a prior report appended whole, a data report inside a
  design report. A plain appendix of lab sheets is not one. Give its page
  range in 0-based PDF page numbers, which is what the ledger uses. This
  answer is ACTED ON: each range you name is read separately into its own
  record, so that an earlier firm's borings are that firm's and not this
  report's. Give the range the document actually occupies, from its own
  cover to its last page, and leave the list empty when this is one
  document.
- toc_agreement: 'matched' when the contents list's sections are where it
  says they are; 'partial' when some are and some are not, or the list is
  incomplete; 'none' when the list and the file disagree; 'no_toc' when the
  file prints no contents list.
- workflow: 'standard' for a whole report with narrative, logs and lab;
  'appendix_only' when the file is appendix or figure material with no
  report around it; 'partial' when a whole report was meant to be here and
  pages are plainly missing; 'multi_document' when several documents are
  bound together and each needs its own pass; 'scanned' when most of the
  file has no reliable text and must be read optically or by eye;
  'needs_person' when you cannot tell what this is, or something is wrong
  enough that a reader should look before a machine does.
- Say 'needs_person' when you mean it. A wrong confident answer costs more
  than a flag.
- anomalies are short and concrete: what a reader would want warning about.
- rationale: 80 words or fewer, plain sentences.
"""


def triage(doc, roles=None, outline=None, *, engine: Engine,
           facts: Optional[DocumentFacts] = None,
           max_ledger_chars: int = 120000) -> DocumentProfile:
    """Run pass 0b over one open document: one structured call.

    ``doc`` is an open :class:`planlens.document.Document`; ``roles`` and
    ``outline`` are :func:`~planlens.document.roles.page_roles` and
    :func:`~planlens.document.roles.document_outline` when the caller
    already has them, which saves reading every page again.
    """
    from planlens.document.roles import document_outline, page_ledger, page_roles

    if roles is None:
        roles = page_roles(doc)
    if outline is None:
        outline = document_outline(doc)
    if facts is None:
        facts = document_facts(doc, roles)

    ledger = "\n".join(page_ledger(doc, roles))
    if len(ledger) > max_ledger_chars:
        # A 700-page ledger still fits comfortably; this only bites on a
        # document far outside the corpus, and a cut ledger must say so.
        ledger = (ledger[:max_ledger_chars]
                  + "\n[ledger cut at the size limit; later pages not shown]")

    sheets = doc.render_thumbnails(pages=None, per_sheet=CONTACT_SHEET_PAGES)
    first_sheet = [sheets[0][0]] if sheets else []
    if sheets:
        shown = sheets[0][1].get("pages") or []
        sheet_note = (
            f"The picture is a contact sheet of pages {shown[0]} to "
            f"{shown[-1]}, {len(sheets)} sheet(s) in all. Each thumbnail is "
            f"labelled with its 0-based page index and its kind.")
    else:
        sheet_note = "No contact sheet could be rendered."

    body = [
        "FACTS ALREADY COUNTED FROM THE FILE",
        facts.as_prompt(),
        "",
        "WHAT THE DOCUMENT PRINTS ABOUT ITSELF (its outline)",
        outline_text(outline),
        "",
        "FRONT MATTER, VERBATIM",
        front_matter_text(doc, roles),
        "",
        "PAGE LEDGER, ONE LINE PER PAGE",
        ledger,
        "",
        sheet_note,
    ]
    reply = engine.complete([user(text_block("\n".join(body)))],
                            system=TRIAGE_SYSTEM,
                            images=first_sheet,
                            output_format=TriageFindings)
    found = reply.parsed
    if found is None:
        raise RuntimeError(
            f"triage returned no structured answer (stop_reason "
            f"{reply.stop_reason!r})")
    return DocumentProfile(
        document_type=found.document_type,
        workflow=found.workflow,
        bound_together=[b.model_dump() for b in found.bound_together],
        has_narrative=found.has_narrative,
        has_logs=found.has_logs,
        has_lab=found.has_lab,
        has_calcs=found.has_calcs,
        languages=list(found.languages),
        toc_agreement=found.toc_agreement,
        anomalies=list(found.anomalies),
        rationale=found.rationale,
        scan_fraction=facts.scan_fraction,
        unreliable_text_fraction=facts.unreliable_text_fraction,
        page_numbering_restarts=list(facts.page_numbering_restarts),
        facts=facts,
        model=reply.model or getattr(engine, "name", ""),
        cost={"calls": 1,
              "input_tokens": reply.usage.input_tokens,
              "output_tokens": reply.usage.output_tokens,
              "cache_read_tokens": reply.usage.cache_read_tokens,
              "seconds": round(reply.seconds, 1),
              "dollars": round(reply.usage.dollars(reply.model), 4)},
    )


def outline_text(outline, max_chars: int = 30000) -> str:
    """The outline as lines a model reads, not as JSON."""
    out: List[str] = []
    if getattr(outline, "no_dividers", False):
        out.append("NO APPENDIX TAB ANYWHERE IN THIS DOCUMENT. Nothing "
                   "inherits a role from a tab, so the rule labels lean on "
                   "each page's own shape and are weaker than usual.")
    for kind, title in (("contents", "CONTENTS"), ("figure", "LIST OF FIGURES"),
                        ("table", "LIST OF TABLES"),
                        ("appendix", "LIST OF APPENDICES")):
        entries = outline.of_kind(kind)
        if not entries:
            continue
        out.append(f"{title} ({len(entries)} entries)")
        for e in entries:
            where = (f"-> page {e.page}" if e.page is not None
                     else "-> not placed")
            printed = f" printed {e.printed_page}" if e.printed_page else ""
            number = f"{e.number} " if e.number else ""
            out.append(f"  {number}{e.title}{printed} {where}")
    if outline.dividers:
        out.append(f"DIVIDERS AND TABS ({len(outline.dividers)})")
        for m in outline.dividers:
            out.append(f"  page {m.page}: {m.text}")
    if outline.captions:
        out.append(f"FIGURE CAPTIONS ({len(outline.captions)})")
        for m in outline.captions:
            out.append(f"  page {m.page}: {m.text}")
    if outline.headings:
        out.append(f"NARRATIVE SECTION HEADINGS ({len(outline.headings)})")
        for m in outline.headings:
            out.append(f"  page {m.page}: {m.text}")
    text = "\n".join(out) if out else "the document prints no outline at all"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[outline cut at the size limit]"
    return text
