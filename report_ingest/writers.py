"""The record's four outputs, and the index that makes many of them a library.

The record is the product. These are its exports, and every one of them is
written from the record alone -- nothing here asks a model anything, and
nothing here knows a fact the record does not carry.

``report.record.json``
    The record itself, as JSON. Everything else on this list is derived from
    it, so a consumer that wants the data reads this and ignores the rest.

``report.summary.md``
    The two query schemas answered in prose, each answer with the pages it
    came from, then what was extracted and the QA list. Written for a person
    who is about to decide whether to open the report.

``report.page.md``
    The same report in the owner's WikiLLM page format -- front matter, a
    summary, key takeaways, key parameters -- so a report sits in the library
    beside the papers and manuals and is searched the same way. The record's
    own sections follow as tables, which is what makes the page answerable:
    a library agent can read the explorations off it without opening the JSON.

``report.diggs.xml``
    DIGGS 2.6, through the existing writer, with BOTH gates run: the bundled
    XSD, and a round trip through the app's own readers compared value by
    value. Each verdict is written into the record's QA, because a file that
    failed its gate and a file that was never checked must not look the same.

``reports.db``
    One SQLite row per report, keyed by a stable hash of the source file, so a
    folder of reports becomes something searchable. Re-ingesting a report
    updates its row rather than adding a second one.

A REPORT BOUND INSIDE ANOTHER ONE gets the same five outputs, in its own
folder, and is a row of its own in the same library with ``parent`` set to the
key of the report it is bound inside. Its DIGGS file holds ITS explorations
and the parent's holds the parent's: DIGGS 2.6 has no way to say "this
sampling feature belongs to a different report bound into this one", and a
file that quietly merged the two would be wrong in the one way this whole
exercise exists to prevent. The link lives in the record
(``bound_documents`` and ``parent``), in the summary page's own section and
in the library's ``parent`` column.

THE ORDER MATTERS. The DIGGS file is written and gated FIRST, because its two
verdicts are QA entries and the record JSON has to carry them. Then the
record, then the two pages, then the database row.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from report_ingest.model import (
    GENERAL_FIELDS, NATURAL_HAZARD_FIELDS, QAEntry, Quantity, ReportRecord,
)

__all__ = [
    "write_outputs", "WrittenOutputs", "record_key", "parent_key",
    "summary_markdown", "bound_section", "library_page", "front_matter",
    "upsert_report", "open_library", "DB_SCHEMA_VERSION", "TIERS",
    "STATUSES", "CONFIDENCES",
]

#: The library database's own schema version, stored in its ``meta`` table.
#: Bumped when a column changes meaning; a new column is not a bump.
DB_SCHEMA_VERSION = "1"

#: The WikiLLM vocabulary these exports use, kept here so a reader of this
#: module can see the whole of what a page may say without opening the
#: library's schema note.
TIERS: Tuple[str, ...] = ("deep", "reference")
STATUSES: Tuple[str, ...] = ("summarized", "pending", "needs_ocr", "failed")
CONFIDENCES: Tuple[str, ...] = ("high", "medium", "low")

#: The file names. Fixed: a consumer that has one output knows the others.
RECORD_NAME = "report.record.json"
SUMMARY_NAME = "report.summary.md"
PAGE_NAME = "report.page.md"
DIGGS_NAME = "report.diggs.xml"
DB_NAME = "reports.db"

#: How much of a file is hashed to key it. A geotechnical report can run to
#: hundreds of megabytes of scanned images and the key has to be cheap; the
#: first megabyte plus the length distinguishes any two documents that are
#: not byte-identical openings.
HASH_BYTES = 1_000_000


# ---------------------------------------------------------------------------
# the key
# ---------------------------------------------------------------------------

def record_key(record: ReportRecord, source: Any = None,
               parent_base: str = "") -> str:
    """A stable identifier for this report: the same file, the same key.

    From the SOURCE FILE when there is one -- its length and its first
    megabyte -- so that re-ingesting the same PDF updates its row instead of
    adding a second one, whatever the file has been renamed to since. With no
    readable source it falls back to what the record itself says it is (the
    report id, the project number and name), which is stable across runs of
    the same report and is all there is to go on.

    A record of a report BOUND INSIDE another one comes off the same file as
    its parent, so the file's key alone would put the two of them in one
    library row and the second written would win. Its own key is therefore
    the file's key folded with its handle inside the parent -- stable across
    runs, distinct from the parent's, and :func:`parent_key` recovers the
    parent's from the same two things.

    ``parent_base`` is the parent's own key, for a caller that wrote the
    parent and KNOWS it. It matters only where there is no readable source
    file and the fallback is the record's own identity: a bound document's
    identity is not its parent's, so the fallback cannot reconstruct the
    parent's key and the caller has to say. Ignored for a record with no
    parent.
    """
    parent = record.parent
    if parent is None:
        return parent_key(record, source)
    base = parent_base or parent_key(record, source)
    if parent.bound_id:
        return hashlib.sha256(
            f"{base}|{parent.bound_id}".encode("utf-8")).hexdigest()[:16]
    return base


def parent_key(record: ReportRecord, source: Any = None) -> str:
    """The key of the file this record came off, bound document or not.

    For an ordinary record this IS its key. For a bound document it is the
    key of the record it is bound inside, which is what the library's
    ``parent`` column stores and what makes one SQL query return a report
    and everything bound into it.
    """
    digest = hashlib.sha256()
    path = str(source) if source else ""
    if path and os.path.isfile(path):
        size = os.path.getsize(path)
        digest.update(str(size).encode("utf-8"))
        with open(path, "rb") as handle:
            digest.update(handle.read(HASH_BYTES))
    elif isinstance(source, (bytes, bytearray)):
        digest.update(str(len(source)).encode("utf-8"))
        digest.update(bytes(source[:HASH_BYTES]))
    else:
        parts = [record.document.report_id, record.project.number,
                 record.project.name, record.general.projectNumber or "",
                 record.general.projectName or ""]
        digest.update("|".join(str(p) for p in parts).encode("utf-8"))
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# what the record says about itself
# ---------------------------------------------------------------------------

def _first(*values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def title_of(record: ReportRecord) -> str:
    """What to call this report on a library page."""
    name = _first(record.general.projectName, record.project.name)
    kind = (record.general.documentType or "geotechnical report").strip()
    if name:
        return f"{name} - {kind}"
    number = _first(record.general.projectNumber, record.project.number,
                    record.document.report_id)
    return f"{kind}{f' {number}' if number else ''}".strip().capitalize()


def year_of(record: ReportRecord) -> Optional[int]:
    iso = record.natural_hazards.reportDateISO
    if iso and len(iso) >= 4 and iso[:4].isdigit():
        return int(iso[:4])
    return None


def tier_of(record: ReportRecord) -> str:
    """``deep`` for a report, ``reference`` for a pile of appendix material.

    The library's own rule is about length and document type: a work that is
    a single finding gets the full treatment, and a comprehensive reference
    gets a scope note instead. A geotechnical report is the first of those.
    A file that is appendix or figure material has no findings to summarise
    and is filed as the second.
    """
    if record.general.documentType == "report appendix or figure(s)":
        return "reference"
    return "deep"


def status_of(record: ReportRecord) -> str:
    """Whether this record is finished, and if not, why not."""
    if record.document.workflow == "needs_person":
        return "pending"
    if any(entry.kind == "unreadable" for entry in record.qa) or \
            record.document.scan_fraction >= 0.5:
        return "needs_ocr"
    if not record.general.answered() and not record.investigations:
        return "failed"
    return "summarized"


def confidence_of(record: ReportRecord) -> str:
    """How much of this record a reader should take on trust.

    Low when the narrative barely answered or a person was asked for; medium
    when something disagrees with something else, or the file is largely a
    scan; high otherwise. Deliberately blunt: it is a flag for a human
    reviewer's attention, not a probability.
    """
    if record.document.workflow == "needs_person" or \
            len(record.general.answered()) < 5:
        return "low"
    if record.document.scan_fraction >= 0.5:
        return "low"
    if any(entry.kind in ("conflict", "count_mismatch", "unreadable")
           for entry in record.qa):
        return "medium"
    return "high"


#: ``USCS first letter -> the library's material tag``.
_USCS_MATERIAL = {"C": "Clay", "S": "Sand", "M": "Silt", "G": "Gravel",
                  "O": "Organic soil", "P": "Peat"}
#: ``lab kind -> the library's method tag``, where the library has one.
_LAB_METHOD = {
    "triaxial": "Triaxial", "direct_shear": "Direct Shear",
    "swell_consolidation": "Consolidation Test",
    "atterberg": "Atterberg Limits", "gradation": "Sieve Analysis",
    "compaction": "Proctor Compaction", "cbr": "CBR",
    "permeability": "Permeability", "chemical": "Corrosivity Testing",
    "unconfined": "Unconfined Compression",
    "unconfined_rock": "Unconfined Compression (rock)",
}
#: Words in a recommended foundation that name a library topic.
_FOUNDATION_TOPIC = (
    ("pile", "Deep Foundations (Piles)"),
    ("drilled shaft", "Drilled Shafts/Caissons"),
    ("caisson", "Drilled Shafts/Caissons"),
    ("micropile", "Micropiles"),
    ("mat", "Shallow Foundations"),
    ("footing", "Shallow Foundations"),
    ("raft", "Shallow Foundations"),
    ("slab", "Shallow Foundations"),
    ("ground improvement", "Ground Improvement"),
    ("aggregate pier", "Ground Improvement"),
    ("stone column", "Ground Improvement"),
    ("retaining", "Retaining Walls"),
)


def _add(tags: List[str], value: str) -> None:
    if value and value not in tags:
        tags.append(value)


def tags_for(record: ReportRecord) -> Dict[str, List[str]]:
    """The library's four tag lists, derived from what the record holds.

    Every tag is earned by something IN the record -- an investigation kind, a
    test kind, a USCS symbol, an answered hazard question. Nothing is tagged
    because reports usually have it.
    """
    disciplines: List[str] = ["Geotechnical"]
    topics: List[str] = []
    methods: List[str] = []
    materials: List[str] = []

    hazards = record.natural_hazards
    if hazards.answered():
        _add(disciplines, "Seismic/Earthquake")
    if record.general.documentType == "environmental report":
        _add(disciplines, "Environmental/Geoenvironmental")

    kinds = {inv.kind for inv in record.investigations}
    if record.investigations:
        _add(topics, "Site Investigation/Drilling")
    if any(inv.spt for inv in record.investigations):
        _add(topics, "In-situ Testing")
        _add(methods, "SPT")
    if "cpt" in kinds:
        _add(topics, "In-situ Testing")
        _add(methods, "CPT/CPTu")
    if "dcp" in kinds:
        _add(methods, "DCP")
    if record.lab_tests:
        _add(topics, "Lab Testing")
    for test in record.lab_tests:
        _add(methods, _LAB_METHOD.get(test.kind, ""))
        if test.kind == "swell_consolidation":
            _add(topics, "Settlement/Consolidation")
        if test.kind == "chemical":
            _add(topics, "Corrosion/Durability")
    if record.general.bearingCapacityValues or record.general.bearingCapacity:
        _add(topics, "Bearing Capacity")
    for foundation in (record.general.recommendedFoundations or ()):
        lowered = foundation.lower()
        for word, topic in _FOUNDATION_TOPIC:
            if word in lowered:
                _add(topics, topic)
    if hazards.liquefactionPotential and \
            hazards.liquefactionPotential not in ("none", "not evaluated"):
        _add(topics, "Liquefaction")
    if hazards.soilCorrosion == "yes":
        _add(topics, "Corrosion/Durability")
    for hazard in (hazards.earthHazardsExposed or ()):
        if hazard == "landslide":
            _add(topics, "Slope Stability")

    for inv in record.investigations:
        for layer in inv.layers:
            _add(materials, _USCS_MATERIAL.get((layer.uscs or "")[:1], ""))
        for sample in inv.samples:
            if sample.rqd_percent is not None:
                _add(materials, "Rock")
    for stratum in record.general.strataList:
        _add(materials, _USCS_MATERIAL.get((stratum.uscs or "")[:1], ""))

    standards: List[str] = []
    for test in record.lab_tests:
        body = (test.standard or "").split()
        if body:
            _add(standards, body[0].rstrip(":,"))
    if hazards.asceSevenVersionNormalized:
        _add(standards, "ASCE")
    return {"disciplines": disciplines, "topics": topics, "methods": methods,
            "materials": materials, "standards_referenced": standards}


def _show(value: Any) -> str:
    """One value as a page prints it."""
    if value is None:
        return ""
    if isinstance(value, Quantity):
        return f"{value.value:g} {value.unit}".strip()
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "; ".join(_show(v) for v in value)
    return str(value)


def _cites(record: ReportRecord, section: str, name: str) -> str:
    holder = getattr(record, section)
    rows = holder.citations.get(name) or []
    return ", ".join(f"p{row.page}" for row in rows)


def key_parameters(record: ReportRecord) -> List[str]:
    """The numbers a reader of the library page came for."""
    out: List[str] = []
    for value in record.general.bearingCapacityValues:
        detail = " ".join(x for x in (value.foundation_type, value.condition)
                          if x)
        out.append(f"Allowable bearing pressure {_show(value.value)}"
                   + (f" - {detail}" if detail else ""))
    hazards = record.natural_hazards
    if hazards.siteClassNormalized:
        out.append(f"Site class {hazards.siteClassNormalized}"
                   + (f" ({hazards.seismicCodeUsed})"
                      if hazards.seismicCodeUsed else ""))
    if hazards.liquefactionPotential:
        out.append(f"Liquefaction potential: {hazards.liquefactionPotential}")
    n_values = [spt.n for inv in record.investigations for spt in inv.spt
                if spt.n is not None]
    if n_values:
        out.append(f"SPT N from {min(n_values)} to {max(n_values)} over "
                   f"{len(n_values)} driven record(s)")
    depths = [inv.total_depth for inv in record.investigations
              if inv.total_depth is not None]
    if depths:
        deepest = max(depths, key=lambda q: q.to_si().value
                      if q.to_si() else q.value)
        out.append(f"Deepest exploration {_show(deepest)}")
    water = [level.depth for inv in record.investigations
             for level in inv.water if level.depth is not None]
    if water:
        shallowest = min(water, key=lambda q: q.to_si().value
                         if q.to_si() else q.value)
        out.append(f"Groundwater as shallow as {_show(shallowest)}")
    return out


def key_takeaways(record: ReportRecord) -> List[str]:
    """What an engineer would want at a glance, from the record alone."""
    out: List[str] = []
    general = record.general
    if general.testingProgramSummary:
        out.append(general.testingProgramSummary)
    counts = record.narrative.found_counts or {}
    if counts:
        got = ", ".join(f"{value} {name.replace('_', ' ')}"
                        for name, value in sorted(counts.items()) if value)
        if got:
            out.append(f"Read from this report: {got}.")
    if general.recommendedFoundations:
        out.append("Recommended foundations: "
                   + "; ".join(general.recommendedFoundations) + ".")
    for line in (general.bearingCapacity or [])[:3]:
        out.append(line)
    if record.natural_hazards.naturalHazardSummary:
        out.append(record.natural_hazards.naturalHazardSummary)
    if general.strata:
        out.append(general.strata)
    conflicts = [entry for entry in record.qa if entry.kind == "conflict"]
    if conflicts:
        out.append(f"{len(conflicts)} value(s) disagree between the summary "
                   f"table and the sheets; see the QA section.")
    return out


# ---------------------------------------------------------------------------
# report.summary.md
# ---------------------------------------------------------------------------

def _answer_rows(record: ReportRecord, section: str,
                 names: Sequence[str]) -> List[str]:
    holder = getattr(record, section)
    rows = ["| Question | Answer | Pages |", "|---|---|---|"]
    for name in names:
        value = getattr(holder, name, None)
        if value is None:
            continue
        rows.append(f"| `{name}` | {_md(_show(value))} | "
                    f"{_cites(record, section, name)} |")
    if len(rows) == 2:
        return ["_Nothing in this section was answered._"]
    return rows


def _md(text: str) -> str:
    """One cell's text, safe inside a markdown table."""
    return " ".join(str(text).split()).replace("|", "\\|")


#: ``BoundReport.kind`` as a summary page prints it.
_BOUND_KIND_WORDS = {
    "volume": "another volume of this report",
    "appended_prior_report": "an earlier report, appended whole",
    "data_report": "a data report bound into this one",
    "other": "a report bound into this one",
}

#: What the counts cell names, in the order it names them.
_BOUND_HOLDS = (("investigations", "exploration"), ("lab_tests", "lab test"),
                ("samples", "sample"), ("spt", "driven record"))


def _bound_holds(counts: Dict[str, int]) -> str:
    """What a bound document turned out to hold, as a cell of a table."""
    parts = []
    for name, word in _BOUND_HOLDS:
        value = int(counts.get(name) or 0)
        if value:
            parts.append(f"{value} {word}" + ("s" if value != 1 else ""))
    return ", ".join(parts) or "nothing that was read"


def bound_section(record: ReportRecord) -> List[str]:
    """The reports bound inside this one, and where each of them went.

    Written for the person about to decide whether to open the PDF, and it
    has one job beyond listing: to say that the explorations in those pages
    are NOT in the counts above. A reader who takes this report's boring
    count as the number of borings in the file is wrong by the number in the
    appendix, and that is the mistake reading them separately exists to stop.
    """
    if not record.bound_documents:
        return []
    out = ["", "## Reports bound inside this one", "",
           "These page ranges are other reports reproduced whole inside this "
           "file. Each is read into its OWN record: what they hold is theirs "
           "and is in none of the counts above.", "",
           "| Report | What it is | Pages | What it holds | Where |",
           "|---|---|---|---|---|"]
    for row in record.bound_documents:
        named = " · ".join(x for x in (row.title, row.firm, row.date) if x)
        where = row.folder or "-"
        if not row.read:
            out.append(f"| {_md(named or 'not identified')} | "
                       f"{_md(_BOUND_KIND_WORDS.get(row.kind, row.kind))} | "
                       f"{row.pages} | _not read; see the QA section_ | - |")
            continue
        out.append(f"| {_md(named or row.report_id or row.bound_id)} | "
                   f"{_md(_BOUND_KIND_WORDS.get(row.kind, row.kind))} | "
                   f"{row.pages} | {_md(_bound_holds(row.counts))} | "
                   f"`{where}` |")
    return out


def summary_markdown(record: ReportRecord) -> str:
    """The two schemas answered in prose, then the counts and the QA list."""
    general, hazards = record.general, record.natural_hazards
    out: List[str] = [f"# {title_of(record)}", ""]

    line = " · ".join(x for x in (
        general.geotechnicalEngineerFirm or "",
        hazards.reportDate or "",
        f"{record.document.n_pages} pages" if record.document.n_pages else "",
        general.documentType or "") if x)
    if line:
        out += [line, ""]
    parent = record.parent
    if parent is not None:
        out += [f"_This report is bound inside another one"
                + (f" (`{parent.report_id}`)" if parent.report_id else "")
                + f", at pages {parent.pages} of that file. Every page "
                  f"number below is a page of that same file._", ""]
    if general.quickSummary:
        out += [general.quickSummary, ""]

    out += ["## The general questions", ""]
    out += _answer_rows(record, "general", GENERAL_FIELDS)
    out += ["", "## The natural-hazard questions", ""]
    out += _answer_rows(record, "natural_hazards", NATURAL_HAZARD_FIELDS)

    if general.bearingCapacityValues:
        out += ["", "## Bearing recommendations, as values", "",
                "| Pressure | Foundation | Condition | Pages |",
                "|---|---|---|---|"]
        for value in general.bearingCapacityValues:
            pages = ", ".join(f"p{c.page}" for c in value.citation)
            out.append(f"| {_show(value.value)} | "
                       f"{_md(value.foundation_type)} | "
                       f"{_md(value.condition)} | {pages} |")

    if general.strataList:
        out += ["", "## The profile, as records", "",
                "| Stratum | Description | Top | Bottom | USCS |",
                "|---|---|---|---|---|"]
        for stratum in general.strataList:
            out.append(f"| {_md(stratum.name)} | {_md(stratum.description)} | "
                       f"{_show(stratum.top)} | {_show(stratum.bottom)} | "
                       f"{stratum.uscs} |")

    out += bound_section(record)

    asked = record.narrative.extra_answers
    if asked:
        out += ["", "## What else was asked", "",
                "| Question | Answer | Page |", "|---|---|---|"]
        for row in asked:
            answer = row.get("answer") or "_the narrative does not say_"
            page = row.get("page")
            out.append(f"| {_md(row.get('question', ''))} | {_md(answer)} | "
                       f"{'p' + str(page) if page is not None else ''} |")

    out += ["", "## What was extracted", ""]
    counts = record.counts()
    out += ["| What | How many |", "|---|---|"]
    for name, value in counts.items():
        out.append(f"| {name.replace('_', ' ')} | {value} |")
    stated = record.narrative.stated_counts
    found = record.narrative.found_counts
    if stated or found:
        out += ["", "The narrative states "
                + (", ".join(f"{v} {k.replace('_', ' ')}"
                             for k, v in sorted(stated.items()))
                   or "no counts")
                + "; the appendix yielded "
                + (", ".join(f"{v} {k.replace('_', ' ')}"
                             for k, v in sorted(found.items()))
                   or "nothing") + "."]

    out += ["", "## What a reviewer should know", ""]
    if not record.qa:
        out.append("_Nothing was skipped, partial or in conflict._")
    else:
        out += ["| Kind | Where | What |", "|---|---|---|"]
        for entry in record.qa:
            detail = entry.detail
            if entry.values:
                detail += " (" + "; ".join(entry.values[:4]) + ")"
            if entry.pages:
                detail += " [pages " + ", ".join(
                    str(p) for p in entry.pages[:10]) + "]"
            out.append(f"| {entry.kind} | {_md(entry.where)} | "
                       f"{_md(detail)} |")

    document = record.document
    out += ["", "---", "",
            f"Read with planlens {document.planlens_version or '?'} on the "
            f"{document.workflow or 'standard'} workflow; "
            f"{document.model_calls} model call(s), "
            f"{document.input_tokens:,} input and "
            f"{document.output_tokens:,} output tokens. Record schema "
            f"{record.schema_version}."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# report.page.md -- the library page
# ---------------------------------------------------------------------------

def _yaml_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(json.dumps(str(v)) for v in value) + "]"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return json.dumps(str(value))


def front_matter(record: ReportRecord, key: str,
                 source: str = "") -> Dict[str, Any]:
    """The library page's front matter, field for field.

    The owner's library keys every record the same way -- id, title, authors,
    year, source, doc_type, the four tag lists, the standards, a tier, a
    status and a confidence -- so a geotechnical report drops into it beside
    the papers and the manuals and is searched by the same fields. ``doc_type``
    is the owner's own ``documentType`` answer rather than the library's
    document vocabulary, because that is the answer this pipeline produces and
    it is the one that has to stay comparable.
    """
    tags = tags_for(record)
    year = year_of(record)
    return {
        "id": key,
        "report_id": record.document.report_id,
        "title": title_of(record),
        "authors": record.general.geotechnicalEngineerFirm or "",
        "year": year if year is not None else "",
        "source": _first(record.general.geotechnicalEngineerFirm,
                         record.project.client),
        "doc_type": record.general.documentType or "",
        "tier": tier_of(record),
        "disciplines": tags["disciplines"],
        "topics": tags["topics"],
        "methods": tags["methods"],
        "materials": tags["materials"],
        "standards_referenced": tags["standards_referenced"],
        "confidence": confidence_of(record),
        "status": status_of(record),
        "n_pages": record.document.n_pages,
        "original_path": source,
    }


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    empty = True
    for row in rows:
        empty = False
        out.append("| " + " | ".join(_md(_show(cell)) for cell in row) + " |")
    return [] if empty else out


def library_page(record: ReportRecord, key: str, source: str = "") -> str:
    """The record as a page in the owner's WikiLLM format."""
    meta = front_matter(record, key, source)
    out: List[str] = ["---"]
    for name, value in meta.items():
        if value == "" or value == []:
            continue
        out.append(f"{name}: {_yaml_value(value)}")
    out += ["---", "", f"# {meta['title']}", ""]

    byline = " · ".join(x for x in (
        meta["authors"], str(meta["year"] or ""),
        record.natural_hazards.reportDate or "") if x)
    if byline:
        out += [f"**{byline}**", ""]

    out += ["## Summary", "",
            record.general.quickSummary
            or "_The narrative gave no summary._", ""]

    takeaways = key_takeaways(record)
    if takeaways and meta["tier"] == "deep":
        out += ["## Key takeaways", ""]
        out += [f"- {line}" for line in takeaways]
        out.append("")
    parameters = key_parameters(record)
    if parameters and meta["tier"] == "deep":
        out += ["## Key parameters", ""]
        out += [f"- {line}" for line in parameters]
        out.append("")

    out += ["## The questions answered", ""]
    out += _answer_rows(record, "general", GENERAL_FIELDS)
    out += [""]
    out += _answer_rows(record, "natural_hazards", NATURAL_HAZARD_FIELDS)
    out += [""]

    rows = _table(
        ["Exploration", "Kind", "Total depth", "Ground elevation", "Layers",
         "Samples", "N values", "Water"],
        [[inv.investigation_id, inv.kind, inv.total_depth, inv.elevation,
          len(inv.layers), len(inv.samples),
          ", ".join(str(s.n) for s in inv.spt if s.n is not None) or "",
          ", ".join(_show(w.depth) for w in inv.water if w.depth)]
         for inv in record.investigations])
    if rows:
        out += ["## Explorations", ""] + rows + [""]

    rows = _table(
        ["Test", "Exploration", "Depth", "Standard", "Linked sample", "Pages"],
        [[test.kind, test.investigation_id or "-", test.depth_top,
          test.standard, test.linked_sample_id or "-",
          ", ".join(str(p) for p in test.pages)]
         for test in record.lab_tests])
    if rows:
        out += ["## Laboratory testing", ""] + rows + [""]

    rows = _table(
        ["Kind", "Where", "What"],
        [[entry.kind, entry.where,
          entry.detail + (" (" + "; ".join(entry.values[:3]) + ")"
                          if entry.values else "")]
         for entry in record.qa])
    if rows:
        out += ["## Quality assurance", ""] + rows + [""]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# reports.db
# ---------------------------------------------------------------------------

#: The library index. One row per report, keyed by the source's hash, plus a
#: meta table carrying the schema version. The tag lists are stored as JSON
#: arrays: SQLite has no list type and a second table would buy nothing a
#: library of a few thousand rows needs.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id                   TEXT PRIMARY KEY,
    report_id            TEXT,
    title                TEXT,
    authors              TEXT,
    year                 INTEGER,
    source               TEXT,
    doc_type             TEXT,
    tier                 TEXT,
    disciplines          TEXT,
    topics               TEXT,
    methods              TEXT,
    materials            TEXT,
    standards_referenced TEXT,
    status               TEXT,
    confidence           TEXT,
    summary              TEXT,
    key_takeaways        TEXT,
    key_parameters       TEXT,
    n_pages              INTEGER,
    n_investigations     INTEGER,
    n_lab_tests          INTEGER,
    n_qa                 INTEGER,
    workflow             TEXT,
    schema_version       TEXT,
    original_path        TEXT,
    record_path          TEXT,
    page_path            TEXT,
    summary_path         TEXT,
    diggs_path           TEXT,
    parent               TEXT,
    updated              TEXT
);
CREATE INDEX IF NOT EXISTS reports_report_id ON reports (report_id);
CREATE INDEX IF NOT EXISTS reports_year ON reports (year);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

#: Columns added after a library was first written, with the index each
#: wants. A new column is not a schema-version bump -- an old row simply has
#: NULL in it -- but an existing database will not grow one from ``CREATE
#: TABLE IF NOT EXISTS``, so each is added here if it is missing. The index
#: is NOT in :data:`_SCHEMA` for the same reason: an older library has no
#: such column and indexing one that is not there yet fails the open.
_ADDED_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("parent", "TEXT", "CREATE INDEX IF NOT EXISTS reports_parent "
                       "ON reports (parent)"),
)


def open_library(db_path: Any) -> sqlite3.Connection:
    """Open (and create) the library index at ``db_path``.

    A library written by an older version is migrated in place: the columns
    in :data:`_ADDED_COLUMNS` are added where they are missing, so a folder
    of reports read last month and a report read today land in one table.
    """
    path = str(db_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(_SCHEMA)
    have = {row["name"] for row in
            connection.execute("PRAGMA table_info(reports)")}
    for name, kind, index in _ADDED_COLUMNS:
        if name not in have:
            connection.execute(f"ALTER TABLE reports ADD COLUMN {name} {kind}")
        connection.execute(index)
    connection.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (DB_SCHEMA_VERSION,))
    connection.commit()
    return connection


def upsert_report(connection: sqlite3.Connection, record: ReportRecord,
                  key: str, paths: Optional[Dict[str, str]] = None,
                  source: str = "", parent: str = "") -> None:
    """Write this report's row, replacing the one it had if any.

    ``parent`` is the library key of the report this one was bound inside,
    for a record that carries a :class:`~report_ingest.model.ParentReport`;
    :func:`write_outputs` computes it with :func:`parent_key` off the same
    source object the key came from. It is ignored for an ordinary record,
    which has no parent to point at.
    """
    meta = front_matter(record, key, source)
    paths = paths or {}
    row = {
        "id": key,
        "report_id": meta["report_id"],
        "title": meta["title"],
        "authors": meta["authors"],
        "year": meta["year"] if isinstance(meta["year"], int) else None,
        "source": meta["source"],
        "doc_type": meta["doc_type"],
        "tier": meta["tier"],
        "disciplines": json.dumps(meta["disciplines"]),
        "topics": json.dumps(meta["topics"]),
        "methods": json.dumps(meta["methods"]),
        "materials": json.dumps(meta["materials"]),
        "standards_referenced": json.dumps(meta["standards_referenced"]),
        "status": meta["status"],
        "confidence": meta["confidence"],
        "summary": record.general.quickSummary or "",
        "key_takeaways": json.dumps(key_takeaways(record)),
        "key_parameters": json.dumps(key_parameters(record)),
        "n_pages": record.document.n_pages,
        "n_investigations": len(record.investigations),
        "n_lab_tests": len(record.lab_tests),
        "n_qa": len(record.qa),
        "workflow": record.document.workflow,
        "schema_version": record.schema_version,
        "original_path": source,
        "record_path": paths.get("record", ""),
        "page_path": paths.get("page", ""),
        "summary_path": paths.get("summary", ""),
        "diggs_path": paths.get("diggs", ""),
        # Empty for an ordinary report; for one bound inside another, the
        # library key of the report it is bound inside.
        "parent": parent if record.parent is not None else "",
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    columns = ", ".join(row)
    placeholders = ", ".join(f":{name}" for name in row)
    updates = ", ".join(f"{name} = excluded.{name}" for name in row
                        if name != "id")
    connection.execute(
        f"INSERT INTO reports ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(id) DO UPDATE SET {updates}", row)
    connection.commit()


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------

@dataclass
class WrittenOutputs:
    """Where everything went, and what the two DIGGS gates said."""

    record: str = ""
    summary: str = ""
    page: str = ""
    diggs: str = ""
    db: str = ""
    key: str = ""
    schema_ok: Optional[bool] = None
    roundtrip_ok: Optional[bool] = None
    schema_errors: List[str] = field(default_factory=list)
    roundtrip_diffs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "paths": {name: value for name, value in (
                ("record", self.record), ("summary", self.summary),
                ("page", self.page), ("diggs", self.diggs), ("db", self.db))
                if value},
            "diggs": {"schema_ok": self.schema_ok,
                      "roundtrip_ok": self.roundtrip_ok,
                      "schema_errors": self.schema_errors[:5],
                      "roundtrip_diffs": self.roundtrip_diffs[:5]},
        }

    @property
    def paths(self) -> Dict[str, str]:
        return {name: value for name, value in (
            ("record", self.record), ("summary", self.summary),
            ("page", self.page), ("diggs", self.diggs), ("db", self.db))
            if value}


def write_outputs(record: ReportRecord, out_dir: Any, *,
                  source: Any = None, db_path: Any = None,
                  write_diggs_file: bool = True,
                  parent_base: str = "") -> WrittenOutputs:
    """Write the record and its exports into ``out_dir``.

    ``source`` is the PDF this record came from, when there is one: it keys
    the library row and is recorded on the page. ``db_path`` defaults to
    ``reports.db`` beside the outputs; pass one path for a whole folder of
    reports and they land in one library.

    ``parent_base`` is the library key of the report this one was bound
    inside, for a record that carries a
    :class:`~report_ingest.model.ParentReport`. The graph passes it because
    it wrote that record too; see :func:`record_key` for when it matters.

    The DIGGS file is written and gated FIRST so that its two verdicts are in
    the record's QA before the record is written. Both verdicts are always
    recorded -- including "nothing checked this", which is what an absent
    pydiggs means and is not the same as a pass.
    """
    out = str(out_dir)
    os.makedirs(out, exist_ok=True)
    key = record_key(record, source, parent_base)
    source_text = str(source) if source else ""
    written = WrittenOutputs(key=key)

    if write_diggs_file:
        _write_diggs(record, out, written)

    path = os.path.join(out, RECORD_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record.model_dump(mode="json"), handle, indent=2)
    written.record = path

    path = os.path.join(out, SUMMARY_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(summary_markdown(record))
    written.summary = path

    path = os.path.join(out, PAGE_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(library_page(record, key, source_text))
    written.page = path

    db = str(db_path) if db_path else os.path.join(out, DB_NAME)
    connection = open_library(db)
    try:
        upsert_report(connection, record, key, written.paths, source_text,
                      parent=(parent_base or parent_key(record, source)
                              if record.parent is not None else ""))
    finally:
        connection.close()
    written.db = db
    return written


def _write_diggs(record: ReportRecord, out: str,
                 written: WrittenOutputs) -> None:
    """The DIGGS file, both gates, and the verdicts into the QA."""
    from report_ingest.diggs_writer import (
        diggs_roundtrip_gate, diggs_schema_gate, write_diggs,
    )

    # Writing the same record twice -- the graph resuming, a folder run
    # rewriting one report -- must leave ONE verdict in the QA, not two. The
    # verdicts are about the file being written now, so the previous run's
    # are dropped rather than kept beside them.
    record.qa[:] = [entry for entry in record.qa
                    if not entry.where.startswith("diggs")]

    if not record.investigations and not record.lab_tests:
        record.qa.append(QAEntry(
            kind="skipped", where="diggs",
            detail="no explorations and no laboratory tests were read, so no "
                   "DIGGS file was written"))
        return
    try:
        xml = write_diggs(record)
    except Exception as exc:                     # a record DIGGS cannot hold
        record.qa.append(QAEntry(
            kind="skipped", where="diggs",
            detail=f"the DIGGS file could not be written: "
                   f"{type(exc).__name__}: {exc}"))
        return

    path = os.path.join(out, DIGGS_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml)
    written.diggs = path

    ok, errors = diggs_schema_gate(xml)
    written.schema_ok, written.schema_errors = ok, list(errors)
    record.qa.append(QAEntry(
        kind="note" if ok else "out_of_range", where="diggs.schema",
        detail=("the DIGGS file is valid against the bundled 2.6 schema"
                if ok else "the DIGGS file did NOT pass the 2.6 schema"),
        values=list(errors)[:5]))

    ok, diffs = diggs_roundtrip_gate(xml, record.investigations,
                                     project=record.project,
                                     lab_tests=record.lab_tests)
    written.roundtrip_ok, written.roundtrip_diffs = ok, list(diffs)
    record.qa.append(QAEntry(
        kind="note" if ok else "conflict", where="diggs.roundtrip",
        detail=("the DIGGS file reads back equal to the record, value by "
                "value" if ok else
                "reading the DIGGS file back does not give the record's own "
                "values"),
        values=list(diffs)[:5]))
