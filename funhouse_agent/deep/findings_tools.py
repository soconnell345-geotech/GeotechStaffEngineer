"""The findings tools on the lean review agent (``GEOTECH_REVIEW_FINDINGS=1``).

``record_finding`` keeps one finding in the shared format
(:mod:`funhouse_agent.review_findings`) in ``findings.json`` in the
conversation's working folder, after checking each quote against the page it
cites; ``update_finding`` corrects one (its status, confidence, severity,
statement or citations - quotes are checked again when the citations change);
``list_findings`` reads them back (a review picked up in a later turn starts
here); ``findings_report`` renders them as a Word comment log (the same writer
``write_docx`` uses) or Markdown, and optionally as a marked-up copy of each
cited PDF (the same writer ``annotate_document`` uses). Rejected findings are
left out of the list and the report unless asked for.

A citation may name its document by the handle ``open_document`` returned
(``doc_...``): it is resolved through the app's open documents and recorded
under the name the document was opened as, so the ledger outlives the handle.

The working folder (``findings.json``, the marked-up copies) is bound when
the agent is built (``working_dir``), so another conversation in the same
process re-pointing the process-wide working folder cannot redirect this
one's findings.

The descriptions are written for a document reviewer and name no tool the
review page lacks.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from funhouse_agent import document_tools
from funhouse_agent.review_findings import (
    Finding, FindingsLedger, PlanlensPageText, ledger_path_in, sort_findings,
    to_markdown, to_markups, verify_quotes)

#: Who wrote a finding recorded through these tools.
SOURCE = "review agent"
#: The Markdown a report returns in the chat stays under this.
MARKDOWN_RESULT_CHARS = 7000
#: A list of findings stays under this (the general tool cap is 8,000).
LIST_RESULT_CHARS = 7500
#: The marked-up copy findings_report writes: ``<document>_findings.pdf`` -
#: never ``<document>_marked.pdf``, which is annotate_document's own default
#: and may hold the agent's (or the user's) hand-placed comments.
FINDINGS_PDF_SUFFIX = "_findings.pdf"

RECORD_DESCRIPTION = (
    "Record one review finding once you have checked it on the page, so it "
    "is kept for the rest of the review and for the deliverables. Give the "
    "statement as it would read in a comment log; severity (info, minor, "
    "major, critical); your confidence (high, medium, low); evidence - read "
    "(from the text), seen (by looking at the page) or computed; and "
    "citations: the document as you opened it (its name, or the handle "
    "open_document gave), the 0-based page, and the exact words the finding "
    "rests on as quote, or a bbox in PDF points for a spot on a drawing "
    "(sheet or printed_page too when the page shows one). Each quote is "
    "checked against its page and its review markups; a finding whose quotes "
    "are not there is kept at low confidence - re-check it. A finding that "
    "turns out wrong is corrected with update_finding (status='rejected' to "
    "withdraw it), not recorded again.")

UPDATE_DESCRIPTION = (
    "Correct a finding already recorded, by its id (F1, F2, ...): change its "
    "status (draft, confirmed, rejected - rejected withdraws it from the "
    "list and the deliverables), confidence, severity, statement or "
    "citations. New citations replace the old ones and their quotes are "
    "checked again. Only what you pass changes.")

LIST_DESCRIPTION = (
    "The findings recorded so far in this conversation - id, severity, "
    "statement, where, evidence, whether each quote was found on its page - "
    "optionally only one status (draft, confirmed, rejected), severity or "
    "discipline. Rejected findings are left out unless you ask for "
    "status='rejected'. Start here when you pick a review up again.")

REPORT_DESCRIPTION = (
    "Turn the recorded findings into the review's deliverables, most severe "
    "first: a Word comment log (format='docx'; path a bare file name) or the "
    "same table as Markdown (format='markdown'); marked_up_pdf=true also "
    "writes a copy of each cited PDF with one comment per citation "
    "(<document>_findings.pdf, redone from the findings each time). Rejected "
    "findings are left out.")


class CitationArg(BaseModel):
    """Where a finding is."""
    document: str = Field(description="The document as you opened it (its "
                                      "attachment name or path, or the "
                                      "handle open_document gave).")
    page: int = Field(description="0-based page, as the tools number pages.")
    quote: Optional[str] = Field(
        default=None, description="The exact words on that page the finding "
                                  "rests on.")
    bbox: Optional[List[float]] = Field(
        default=None, description="[x0, y0, x1, y1] in PDF points, top-left "
                                  "origin: a spot on a drawing.")
    sheet: Optional[str] = Field(default=None,
                                 description="The sheet number, if any.")
    printed_page: Optional[str] = Field(
        default=None, description="The page number printed on the page, if "
                                  "any.")


class RecordFindingArgs(BaseModel):
    statement: str = Field(description="The finding in a sentence or two, as "
                                       "it would read in a comment log.")
    severity: str = Field(description="info, minor, major or critical.")
    confidence: str = Field(description="high, medium or low.")
    evidence: str = Field(description="read (from the text), seen (by "
                                      "looking at the page) or computed.")
    citations: List[CitationArg] = Field(
        description="Where it is: at least one citation.")
    disciplines: Optional[List[str]] = Field(
        default=None, description="Optional, e.g. ['geotechnical'].")


class UpdateFindingArgs(BaseModel):
    id: str = Field(description="The finding's id, e.g. 'F2'.")
    status: Optional[str] = Field(
        default=None, description="draft, confirmed or rejected.")
    confidence: Optional[str] = Field(default=None,
                                      description="high, medium or low.")
    severity: Optional[str] = Field(
        default=None, description="info, minor, major or critical.")
    statement: Optional[str] = Field(default=None,
                                     description="The corrected statement.")
    citations: Optional[List[CitationArg]] = Field(
        default=None, description="Replacement citations (all of them); "
                                  "their quotes are checked again.")


def _plain(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return dict(obj)
    return obj


# ---------------------------------------------------------------------------
# Documents named by an open_document handle
# ---------------------------------------------------------------------------

#: What an open_document handle looks like (``doc_`` + a content hash).
_HANDLE = re.compile(r"^doc_[0-9a-f]{6,}$")


def is_handle(document: Any) -> bool:
    """Whether ``document`` is an ``open_document`` handle."""
    return bool(_HANDLE.match(str(document or "").strip()))


def _open_entry(handle: str):
    """``(name, entry)`` for a handle the app's document tools have open
    (``entry`` is planlens' own record when it can be reached, else
    ``None``); ``None`` when the handle is not open."""
    if not is_handle(handle):
        return None
    name = document_tools._document_name(handle)
    if name is None:
        return None
    entry = None
    try:
        entries = getattr(document_tools._toolkit(), "_entries", None)
        if entries is not None:
            entry = entries.get(handle)
    except Exception:  # noqa: BLE001 - the name alone still helps
        entry = None
    return name, entry


def handle_source(handle: str, attachments: Optional[Dict[str, bytes]]
                  ) -> Optional[Tuple[str, Any]]:
    """``(label, bytes or path)`` for the document behind an open handle:
    ``label`` is what it was opened as (an attachment key or a path) where
    that is known, else its file name; the source is that label resolved as
    the document tools resolve a source, else the open document's own file
    or bytes. ``None`` when the handle is not open."""
    got = _open_entry(str(handle or "").strip())
    if got is None:
        return None
    name, entry = got
    label = str(getattr(entry, "source", "") or "") or name
    for candidate in dict.fromkeys((label, name)):
        try:
            return label, document_tools.resolve_document_source(
                candidate, attachments)
        except Exception:  # noqa: BLE001 - try the open document itself
            continue
    if entry is not None:
        path = getattr(entry, "path", None)
        if path and os.path.isfile(path):
            return label, path
        try:
            with entry.lock:
                return label, entry.doc.tobytes()
        except Exception:  # noqa: BLE001 - closed meanwhile
            return None
    return None


def _sources_for(finding: Finding, attachments: Dict[str, bytes]
                 ) -> Dict[str, Any]:
    """Each cited document, resolved the way the document tools resolve a
    source (attachments, then real paths, then the working folder). A
    citation naming an open ``doc_...`` handle is renamed to the document
    it stands for, so the ledger does not depend on the handle."""
    out: Dict[str, Any] = {}
    for c in finding.citations:
        if is_handle(c.document):
            got = handle_source(c.document, attachments)
            if got is not None:
                c.document, out[got[0]] = got[0], got[1]
                continue
        if c.document in out:
            continue
        try:
            out[c.document] = document_tools.resolve_document_source(
                c.document, attachments)
        except Exception:  # noqa: BLE001 - an unknown document fails its quote
            if c.document in (attachments or {}):
                out[c.document] = attachments[c.document]
            elif os.path.isfile(c.document):
                out[c.document] = c.document
    return out


def _verify(finding: Finding, attachments: Dict[str, bytes]) -> None:
    """Resolve the finding's documents and check its quotes (in place)."""
    reader = PlanlensPageText(_sources_for(finding, attachments))
    try:
        verify_quotes([finding], reader)
    finally:
        reader.close()


def _compact(f: Finding) -> Dict[str, Any]:
    row: Dict[str, Any] = {"id": f.id, "severity": f.severity,
                           "confidence": f.confidence, "status": f.status,
                           "statement": f.statement, "where": f.where(),
                           "evidence": f.evidence}
    if f.disciplines:
        row["disciplines"] = f.disciplines
    if f.quote_verified is not None:
        row["quote_verified"] = f.quote_verified
    return row


def _fit_list(rows: List[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"count": len(rows), "findings": rows}
    if len(json.dumps(out)) <= limit:
        return out
    for row in rows:
        if len(row["statement"]) > 200:
            row["statement"] = row["statement"][:200] + " ..."
    kept = list(rows)
    while kept and len(json.dumps({"count": len(rows), "findings": kept})) \
            > limit - 200:
        kept.pop()
    return {"count": len(rows), "findings": kept,
            "note": f"{len(rows) - len(kept)} more not shown; filter by "
                    f"severity, status or discipline"}


def _ledger_error(exc: OSError) -> str:
    return json.dumps({"error": f"the findings file could not be read or "
                                f"written just now: {type(exc).__name__}: "
                                f"{exc}",
                       "hint": "try again in a moment; nothing was lost"})


def _check_result(finding: Finding, out: Dict[str, Any]) -> Dict[str, Any]:
    if finding.quote_note:
        out["quote_check"] = finding.quote_note
    if finding.quote_verified is False:
        out["note"] = ("a quote was not found on the page it cites: look at "
                       "that page again before you report this"
                       if finding.evidence != "seen" else
                       "a quote was not found in the page's text: if you read "
                       "it by looking, zoom on it once more to be sure")
    return out


def make_findings_tools(attachments: Optional[Dict[str, bytes]] = None,
                        save_fn: Optional[Callable] = None,
                        markup_author: Optional[str] = None,
                        ledger: Optional[FindingsLedger] = None,
                        source: str = SOURCE,
                        working_dir: Optional[str] = None) -> list:
    """``record_finding``, ``update_finding``, ``list_findings`` and
    ``findings_report`` bound to this agent's uploads (the live dict the host
    mutates), its save function and its markup author.

    ``working_dir`` is the conversation's working folder, bound when the
    agent is built: ``findings.json`` and the marked-up copies go there.
    Without it (and without a ``ledger``) the ledger follows the working
    folder the host sets per call (``default_output_dir()``)."""
    from langchain_core.tools import StructuredTool

    attachments = {} if attachments is None else attachments
    if ledger is None:
        ledger = FindingsLedger(ledger_path_in(working_dir) if working_dir
                                else None)

    def record_finding(statement: str, severity: str, confidence: str,
                       evidence: str, citations: List[Any],
                       disciplines: Optional[List[str]] = None) -> str:
        try:
            finding = Finding(
                statement=statement, severity=severity, confidence=confidence,
                evidence=evidence, disciplines=list(disciplines or []),
                citations=[_plain(c) for c in (citations or [])],
                source=source)
        except ValueError as exc:
            return json.dumps({"error": str(exc),
                               "hint": "fix that and record it again"})
        _verify(finding, attachments)
        try:
            finding = ledger.add(finding)
        except OSError as exc:
            return json.dumps({"error": f"could not save the finding: "
                                        f"{type(exc).__name__}: {exc}"})
        out: Dict[str, Any] = {"recorded": finding.id,
                               "confidence": finding.confidence,
                               "quote_verified": finding.quote_verified}
        return json.dumps(_check_result(finding, out))

    def update_finding(id: str, status: Optional[str] = None,
                       confidence: Optional[str] = None,
                       severity: Optional[str] = None,
                       statement: Optional[str] = None,
                       citations: Optional[List[Any]] = None) -> str:
        fields: Dict[str, Any] = {
            k: v for k, v in (("status", status), ("confidence", confidence),
                              ("severity", severity),
                              ("statement", statement))
            if v is not None and str(v).strip() != ""}
        if not fields and citations is None:
            return json.dumps({"error": "nothing to change",
                               "hint": "pass status, confidence, severity, "
                                       "statement or citations"})
        try:
            current = ledger.get(id)
        except OSError as exc:
            return _ledger_error(exc)
        if current is None:
            return json.dumps({"error": f"no finding {id!r}",
                               "hint": "list_findings shows the ids"})
        checked: Optional[Finding] = None
        try:
            if citations is not None:
                data = current.to_dict()
                data.update(fields)
                data.update(citations=[_plain(c) for c in citations],
                            quote_verified=None, quote_note=None)
                checked = Finding.from_dict(data)
                _verify(checked, attachments)
                fields.update(
                    citations=[c.to_dict() for c in checked.citations],
                    quote_verified=checked.quote_verified,
                    quote_note=checked.quote_note,
                    confidence=checked.confidence)
            updated = ledger.update(current.id, **fields)
        except ValueError as exc:
            return json.dumps({"error": str(exc),
                               "hint": "fix that and update it again"})
        except KeyError:
            return json.dumps({"error": f"no finding {id!r}",
                               "hint": "list_findings shows the ids"})
        except OSError as exc:
            return _ledger_error(exc)
        out: Dict[str, Any] = {"updated": updated.id,
                               "status": updated.status,
                               "severity": updated.severity,
                               "confidence": updated.confidence,
                               "quote_verified": updated.quote_verified}
        if checked is not None:
            _check_result(updated, out)
        if updated.status == "rejected":
            out["note"] = ("withdrawn: left out of list_findings and the "
                           "deliverables")
        return json.dumps(out)

    def list_findings(status: str = "", severity: str = "",
                      discipline: str = "") -> str:
        try:
            found = ledger.list(status=status or None,
                                severity=severity or None,
                                discipline=discipline or None)
        except OSError as exc:
            return _ledger_error(exc)
        hidden = 0
        if not status:
            hidden = sum(1 for f in found if f.status == "rejected")
            found = [f for f in found if f.status != "rejected"]
        out = _fit_list([_compact(f) for f in found], LIST_RESULT_CHARS)
        if hidden:
            out["rejected_not_shown"] = hidden
        return json.dumps(out)

    def findings_report(format: str = "docx", path: str = "",
                        title: str = "", marked_up_pdf: bool = False) -> str:
        try:
            findings = sort_findings(f for f in ledger.load()
                                     if f.status != "rejected")
        except OSError as exc:
            return _ledger_error(exc)
        if not findings:
            return json.dumps({"error": "no findings recorded yet",
                               "hint": "record them with record_finding "
                                       "first"})
        md = to_markdown(findings)
        out: Dict[str, Any] = {"count": len(findings)}
        fmt = str(format or "docx").strip().lower()
        if fmt in ("md", "markdown"):
            if len(md) > MARKDOWN_RESULT_CHARS:
                out["markdown"] = md[:MARKDOWN_RESULT_CHARS] + "\n..."
                out["note"] = ("the log is longer than shown: write it to "
                               "Word (format='docx') for all of it")
            else:
                out["markdown"] = md
        else:
            name = path or "review_comments.docx"
            if working_dir and not os.path.isabs(os.path.expanduser(name)):
                name = os.path.join(working_dir, os.path.basename(name))
            out["docx"] = _write_docx(md, name, title or "Review comments",
                                      save_fn)
        if marked_up_pdf:
            out["marked_up"] = _marked_up(findings, attachments,
                                          markup_author, working_dir)
        return json.dumps(out, default=str)

    return [
        StructuredTool.from_function(record_finding, name="record_finding",
                                     description=RECORD_DESCRIPTION,
                                     args_schema=RecordFindingArgs),
        StructuredTool.from_function(update_finding, name="update_finding",
                                     description=UPDATE_DESCRIPTION,
                                     args_schema=UpdateFindingArgs),
        StructuredTool.from_function(list_findings, name="list_findings",
                                     description=LIST_DESCRIPTION),
        StructuredTool.from_function(findings_report, name="findings_report",
                                     description=REPORT_DESCRIPTION),
    ]


def _write_docx(markdown: str, path: str, title: str,
                save_fn: Optional[Callable]) -> Dict[str, Any]:
    """The comment log through ``write_docx``'s own writer (same working
    folder, verification and download card)."""
    from funhouse_agent import vision_tools
    from funhouse_agent.deep.tools import _with_saved_note
    if not vision_tools.docx_available():
        return {"error": "Word output is not available here",
                "hint": "ask for format='markdown' instead"}
    writer = save_fn or vision_tools._default_save_fn
    raw = vision_tools._dispatch_write_docx(
        {"path": path, "markdown": markdown, "title": title}, writer)
    try:
        return json.loads(_with_saved_note(raw, writer))
    except (TypeError, ValueError):
        return {"result": raw}


def findings_pdf_path(document_name: Optional[str],
                      working_dir: Optional[str] = None) -> str:
    """Where the marked-up copy of one cited document goes:
    ``<document>_findings.pdf`` in the bound working folder, else in the
    working folder resolved now."""
    stem = os.path.splitext(os.path.basename(str(document_name or "")))[0] \
        or "document"
    name = f"{stem}{FINDINGS_PDF_SUFFIX}"
    if working_dir:
        return os.path.abspath(os.path.join(working_dir, name))
    return document_tools.markup_output_path(name)


def _marked_up(findings: List[Finding], attachments: Dict[str, bytes],
               author: Optional[str], working_dir: Optional[str] = None
               ) -> List[Dict[str, Any]]:
    """One marked-up copy per cited document, through the app's own
    ``annotate_document`` path, redone from the findings each time."""
    if not document_tools.has_tool("annotate_document"):
        return [{"error": "marked-up PDFs need a newer planlens "
                          "(annotate_document)"}]
    docs: List[str] = []
    for f in findings:
        for c in f.citations:
            if c.document not in docs:
                docs.append(c.document)
    results = []
    who = author or document_tools.markup_author()
    for doc in docs:
        entry: Dict[str, Any] = {"document": doc}
        opened: Dict[str, Any] = {}
        try:
            if is_handle(doc) and document_tools._document_name(doc):
                # a citation left under an open handle: use it as it is
                handle = doc
            else:
                opened = json.loads(document_tools.dispatch_document_tool(
                    "open_document", {"source": doc}, attachments=attachments,
                    max_chars=6000))
                handle = opened.get("handle")
            if not handle:
                entry["error"] = opened.get("error") or "could not open it"
                results.append(entry)
                continue
            out_path = findings_pdf_path(
                document_tools._document_name(handle) or doc, working_dir)
            marks = to_markups(findings, doc)

            def annotate(markups, append):
                return json.loads(document_tools.dispatch_document_tool(
                    "annotate_document", {
                        "handle": handle, "output_path": out_path,
                        "markups": markups, "author": who, "append": append},
                    attachments=attachments, max_chars=6000))

            written = annotate(marks, False)
            if written.get("error"):
                entry["error"] = written["error"]
                results.append(entry)
                continue
            n_written = int(written.get("n_written") or 0)
            skipped = list(written.get("skipped") or [])
            # planlens' own count: its list of skipped rows is cut to fit.
            n_skipped = int(written.get("n_skipped", len(skipped)) or 0)
            # A quote planlens cannot anchor (a page with no text layer, or
            # words not on it) still belongs on its page: as a note there.
            retry = [s for s in skipped
                     if isinstance(s.get("index"), int)
                     and 0 <= s["index"] < len(marks)
                     and marks[s["index"]]["kind"] != "note"]
            if retry:
                again = annotate([_as_note(marks[s["index"]]) for s in retry],
                                 True)
                if not again.get("error"):
                    placed = int(again.get("n_written") or 0)
                    n_written += placed
                    n_skipped -= placed
                    entry["placed_as_notes"] = placed
                    skipped = ([s for s in skipped if s not in retry]
                               + list(again.get("skipped") or []))
        except Exception as exc:  # noqa: BLE001 - reported per document
            entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
            continue
        entry.update(output_path=written.get("output_path"),
                     n_written=n_written, n_skipped=max(0, n_skipped))
        if skipped:
            entry["skipped"] = skipped[:10]
        if n_skipped > len(entry.get("skipped") or []):
            entry["skipped_not_listed"] = n_skipped - len(
                entry.get("skipped") or [])
        results.append(entry)
    return results


def _as_note(mark: Dict[str, Any]) -> Dict[str, Any]:
    """A markup that could not be anchored, as a note in the page's corner."""
    from funhouse_agent.review_findings import NOTE_POINT
    return {"kind": "note", "page": mark["page"], "point": list(NOTE_POINT),
            "comment": mark["comment"] + " (the words it cites could not be "
                                         "placed on this page)"}


__all__ = ["make_findings_tools", "RECORD_DESCRIPTION", "UPDATE_DESCRIPTION",
           "LIST_DESCRIPTION", "REPORT_DESCRIPTION", "RecordFindingArgs",
           "UpdateFindingArgs", "CitationArg", "FINDINGS_PDF_SUFFIX",
           "findings_pdf_path", "handle_source", "is_handle"]
