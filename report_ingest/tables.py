"""The report tables: one ingest's results as flat, keyed rows (BACKLOG RI-209).

An ingest leaves a folder per report: the record, the summary, the library
page, the DIGGS file, the page labels, and on some runs the owner's question
list. A pipeline cannot read that as tables. This module turns a report's
SAVED outputs into an export a pipeline reads one step per table, under one
folder (``<root>/report_tables/v1/``)::

    schema.json                       the tables, their columns, types, keys
    report/<report_key>.jsonl         one row: the report as it stands now
    report_page/<report_key>.jsonl    one row per page: what it is
    report_answer/<report_key>.jsonl  one row per standard question
    report_file/<report_key>.jsonl    one row per file delivered below
    ingest_run/<report_key>.<run_id>.jsonl   one row per ingest (history)
    diggs/<report_key>.diggs.xml
    record/<report_key>.record.json
    summary/<report_key>.summary.md
    questions/<report_key>.questions.xlsx    when the question list ran

The tables are defined ONCE, in :data:`TABLES` (name, columns with a type
and a line saying what each holds, primary key, links); ``schema.json``, the
rows' checks and ``docs/REPORT_TABLES.md`` all come from that definition.
Every row carries every column of its table, nulls written out; the types
are the eight in :data:`TYPES`, and a structured value is a JSON string in a
``*_json`` column. Pages are PDF pages counted from 1, as a viewer shows
them; the record counts from 0.

ONE CODE PATH. The export is built from the files on disk, never from what
an ingest holds in memory: :func:`export_report` over one report's folder
(its bound-in reports with it, each a report of its own), and
:func:`export_folder` over a folder of reports already ingested (``python
-m report_ingest.tables <run_root>``), which costs no model call. At the end
of every ingest the graph calls :func:`after_ingest`, which first saves the
run's facts the folder does not otherwise keep (the run id and time, the
source file's name, size and SHA-256, the package versions) as a small file
beside the record (:data:`RUN_NAME`), then exports. A folder older than this
module has no such file: its key comes from the library page, its source
facts are null and its time is the record file's.

THE SETTINGS. :data:`ENV_ROOT` (``GSE_REPORT_TABLES``) unset, the export is
ON and its root is the folder holding the ingest's ``reports.db`` (a folder
run's or the cluster stage's shared library), else the app's shared report
library (:data:`LIBRARY_ENV`) for a report ingested into it, else the
report's own folder; set to a folder it names another root; ``0`` or ``off`` turns it off, and
then nothing at all is written. ``ingest_report(report_tables=...)`` wins
over it. :data:`ENV_DATASET` (``GSE_REPORT_TABLES_DATASET``) names a Foundry
workspace dataset alias: on a Foundry deployment, each report's export
files then go to that dataset in ONE save (:func:`send_to_dataset`), under the
application's top folder. Unset, or not on Foundry, nothing is sent and
nothing of Foundry's is imported.

NEVER IN THE WAY. The export never changes the record and never raises
into an ingest: a failure is logged and comes back in the
:class:`ExportOutcome`, which the ingest also keeps in :data:`RUN_NAME`
(:func:`export_outcome` reads it back).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, get_args, get_origin)

__all__ = [
    "EXPORT_VERSION", "EXPORT_DIR", "ENV_ROOT", "ENV_DATASET", "RUN_NAME",
    "LIBRARY_ENV", "TYPES", "Column", "Link", "Table", "Document", "TABLES",
    "TABLE",
    "DOCUMENTS", "ExportOutcome", "schema", "columns_markdown", "make_row",
    "check_row",
    "export_root", "export_dir", "after_ingest", "export_report",
    "export_folder", "export_outcome", "send_to_dataset", "dataset_alias",
    "source_facts", "main",
]

_log = logging.getLogger(__name__)

#: The export's version: its folder name and every row's ``export_version``.
#: A column that changes meaning is a new version; a new column is not.
EXPORT_VERSION = "v1"
#: The folder under the root that holds every version of the export.
EXPORT_DIR = "report_tables"

#: Where the export goes: unset, beside the ingest's ``reports.db``; a folder
#: names another root; ``0`` / ``off`` turns it off.
ENV_ROOT = "GSE_REPORT_TABLES"
#: The Foundry workspace dataset alias each report's export is sent to.
ENV_DATASET = "GSE_REPORT_TABLES_DATASET"
#: The app's shared report library (``funhouse_agent.deep.tools``'
#: ``REPORT_LIBRARY_ENV``): a report ingested into it exports into ONE
#: ``report_tables/`` at the library's top, beside every other report.
LIBRARY_ENV = "GEOTECH_REPORT_LIBRARY"

#: The run's facts beside each record: written by :func:`after_ingest`.
RUN_NAME = "report.run.json"
#: The run file's own format, inside it.
RUN_FORMAT = 1

#: The distributions whose versions an ingest row records.
APP_DIST = "geotech-staff-engineer"
CORE_DIST = "aec-agent-core"
PLANLENS_DIST = "planlens"

_OFF = ("0", "off", "false", "no")
_ON = ("1", "on", "true", "yes")

#: The record's own file names (``report_ingest.writers``), spelt here so
#: importing this module costs nothing.
_RECORD = "report.record.json"
_SUMMARY = "report.summary.md"
_PAGE = "report.page.md"
_DIGGS = "report.diggs.xml"
_SHEET = "report.questions.xlsx"
_ANSWERS = "report.questions.json"
_CALLS = "question_list.calls.json"

#: The library page's key line, as :mod:`report_ingest.library` reads it.
_FRONT_ID = re.compile(r'^id:\s*"?([0-9a-f]{6,64})"?\s*$', re.MULTILINE)
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# the tables, defined once
# ---------------------------------------------------------------------------

#: The column types, and what each means in a file.
TYPES: Dict[str, str] = {
    "string": "text",
    "integer": "a whole number (may exceed 32 bits: map it to a long)",
    "double": "a number with a fraction",
    "boolean": "true or false",
    "timestamp": "a moment in UTC, ISO 8601 with a Z: 2026-01-02T03:04:05Z",
    "date": "a calendar date, YYYY-MM-DD",
    "array<string>": "a list of text",
    "array<integer>": "a list of whole numbers",
}


@dataclass(frozen=True)
class Column:
    """One column: its name, its type (a key of :data:`TYPES`), whether it
    may be null, and what it holds, in a line."""

    name: str
    type: str
    description: str
    nullable: bool = True


@dataclass(frozen=True)
class Link:
    """Columns of one table that name a row of another (a foreign key)."""

    columns: Tuple[str, ...]
    table: str
    references: Tuple[str, ...]


@dataclass(frozen=True)
class Table:
    """One table: one folder of JSON Lines files with the same columns."""

    name: str
    description: str
    primary_key: Tuple[str, ...]
    columns: Tuple[Column, ...]
    links: Tuple[Link, ...] = ()
    #: How a file in the table's folder is named.
    file_pattern: str = "<report_key>.jsonl"

    @property
    def column_names(self) -> Tuple[str, ...]:
        return tuple(c.name for c in self.columns)


@dataclass(frozen=True)
class Document:
    """One file the export carries as it is: what the report folder calls
    it, and the folder and suffix it is delivered under."""

    kind: str
    source: str
    suffix: str
    media_type: str
    description: str


def _c(name: str, type_: str, description: str,
       nullable: bool = True) -> Column:
    return Column(name, type_, description, nullable)


_VERSION = _c("export_version", "string",
              "The export's version (v1).", False)
_KEY = _c("report_key", "string",
          "The report's key: the library key the ingest computes, the same "
          "as the library page's id and the reports.db row. For a report, "
          "it comes from its source file's size and first megabyte; for a "
          "report bound inside another one, from its parent's key and its "
          "bound_id.", False)
_REPORT_ID = _c("report_id", "string",
                "The caller's handle for the report (a file stem or a "
                "corpus id); null when none was given.")
_SOURCE_NAME = _c("source_name", "string",
                  "The source file's name, never a path; for a bound-in "
                  "report, its parent's file. Null when the ingest was given "
                  "the file's bytes or a document with no path, and for a "
                  "folder ingested before this export.")
_SOURCE_BYTES = _c("source_bytes", "integer",
                   "The source file's size in bytes; null as source_name.")
_SOURCE_SHA = _c("source_sha256", "string",
                 "The SHA-256 of the whole source file, in hex, so the PDF "
                 "a dataset holds can be matched to its report; null as "
                 "source_name.")
_N_PAGES = _c("n_pages", "integer",
              "Pages in the report: the whole file for a report, its own "
              "range for a bound-in report.", False)
_DOC_TYPE = _c("document_type", "string",
               "The owner's documentType answer (geotechnical report, "
               "recommendation letter, ...); null when the record does not "
               "state it.")
_WORKFLOW = _c("workflow", "string",
               "The workflow triage chose: standard, appendix_only, "
               "partial, multi_document, scanned or needs_person (nothing "
               "read: no page rows); null when no triage ran.")
_COUNTS: Tuple[Column, ...] = (
    _c("n_investigations", "integer",
       "Explorations (borings, test pits, soundings) read into the record.",
       False),
    _c("n_lab_tests", "integer", "Laboratory tests read into the record.",
       False),
    _c("n_calculations", "integer",
       "Calculation printouts read into the record.", False),
    _c("n_answers", "integer",
       "Rows of report_answer for the report.", False),
    _c("n_answers_answered", "integer",
       "Of those, the rows whose status is answered.", False),
    _c("n_qa_open", "integer",
       "QA entries that need a person (rank open: a disagreement nothing "
       "settled).", False),
    _c("n_qa_changed", "integer",
       "QA entries for a value the record changed or dropped by its own "
       "rule (rank changed).", False),
    _c("n_qa_note", "integer", "QA entries that are notes (rank note).",
       False),
)
_DIGGS_COLUMNS: Tuple[Column, ...] = (
    _c("diggs_written", "boolean",
       "A DIGGS 2.6 file was written by the ingest (a report_file row of "
       "kind diggs then exists). None is written for a record with no "
       "explorations and no laboratory tests, or one the DIGGS writer "
       "could not hold (the record's QA says why).", False),
    _c("diggs_schema_valid", "boolean",
       "true: the DIGGS file passed the DIGGS 2.6 schema; false: it did "
       "not; null: no file, or the check could not run where the ingest "
       "ran."),
    _c("diggs_roundtrip_ok", "boolean",
       "true: reading the DIGGS file back gives the record's own values; "
       "false: it does not; null: no file."),
)
_INGESTED = _c("ingested_at", "timestamp",
               "When the ingest wrote its outputs. For a folder ingested "
               "before this export, the record file's modified time.",
               False)

LABELS_TEXT = ("narrative, figure, plan, profile, boring_log, test_pit_log, "
               "cpt_log, dcp_log, lab_test, field_test, calculation, "
               "appended_report, photos, divider, cover, letter, toc, other")

TABLES: Tuple[Table, ...] = (
    Table(
        name="report",
        description="One row per report, as it stands after its latest "
                    "ingest. A report bound inside another one is a report "
                    "of its own, with parent_report_key set.",
        primary_key=("report_key",),
        columns=(
            _VERSION, _KEY,
            _c("parent_report_key", "string",
               "For a report bound inside another one, the parent's "
               "report_key; null for a report of its own."),
            _c("bound_id", "string",
               "For a bound-in report, its handle inside the parent "
               "(bound1, bound2, in page order); null otherwise. It is a "
               "count, not a name: when the ingest finds the bound-in "
               "reports differently, they are renumbered and their "
               "report_keys change."),
            _c("bound_report_keys", "array<string>",
               "The report_keys of the reports bound inside this one, as of "
               "its latest ingest; empty when none. A report row whose "
               "parent_report_key does not list it any more is left over "
               "from an earlier ingest.", False),
            _REPORT_ID, _SOURCE_NAME, _SOURCE_BYTES, _SOURCE_SHA, _N_PAGES,
            _c("title", "string",
               "What the library page calls the report: its project name "
               "and document type.", False),
            _DOC_TYPE,
            _c("project_name", "string",
               "The project's name as printed (the narrative's answer, else "
               "the record's project block); null when not stated."),
            _c("project_number", "string",
               "The project or job number as printed; null when not "
               "stated."),
            _c("firm", "string",
               "The firm that wrote the report (geotechnicalEngineerFirm); "
               "null when not stated."),
            _c("post_name", "string",
               "The post the site belongs to, as the report names it "
               "(postName); null when not stated."),
            _c("report_date", "date",
               "The report's date (reportDateISO); null when the record "
               "has no full date."),
            _WORKFLOW, *_COUNTS, *_DIGGS_COLUMNS,
            _c("latest_run_id", "string",
               "The ingest that wrote this row: an ingest_run row's run_id.",
               False),
            _INGESTED,
        ),
        links=(
            Link(("parent_report_key",), "report", ("report_key",)),
            Link(("report_key", "latest_run_id"), "ingest_run",
                 ("report_key", "run_id")),
        ),
    ),
    Table(
        name="report_page",
        description="One row per page of a report: what the page is, and "
                    "who said so. A report triage sent to a person "
                    "(workflow needs_person) has none. A bound-in report's "
                    "pages are numbered in its parent's file, and the "
                    "parent's own rows for those pages name it.",
        primary_key=("report_key", "pdf_page"),
        columns=(
            _VERSION, _KEY,
            _c("pdf_page", "integer",
               "The page as a PDF viewer numbers it, from 1 at the file's "
               "first page. For a bound-in report, the page of its "
               "parent's file (file_report_key).", False),
            _c("file_report_key", "string",
               "The report_key of the file pdf_page counts in: the report's "
               "own for a report, its parent's for a bound-in report.",
               False),
            _c("printed_page", "string",
               "The page number printed on the page, as printed; null when "
               "none was read."),
            _c("label", "string",
               f"What the page is, one of: {LABELS_TEXT}.", False),
            _c("label_confidence", "double",
               "The highest confidence (0 to 1) among the voters that gave "
               "this label; null where none of them stated one (the section "
               "labeller, which settles every page on the default path, "
               "states none)."),
            _c("agreed", "boolean",
               "Every voter with a view gave the same label.", False),
            _c("settled_by", "string",
               "vote; review where the label review moved the page; parent "
               "where a bound-in report keeps the label the whole file's "
               "pass gave the page.", False),
            _c("policy", "string",
               "The policy that combined the voters: structural, trust, "
               "confidence or rules."),
            _c("label_rules", "string",
               "The rules' label (planlens), when they gave one."),
            _c("label_vision", "string",
               "The vision voter's label, when it ran (model-first labels "
               "off)."),
            _c("label_ranges", "string",
               "The section labeller's label, when it ran (model-first "
               "labels, the default)."),
            _c("template_family", "string",
               "The printed log form the template voter recognised, when it "
               "recognised one."),
            _c("reason", "string",
               "Why the label is what it is, in a few words (a policy and a "
               "label, never the page's own words), from the run's "
               "labels.json; null when not saved."),
            _c("bound_report_key", "string",
               "On a report's own page: the report bound inside it whose "
               "pages include this one; null otherwise."),
            _c("title", "string",
               "The page's title or main heading as printed, from its saved "
               "description (model-first labels only); null otherwise."),
            _c("kind", "string",
               "What kind of page it is, in plain words, from its saved "
               "description; null otherwise."),
            _c("summary", "string",
               "One sentence on what the page shows, from its saved "
               "description; null otherwise."),
            _c("exploration", "string",
               "The boring, test pit, probe or well the page logs, as "
               "printed, from its saved description; null when it logs "
               "none or no description was saved."),
        ),
        links=(
            Link(("report_key",), "report", ("report_key",)),
            Link(("file_report_key",), "report", ("report_key",)),
            Link(("bound_report_key",), "report", ("report_key",)),
        ),
    ),
    Table(
        name="report_answer",
        description="One row per standard question asked of a report: the "
                    "owner's two standing schemas (general, "
                    "natural_hazards), every field every time, and the "
                    "owner's question list (question_list) when it ran. "
                    "The caller's own free-text questions are not here.",
        primary_key=("report_key", "question_set", "question_key"),
        columns=(
            _VERSION, _KEY,
            _c("question_set", "string",
               "general or natural_hazards (the owner's two schemas), or "
               "question_list (the owner's question list).", False),
            _c("question_key", "string",
               "The owner's field name (general, natural_hazards) or the "
               "list's Key (question_list).", False),
            _c("position", "integer",
               "The question's place in its set, from 1, in the owner's "
               "order.", False),
            _c("question_label", "string",
               "The list's Output; null for the two schemas, whose key is "
               "the owner's own field name."),
            _c("category", "string",
               "The list's Category; null for the two schemas."),
            _c("data_type", "string",
               "The answer's type in the question list's words: string, "
               "boolean, integer, date, list of strings, list of records, "
               "dictionary, integer or list of integers. For the two "
               "schemas, from the field's type; null for a list answer read "
               "back from an older answer sheet."),
            _c("status", "string",
               "answered; not said (the report is silent); does not apply "
               "(the question has no bearing; question list only). Null "
               "where no answer came back at all (a question-list call that "
               "failed; the note says so)."),
            _c("value_text", "string",
               "The answer as text: the schema's answer as printed (a list "
               "joined with '; '), or the question list's answer as its "
               "sheet shows it; 'none' for an empty list."),
            _c("value_number", "double",
               "The answer as a number, when it is one."),
            _c("value_boolean", "boolean",
               "The answer as true or false: a boolean answer, or for the "
               "schemas' yes/no questions yes as true and no as false "
               "(mixed and unclear stay null)."),
            _c("value_date", "date",
               "The answer as a date, when it is a full date (the schema's "
               "reportDate gives its reportDateISO)."),
            _c("value_list", "array<string>",
               "The answer as a list of text, when it is a list of plain "
               "values; null otherwise."),
            _c("value_json", "string",
               "The answer's typed value as JSON. Where the schema keeps a "
               "typed twin beside the printed answer (reportDateISO, "
               "siteClassNormalized, asceSevenVersionNormalized, "
               "bearingCapacityValues, strataList, the yes/no answers' "
               "verdicts), the twin's value; the twins' own page "
               "references are in pdf_pages and quotes, not here. Null "
               "when there is no answer."),
            _c("pdf_pages", "array<integer>",
               "The pages the answer was read from, from 1 at the file's "
               "first page (a bound-in report's: its parent's file).",
               False),
            _c("quotes", "array<string>",
               "The page's own words for the answer, where saved (the "
               "schemas' citations; the question list keeps none).", False),
            _c("note", "string",
               "What the answer comes with: the question list's note."),
            _c("source", "string",
               "Who answered, the question list's From: code, report text, "
               "both: agree, both: disagree, both: compare by eye, "
               "screening; null for the two schemas."),
            _c("code_value_text", "string",
               "On a question-list row answered by both code and the report "
               "text: code's reading, as text."),
            _c("qa_kinds", "array<string>",
               "The kinds of the record's QA entries about this answer (QA "
               "entries whose where is '<question_set>.<question_key>').",
               False),
            _c("question_list", "string",
               "For a question-list row: the list file's name and when it "
               "was last modified; null otherwise."),
            _c("prompt_version", "string",
               "The version of what asked it: the question list's (ql-...), "
               "or the model-first narrative reader's (mf-...), where "
               "saved."),
        ),
        links=(Link(("report_key",), "report", ("report_key",)),),
    ),
    Table(
        name="report_file",
        description="One row per file delivered for a report in the "
                    "export's document folders. Trust these rows, not a "
                    "folder listing: a dataset keeps a file an earlier "
                    "ingest sent even when the latest ingest wrote none.",
        primary_key=("report_key", "kind"),
        columns=(
            _VERSION, _KEY,
            _c("kind", "string",
               "diggs, record, summary or questions.", False),
            _c("path", "string",
               "The file, relative to the export's v1 folder, with forward "
               "slashes.", False),
            _c("bytes", "integer", "The file's size in bytes.", False),
            _c("sha256", "string", "The file's SHA-256, in hex.", False),
            _c("media_type", "string", "The file's media type.", False),
        ),
        links=(Link(("report_key",), "report", ("report_key",)),),
    ),
    Table(
        name="ingest_run",
        description="One row per ingest of a report, never overwritten: "
                    "the history, which is the ingest ledger BACKLOG RI-11 "
                    "asked for. Each ingest adds one small file.",
        primary_key=("report_key", "run_id"),
        file_pattern="<report_key>.<run_id>.jsonl",
        columns=(
            _VERSION, _KEY,
            _c("run_id", "string",
               "The ingest: its UTC time and a short random suffix. A "
               "bound-in report shares its parent's. For a folder ingested "
               "before this export, the record file's time and part of its "
               "hash.", False),
            _INGESTED, _REPORT_ID, _SOURCE_NAME, _SOURCE_BYTES, _SOURCE_SHA,
            _N_PAGES, _DOC_TYPE, _WORKFLOW, *_COUNTS, *_DIGGS_COLUMNS,
            _c("record_schema_version", "string",
               "The record's schema version.", False),
            _c("app_version", "string",
               "The geotech-staff-engineer version that ran the ingest; "
               "null when not installed as a package, and for a folder "
               "ingested before this export."),
            _c("core_version", "string",
               "The aec-agent-core version, likewise."),
            _c("planlens_version", "string",
               "The planlens version that read the file; null when not "
               "recorded."),
            _c("prompt_versions", "array<string>",
               "The versions of what the models were asked, where saved: "
               "the model-first readers' (mf-...), the page describer's, "
               "the question list's (ql-...).", False),
            _c("models", "array<string>",
               "The models the run's saved readings name (its item files, "
               "page descriptions and question-list calls).", False),
            _c("model_calls", "integer",
               "The model calls the ingest made (for a report, its bound-in "
               "reports' included; for a bound-in report, its own). A step "
               "taken from the folder made none.", False),
            _c("input_tokens", "integer", "Their input tokens, likewise.",
               False),
            _c("output_tokens", "integer", "Their output tokens, likewise.",
               False),
            _c("dollars", "double",
               "Their cost in dollars, likewise (0 where pricing is off).",
               False),
            _c("seconds", "double", "The ingest's wall clock, likewise.",
               False),
            _c("resumed", "boolean",
               "The ingest took some steps from results an earlier run of "
               "the same report saved.", False),
        ),
        links=(Link(("report_key",), "report", ("report_key",)),),
    ),
)

TABLE: Dict[str, Table] = {t.name: t for t in TABLES}

DOCUMENTS: Tuple[Document, ...] = (
    Document("diggs", _DIGGS, ".diggs.xml", "application/xml",
             "The DIGGS 2.6 file, when the ingest wrote one."),
    Document("record", _RECORD, ".record.json", "application/json",
             "The whole record (page numbers in it count from 0)."),
    Document("summary", _SUMMARY, ".summary.md", "text/markdown",
             "The summary page."),
    Document("questions", _SHEET, ".questions.xlsx",
             "application/vnd.openxmlformats-officedocument."
             "spreadsheetml.sheet",
             "The owner's question list's answer sheet, when it ran."),
)


def schema() -> Dict[str, Any]:
    """``schema.json``: the tables and the documents, from the definitions."""
    return {
        "export_version": EXPORT_VERSION,
        "pages": "Every page number is a PDF page counted from 1 at the "
                 "file's first page, as a viewer shows it. A bound-in "
                 "report's pages are its parent's file's.",
        "types": dict(TYPES),
        "tables": [{
            "name": t.name, "description": t.description,
            "files": f"{t.name}/{t.file_pattern}",
            "primary_key": list(t.primary_key),
            "links": [{"columns": list(link.columns),
                       "references": {"table": link.table,
                                      "columns": list(link.references)}}
                      for link in t.links],
            "columns": [{"name": c.name, "type": c.type,
                         "nullable": c.nullable,
                         "description": c.description} for c in t.columns],
        } for t in TABLES],
        "documents": [{"kind": d.kind,
                       "files": f"{d.kind}/<report_key>{d.suffix}",
                       "media_type": d.media_type,
                       "description": d.description} for d in DOCUMENTS],
    }


def columns_markdown() -> str:
    """Each table's columns as the Markdown ``docs/REPORT_TABLES.md``
    carries between its two ``columns`` markers (a test holds the two the
    same): ``python -m report_ingest.tables --columns`` prints it."""
    out: List[str] = []
    for table in TABLES:
        out += [f"### `{table.name}`", "", table.description, "",
                f"Files: `{table.name}/{table.file_pattern}`. Primary key: "
                + ", ".join(f"`{c}`" for c in table.primary_key) + ".", "",
                "| Column | Type | Null? | What it holds |",
                "|---|---|---|---|"]
        out += [f"| `{c.name}` | {c.type} | {'yes' if c.nullable else 'no'} "
                f"| {c.description} |" for c in table.columns]
        out.append("")
    return "\n".join(out)


def _valid_date(text: Any) -> bool:
    if not isinstance(text, str) or not _DATE.match(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _fits(type_: str, value: Any) -> bool:
    if type_ == "string":
        return isinstance(value, str)
    if type_ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_ == "double":
        return isinstance(value, float)
    if type_ == "boolean":
        return isinstance(value, bool)
    if type_ == "timestamp":
        return isinstance(value, str) and bool(_TIMESTAMP.match(value))
    if type_ == "date":
        return _valid_date(value)
    if type_ == "array<string>":
        return isinstance(value, list) and all(isinstance(v, str)
                                               for v in value)
    if type_ == "array<integer>":
        return isinstance(value, list) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value)
    raise ValueError(f"unknown column type {type_!r}")


def check_row(table: Table, row: Mapping[str, Any]) -> None:
    """Raise ValueError unless ``row`` has exactly ``table``'s columns, in
    order, each null (where the column may be) or of its type."""
    if tuple(row) != table.column_names:
        raise ValueError(f"{table.name}: the row's columns {list(row)} are "
                         f"not the table's {list(table.column_names)}")
    for column in table.columns:
        value = row[column.name]
        if value is None:
            if not column.nullable:
                raise ValueError(f"{table.name}.{column.name} may not be "
                                 f"null")
            continue
        if not _fits(column.type, value):
            raise ValueError(f"{table.name}.{column.name}: {value!r} is not "
                             f"a {column.type}")


def make_row(table: Table, values: Mapping[str, Any]) -> Dict[str, Any]:
    """A row of ``table`` from ``values``: every column in order, a column
    not given written as null, a whole number in a double column as a
    float; checked (:func:`check_row`). A name the table does not have is
    refused."""
    unknown = set(values) - set(table.column_names)
    if unknown:
        raise ValueError(f"{table.name} has no column {sorted(unknown)}")
    row: Dict[str, Any] = {}
    for column in table.columns:
        value = values.get(column.name)
        if column.type == "double" and isinstance(value, int) \
                and not isinstance(value, bool):
            value = float(value)
        row[column.name] = value
    row["export_version"] = EXPORT_VERSION
    check_row(table, row)
    return row


# ---------------------------------------------------------------------------
# the settings
# ---------------------------------------------------------------------------

def export_root(setting: Any = None, out_dir: Any = ".",
                db_path: Any = None) -> Optional[str]:
    """The folder that holds ``report_tables/``, or None when the export is
    off. ``setting`` is ``ingest_report(report_tables=...)``: None follows
    :data:`ENV_ROOT`; False, ``0`` or ``off`` is off; True, ``1`` or ``on``
    is the default root; anything else names the root. The default is the
    folder holding ``db_path`` (a run's shared library); else, when
    :data:`LIBRARY_ENV` names the app's shared report library and
    ``out_dir`` is inside it, that library's folder (one ``report_tables/``
    for every report ingested into the library); else ``out_dir``."""
    if setting is None:
        setting = os.environ.get(ENV_ROOT, "")
    if isinstance(setting, bool):
        if not setting:
            return None
        setting = ""
    text = str(setting).strip()
    if text.lower() in _OFF:
        return None
    if text and text.lower() not in _ON:
        return os.path.expanduser(text)
    if db_path:
        return os.path.dirname(os.path.abspath(str(db_path)))
    return _library_holding(out_dir) or str(out_dir)


def _library_holding(out_dir: Any) -> str:
    """The app's shared report library's folder (:data:`LIBRARY_ENV`, ``~``
    expanded, made absolute, as the app reads it) when ``out_dir`` is
    inside it; "" otherwise."""
    value = str(os.environ.get(LIBRARY_ENV) or "").strip()
    if not value:
        return ""
    library = os.path.abspath(os.path.expanduser(value))
    here = os.path.abspath(str(out_dir))
    try:
        inside = os.path.commonpath([library, here]) == library
    except ValueError:                        # another drive
        return ""
    return library if inside and here != library else ""


def export_dir(root: Any) -> str:
    """The export's own folder under ``root``: ``<root>/report_tables/v1``."""
    return os.path.join(str(root), EXPORT_DIR, EXPORT_VERSION)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _write_bytes(path: str, data: bytes) -> None:
    """``data`` to ``path`` atomically: a temporary file in the same folder
    (its name ends ``.part``, so no table's pattern takes it), replaced."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=".", suffix=".part", dir=folder)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temp, path)
    except BaseException:
        try:
            os.remove(temp)
        except OSError:
            pass
        raise


def _copy(source: str, path: str) -> None:
    """``source`` copied to ``path`` atomically."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=".", suffix=".part", dir=folder)
    os.close(handle)
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, path)
    except BaseException:
        try:
            os.remove(temp)
        except OSError:
            pass
        raise


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False,
                              separators=(",", ":")) + "\n"
                   for row in rows).encode("utf-8")


def _text(value: Any) -> Optional[str]:
    """A string column's value: None for None or an empty string."""
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _version(dist: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:                                  # noqa: BLE001
        return None


def source_facts(source: Any) -> Dict[str, Any]:
    """The source file's name, size and SHA-256 (``source_name``,
    ``source_bytes``, ``source_sha256``): from a readable path, from bytes
    (no name), from an open document's path; all None otherwise."""
    facts: Dict[str, Any] = {"source_name": None, "source_bytes": None,
                             "source_sha256": None}
    if isinstance(source, (bytes, bytearray)):
        facts["source_bytes"] = len(source)
        facts["source_sha256"] = hashlib.sha256(bytes(source)).hexdigest()
        return facts
    path = getattr(source, "path", None) if not isinstance(
        source, (str, os.PathLike)) else source
    if path and os.path.isfile(str(path)):
        facts["source_name"] = os.path.basename(str(path))
        facts["source_bytes"] = os.path.getsize(str(path))
        facts["source_sha256"] = _sha256_file(str(path))
    return facts


# ---------------------------------------------------------------------------
# one report's folder, read back
# ---------------------------------------------------------------------------

def _page_key(folder: str) -> str:
    """The key the library page carries, or ""."""
    try:
        with open(os.path.join(folder, _PAGE), encoding="utf-8") as handle:
            head = handle.read(2000)
    except OSError:
        return ""
    found = _FRONT_ID.search(head)
    return found.group(1) if found else ""


def _key_in(folder: str, record: Any = None, parent_key: str = "") -> str:
    """A report folder's key: its run file's, else its library page's, else
    recomputed from the record (as the library does)."""
    run = _read_json(os.path.join(folder, RUN_NAME)) or {}
    key = str(run.get("report_key") or "") or _page_key(folder)
    if key or record is None:
        return key
    from report_ingest.writers import record_key
    return record_key(record, None, parent_key)


@dataclass
class _Report:
    """What one report's folder holds, as the export reads it."""

    folder: str
    record: Any
    key: str
    run: Dict[str, Any]
    parent: Optional["_Report"] = None
    parent_key: Optional[str] = None
    children: List["_Report"] = field(default_factory=list)

    @property
    def file_key(self) -> str:
        return self.parent_key or self.key

    @property
    def file_folder(self) -> str:
        """The folder of the report whose file the pages count in."""
        return self.parent.folder if self.parent is not None \
            else self.folder


def _load_record(folder: str) -> Any:
    from report_ingest.model import ReportRecord

    blob = _read_json(os.path.join(folder, _RECORD))
    if blob is None:
        raise ValueError(f"no readable {_RECORD}")
    return ReportRecord.model_validate(blob)


def _load(folder: str, parent: Optional[_Report] = None) -> _Report:
    record = _load_record(folder)
    run = _read_json(os.path.join(folder, RUN_NAME)) or {}
    parent_key: Optional[str] = None
    if record.parent is not None:
        if parent is not None:
            parent_key = parent.key
        else:
            up = os.path.dirname(os.path.normpath(os.path.join(
                folder, record.parent.record_path
                or os.path.join(os.pardir, os.pardir, _RECORD))))
            parent_key = str(run.get("parent_report_key") or "") \
                or _key_in(up) or None
    key = _key_in(folder, record, parent_key or "")
    return _Report(folder=folder, record=record, key=key, run=run,
                   parent=parent, parent_key=parent_key)


def _with_children(top: _Report) -> _Report:
    for row in top.record.bound_documents:
        if not row.read or not row.record_path:
            continue
        child_folder = os.path.dirname(os.path.normpath(
            os.path.join(top.folder, row.record_path)))
        if os.path.isfile(os.path.join(child_folder, _RECORD)):
            top.children.append(_load(child_folder, parent=top))
    return top


def _run_id(report: _Report) -> str:
    got = str(report.run.get("run_id") or "")
    if got:
        return got
    path = os.path.join(report.folder, _RECORD)
    moment = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    return f"{_stamp(moment)}-{_sha256_file(path)[:6]}"


def _ingested_at(report: _Report) -> str:
    got = str(report.run.get("ingested_at") or "")
    if got and _TIMESTAMP.match(got):
        return got
    path = os.path.join(report.folder, _RECORD)
    return _utc(datetime.fromtimestamp(os.path.getmtime(path), timezone.utc))


def _questions(report: _Report) -> Tuple[Optional[Dict[str, Any]],
                                          Optional[str]]:
    """The question list's answers (``report.questions.json``, else the
    sheet read back) and the sheet to deliver, when they belong to the
    latest ingest: a run file that says the list did not run hides an
    earlier run's files."""
    if report.run and not report.run.get("question_list", False):
        return None, None
    sheet = os.path.join(report.folder, _SHEET)
    sheet = sheet if os.path.isfile(sheet) else None
    blob = _read_json(os.path.join(report.folder, _ANSWERS))
    if isinstance(blob, dict) and isinstance(blob.get("answers"), list):
        return blob, sheet
    if sheet is not None:
        return _sheet_answers(sheet), sheet
    return None, None


def _pages_from(text: Any) -> List[int]:
    """"3-5, 9" (the sheet's PDF pages cell) as [3, 4, 5, 9]."""
    out: List[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        low, _sep, high = part.partition("-")
        try:
            a, b = int(low), int(high or low)
        except ValueError:
            continue
        out.extend(range(a, b + 1))
    return sorted(set(out))


def _sheet_answers(path: str) -> Optional[Dict[str, Any]]:
    """An answer sheet written before ``report.questions.json``, read back:
    the answers tab (Output, Category, Key, Answer, PDF pages, From, Note)
    and the Run tab's list, date and prompt version. It holds no status,
    data type or typed value: the status is the Answer cell's (N/A does not
    apply, empty not said, else answered)."""
    try:
        import openpyxl
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:                           # noqa: BLE001
        _log.warning("report tables: the answer sheet %s could not be "
                     "read: %s", os.path.basename(path), exc)
        return None
    try:
        rows = list(book["Answers"].iter_rows(values_only=True)) \
            if "Answers" in book.sheetnames else []
        info = {str(r[0]): r[1] for r in book["Run"].iter_rows(
            values_only=True) if r and r[0]} \
            if "Run" in book.sheetnames else {}
    finally:
        book.close()
    if not rows:
        return None
    head = [str(c or "") for c in rows[0]]

    def cell(row: Sequence[Any], name: str) -> Any:
        return row[head.index(name)] if name in head and \
            head.index(name) < len(row) else None

    answers = []
    for row in rows[1:]:
        key = cell(row, "Key")
        if not key:
            continue
        shown = cell(row, "Answer")
        shown = "" if shown is None else str(shown)
        status = ("does not apply" if shown == "N/A"
                  else "not said" if not shown else "answered")
        answers.append({
            "key": str(key), "output": cell(row, "Output"),
            "category": cell(row, "Category"), "data_type": None,
            "value": None, "status": status,
            "pdf_pages": _pages_from(cell(row, "PDF pages")),
            "note": cell(row, "Note") or "", "source": cell(row, "From"),
            "text": shown if status == "answered" else "",
            "code_value_text": ""})
    return {"format": 0, "answers": answers,
            "run": {"question_list": str(info.get("Question list") or ""),
                    "list_modified": str(info.get("List modified") or ""),
                    "prompt_version": str(info.get("Prompt version") or ""),
                    "calls": []}}


def _descriptions(report: _Report) -> Dict[int, Dict[str, Any]]:
    """The saved page descriptions (model first), by 0-based page, from
    the folder of the report whose file the pages are in."""
    folder = os.path.join(report.file_folder, "descriptions")
    out: Dict[int, Dict[str, Any]] = {}
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not (name.startswith("p") and name.endswith(".json")):
            continue
        blob = _read_json(os.path.join(folder, name))
        if isinstance(blob, dict) and isinstance(blob.get("page"), int):
            out[blob["page"]] = blob
    return out


def _reasons(report: _Report) -> Dict[int, str]:
    blob = _read_json(os.path.join(report.folder, "labels.json")) or {}
    out: Dict[int, str] = {}
    for row in blob.get("pages") or []:
        if isinstance(row, dict) and isinstance(row.get("page"), int) \
                and row.get("why"):
            out[row["page"]] = str(row["why"])
    return out


def _saved_runs(report: _Report, questions: Optional[Dict[str, Any]]
                ) -> Tuple[List[str], List[str], Optional[str]]:
    """The models and prompt versions the report's saved readings name
    (its item files, its page descriptions, the question list's calls),
    and the model-first narrative reader's prompt version."""
    models: set = set()
    prompts: set = set()
    narrative: Optional[str] = None
    items = os.path.join(report.folder, "items")
    names = sorted(os.listdir(items)) if os.path.isdir(items) else []
    for name in names:
        if not name.endswith(".json"):
            continue
        blob = _read_json(os.path.join(items, name))
        if not isinstance(blob, dict):
            continue
        if name == _CALLS:
            if questions is None:
                continue
            models.update(str(v.get("model")) for v in blob.values()
                          if isinstance(v, dict) and v.get("model"))
            continue
        if isinstance(blob.get("model"), str) and blob["model"]:
            models.add(blob["model"])
        version = blob.get("prompt_version")
        if isinstance(version, str) and version:
            prompts.add(version)
            if name.startswith("mf-narrative"):
                narrative = version
    if report.parent is None:
        for blob in _descriptions(report).values():
            if blob.get("model"):
                models.add(str(blob["model"]))
            if blob.get("prompt"):
                prompts.add(str(blob["prompt"]))
    if questions is not None:
        run = questions.get("run") or {}
        if run.get("prompt_version"):
            prompts.add(str(run["prompt_version"]))
        models.update(str(c.get("model")) for c in run.get("calls") or []
                      if isinstance(c, dict) and c.get("model"))
    return sorted(models), sorted(prompts), narrative


def _diggs(report: _Report) -> Tuple[bool, Optional[bool], Optional[bool]]:
    """Written, schema valid, read back equal: off the record's own QA
    (the writer puts both verdicts there before it writes the record), and
    the file. A DIGGS file left in the folder by an earlier ingest, when
    the latest wrote none (QA ``skipped`` at ``diggs``), is not this
    report's."""
    qa = report.record.qa
    skipped = any(e.where == "diggs" and e.kind == "skipped" for e in qa)
    written = not skipped and os.path.isfile(
        os.path.join(report.folder, _DIGGS))
    schema_ok: Optional[bool] = None
    roundtrip: Optional[bool] = None
    if written:
        for entry in qa:
            if entry.where == "diggs.schema" and entry.kind != "not_run":
                schema_ok = entry.kind == "note"
            elif entry.where == "diggs.roundtrip":
                roundtrip = entry.kind == "note"
    return written, schema_ok, roundtrip


# ---------------------------------------------------------------------------
# the rows
# ---------------------------------------------------------------------------

#: The schemas' typed twins: the field a twin sits beside, and the twin.
#: Each is named beside its field in :mod:`report_ingest.model`.
TWINS: Dict[Tuple[str, str], str] = {
    ("general", "bearingCapacity"): "bearingCapacityValues",
    ("general", "strata"): "strataList",
    ("general", "outsideProject"): "outsideProjectAnswer",
    ("natural_hazards", "siteClass"): "siteClassNormalized",
    ("natural_hazards", "asceSevenVersion"): "asceSevenVersionNormalized",
    ("natural_hazards", "reportDate"): "reportDateISO",
    ("natural_hazards", "geophysicalTestingMention"):
        "geophysicalTestingAnswer",
    ("natural_hazards", "soilCorrosion"): "soilCorrosionAnswer",
    ("natural_hazards", "siteResponseMention"): "siteResponseAnswer",
    ("natural_hazards", "hazardAnalysisMention"): "hazardAnalysisAnswer",
}

_VERDICTS = {"yes": True, "no": False}


def _data_type(annotation: Any) -> str:
    """A schema field's type in the question list's words."""
    args = [a for a in get_args(annotation) if a is not type(None)]
    if get_origin(annotation) is not None and len(args) == 1 \
            and type(None) in get_args(annotation):
        annotation = args[0]
    if get_origin(annotation) in (list, List):
        return "list of strings"
    if annotation is int:
        return "integer"
    if annotation is bool:
        return "boolean"
    return "string"


def _strip(value: Any) -> Any:
    """A twin as plain data with its page references (``citation``,
    ``prov``) taken out: they go in the row's pages and quotes."""
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items()
                if k not in ("citation", "prov")}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _citations(value: Any) -> List[Any]:
    """Every citation inside a twin, wherever it sits."""
    found: List[Any] = []
    if isinstance(value, list):
        for item in value:
            found += _citations(item)
    elif hasattr(value, "citation"):
        found += list(getattr(value, "citation") or [])
    return found


def _shown(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(str(v) for v in value) if value else "none"
    return str(value)


def _schema_rows(report: _Report, qa_where: Dict[str, List[str]],
                 narrative_version: Optional[str]) -> List[Dict[str, Any]]:
    from report_ingest.model import GENERAL_FIELDS, NATURAL_HAZARD_FIELDS

    rows: List[Dict[str, Any]] = []
    for section, names in (("general", GENERAL_FIELDS),
                           ("natural_hazards", NATURAL_HAZARD_FIELDS)):
        facts = getattr(report.record, section)
        fields = type(facts).model_fields
        for position, name in enumerate(names, start=1):
            value = getattr(facts, name, None)
            twin_name = TWINS.get((section, name))
            twin = getattr(facts, twin_name, None) if twin_name else None
            if twin in ([], ""):
                twin = None
            cited = list(facts.citations.get(name) or []) + _citations(twin)
            values: Dict[str, Any] = {
                "report_key": report.key, "question_set": section,
                "question_key": name, "position": position,
                "data_type": _data_type(fields[name].annotation),
                "status": "answered" if value is not None else "not said",
                "pdf_pages": sorted({int(c.page) + 1 for c in cited}),
                "quotes": list(dict.fromkeys(
                    c.quote for c in cited if c.quote)),
                "qa_kinds": qa_where.get(f"{section}.{name}", []),
                "prompt_version": narrative_version,
            }
            if value is not None:
                values["value_text"] = _shown(value)
                if isinstance(value, list):
                    values["value_list"] = [str(v) for v in value]
                elif isinstance(value, (int, float)) and \
                        not isinstance(value, bool):
                    values["value_number"] = float(value)
                typed: Any = value
                if twin is not None:
                    typed = twin.model_dump(mode="json") if hasattr(
                        twin, "model_dump") else [
                        t.model_dump(mode="json") for t in twin] if \
                        isinstance(twin, list) else twin
                    if hasattr(twin, "answer"):          # a Mention
                        typed = twin.answer
                    typed = _strip(typed)
                verdict = typed if isinstance(typed, str) else None
                if verdict in _VERDICTS and (
                        hasattr(twin, "answer") or name == "outsideProject"):
                    values["value_boolean"] = _VERDICTS[verdict]
                if twin_name == "reportDateISO" and _valid_date(twin):
                    values["value_date"] = twin
                values["value_json"] = json.dumps(typed, ensure_ascii=False)
            rows.append(make_row(TABLE["report_answer"], values))
    return rows


def _list_rows(report: _Report, questions: Dict[str, Any],
               qa_where: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    run = questions.get("run") or {}
    name = str(run.get("question_list") or "")
    modified = str(run.get("list_modified") or "")
    listed = (f"{name} (modified {modified})" if name and modified
              else name or None)
    rows: List[Dict[str, Any]] = []
    for position, answer in enumerate(questions.get("answers") or [],
                                      start=1):
        key = str(answer.get("key") or "")
        if not key:
            continue
        value = answer.get("value")
        status = str(answer.get("status") or "") or None
        values: Dict[str, Any] = {
            "report_key": report.key, "question_set": "question_list",
            "question_key": key, "position": position,
            "question_label": _text(answer.get("output")),
            "category": _text(answer.get("category")),
            "data_type": _text(answer.get("data_type")),
            "status": status,
            "value_text": _text(answer.get("text")),
            "pdf_pages": sorted({int(p) for p in answer.get("pdf_pages")
                                 or [] if isinstance(p, int)}),
            "quotes": [],
            "note": _text(answer.get("note")),
            "source": _text(answer.get("source")),
            "code_value_text": _text(answer.get("code_value_text")),
            "qa_kinds": qa_where.get(f"question_list.{key}", []),
            "question_list": listed,
            "prompt_version": _text(run.get("prompt_version")),
        }
        if isinstance(value, bool):
            values["value_boolean"] = value
        elif isinstance(value, (int, float)):
            values["value_number"] = float(value)
        elif isinstance(value, str) and answer.get("data_type") == "date" \
                and _valid_date(value):
            values["value_date"] = value
        elif isinstance(value, list) and all(
                not isinstance(v, (dict, list)) for v in value):
            values["value_list"] = [str(v) for v in value]
        if value is not None:
            values["value_json"] = json.dumps(value, ensure_ascii=False,
                                              default=str)
        rows.append(make_row(TABLE["report_answer"], values))
    return rows


def _page_rows(report: _Report) -> List[Dict[str, Any]]:
    record = report.record
    descriptions = _descriptions(report)
    reasons = _reasons(report)
    printed = record.document.printed_pages or {}
    ranges = [(c.record.parent.first_page, c.record.parent.last_page, c.key)
              for c in report.children if c.record.parent is not None]
    rows: List[Dict[str, Any]] = []
    for page in sorted(record.page_labels, key=lambda p: p.page):
        votes: Dict[str, str] = {}
        family = None
        for vote in page.voters:
            if vote.voter == "template":
                family = family or _text(vote.family)
            elif vote.label and vote.voter not in votes:
                votes[vote.voter] = vote.label
        described = (descriptions.get(page.page) or {}).get(
            "description") or {}
        inside = [key for first, last, key in ranges
                  if first <= page.page <= last]
        rows.append(make_row(TABLE["report_page"], {
            "report_key": report.key, "pdf_page": page.page + 1,
            "file_report_key": report.file_key,
            "printed_page": _text(printed.get(page.page)),
            "label": page.label,
            "label_confidence": float(page.confidence)
            if page.confidence else None,
            "agreed": bool(page.agreed), "settled_by": page.settled_by,
            "policy": _text(page.policy),
            "label_rules": votes.get("rules"),
            "label_vision": votes.get("vision"),
            "label_ranges": votes.get("ranges"),
            "template_family": family,
            "reason": reasons.get(page.page),
            "bound_report_key": inside[0] if inside else None,
            "title": _text(described.get("title")),
            "kind": _text(described.get("kind")),
            "summary": _text(described.get("summary")),
            "exploration": _text(described.get("exploration")),
        }))
    return rows


@dataclass
class _Built:
    """One report's rows and the files it delivers (kind -> local file)."""

    key: str
    rows: Dict[str, List[Dict[str, Any]]]
    run_id: str
    documents: Dict[str, str]


def _build(report: _Report) -> _Built:
    record = report.record
    qa_where: Dict[str, List[str]] = {}
    for entry in record.qa:
        kinds = qa_where.setdefault(entry.where, [])
        if entry.kind not in kinds:
            kinds.append(entry.kind)
    questions, sheet = _questions(report)
    models, prompts, narrative = _saved_runs(report, questions)
    answers = _schema_rows(report, qa_where, narrative)
    if questions is not None:
        answers += _list_rows(report, questions, qa_where)
    written, schema_ok, roundtrip = _diggs(report)
    ranks = record.qa_by_rank()
    run_id = _run_id(report)
    general, hazards = record.general, record.natural_hazards
    source = {name: report.run.get(name) for name in (
        "source_name", "source_bytes", "source_sha256")}
    shared = {
        "report_key": report.key,
        "report_id": _text(record.document.report_id),
        **source,
        "n_pages": int(record.document.n_pages),
        "document_type": _text(general.documentType),
        "workflow": _text(record.document.workflow),
        "n_investigations": len(record.investigations),
        "n_lab_tests": len(record.lab_tests),
        "n_calculations": len(record.calculations),
        "n_answers": len(answers),
        "n_answers_answered": sum(1 for r in answers
                                  if r["status"] == "answered"),
        "n_qa_open": ranks.get("open", 0),
        "n_qa_changed": ranks.get("changed", 0),
        "n_qa_note": ranks.get("note", 0),
        "diggs_written": written, "diggs_schema_valid": schema_ok,
        "diggs_roundtrip_ok": roundtrip,
        "ingested_at": _ingested_at(report),
    }
    from report_ingest.writers import title_of

    report_row = make_row(TABLE["report"], {
        **shared,
        "parent_report_key": report.parent_key,
        "bound_id": _text(record.parent.bound_id)
        if record.parent is not None else None,
        "bound_report_keys": [c.key for c in report.children],
        "title": title_of(record),
        "project_name": _text(general.projectName or record.project.name),
        "project_number": _text(general.projectNumber
                                or record.project.number),
        "firm": _text(general.geotechnicalEngineerFirm),
        "post_name": _text(general.postName),
        "report_date": hazards.reportDateISO
        if _valid_date(hazards.reportDateISO) else None,
        "latest_run_id": run_id,
    })
    document = record.document
    run_row = make_row(TABLE["ingest_run"], {
        **shared,
        "run_id": run_id,
        "record_schema_version": str(record.schema_version),
        "app_version": _text(report.run.get("app_version")),
        "core_version": _text(report.run.get("core_version")),
        "planlens_version": _text(report.run.get("planlens_version")
                                  or document.planlens_version),
        "prompt_versions": prompts, "models": models,
        "model_calls": int(document.model_calls),
        "input_tokens": int(document.input_tokens),
        "output_tokens": int(document.output_tokens),
        "dollars": float(document.dollars),
        "seconds": float(document.seconds),
        "resumed": document.resumed is not None,
    })
    documents = {"record": os.path.join(report.folder, _RECORD)}
    if written:
        documents["diggs"] = os.path.join(report.folder, _DIGGS)
    if os.path.isfile(os.path.join(report.folder, _SUMMARY)):
        documents["summary"] = os.path.join(report.folder, _SUMMARY)
    if sheet is not None:
        documents["questions"] = sheet
    file_rows = []
    for doc in DOCUMENTS:
        local = documents.get(doc.kind)
        if local is None:
            continue
        file_rows.append(make_row(TABLE["report_file"], {
            "report_key": report.key, "kind": doc.kind,
            "path": f"{doc.kind}/{report.key}{doc.suffix}",
            "bytes": os.path.getsize(local), "sha256": _sha256_file(local),
            "media_type": doc.media_type}))
    return _Built(key=report.key, run_id=run_id, documents=documents, rows={
        "report": [report_row], "report_page": _page_rows(report),
        "report_answer": answers, "report_file": file_rows,
        "ingest_run": [run_row]})


# ---------------------------------------------------------------------------
# writing, and the outcome
# ---------------------------------------------------------------------------

@dataclass
class ExportOutcome:
    """What one export did: ``written``, ``off``, or ``error`` (nothing
    of that report was written; ``detail`` says why). ``files`` are the
    files written, relative to the export's folder (``folder``, local);
    ``delivery`` says what the Foundry dataset got: ``sent N file(s)``,
    ``not set`` (no dataset named), ``not a Foundry deployment``,
    ``busy: ...`` or ``failed: ...``; empty when no delivery was asked."""

    status: str
    folder: str = ""
    report_key: str = ""
    keys: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    rows: Dict[str, int] = field(default_factory=dict)
    delivery: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "folder": self.folder,
                "report_key": self.report_key, "keys": list(self.keys),
                "files": len(self.files), "rows": dict(self.rows),
                "delivery": self.delivery, "detail": self.detail}


def _doc_path(v1: str, kind: str, key: str) -> str:
    doc = next(d for d in DOCUMENTS if d.kind == kind)
    return os.path.join(v1, kind, f"{key}{doc.suffix}")


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _old_children(v1: str, key: str) -> List[str]:
    """The bound-in reports the export's report row named before."""
    try:
        with open(os.path.join(v1, "report", f"{key}.jsonl"),
                  encoding="utf-8") as handle:
            row = json.loads(handle.readline() or "{}")
    except (OSError, ValueError):
        return []
    return [str(k) for k in row.get("bound_report_keys") or []]


def _write_group(built: Sequence[_Built], v1: str) -> List[str]:
    """Each report's files, replacing what the export held for its key;
    a document the report no longer has, and a bound-in report the parent
    no longer lists, removed (their ingest_run files stay)."""
    written: List[str] = []
    top = built[0]
    for gone in set(_old_children(v1, top.key)) - {b.key for b in built}:
        for table in ("report", "report_page", "report_answer",
                      "report_file"):
            _remove(os.path.join(v1, table, f"{gone}.jsonl"))
        for doc in DOCUMENTS:
            _remove(_doc_path(v1, doc.kind, gone))
    for one in built:
        for doc in DOCUMENTS:
            target = _doc_path(v1, doc.kind, one.key)
            if doc.kind in one.documents:
                _copy(one.documents[doc.kind], target)
                written.append(target)
            else:
                _remove(target)
        for table in TABLES:
            name = (f"{one.key}.{one.run_id}.jsonl"
                    if table.name == "ingest_run" else f"{one.key}.jsonl")
            target = os.path.join(v1, table.name, name)
            _write_bytes(target, _jsonl(one.rows[table.name]))
            written.append(target)
    target = os.path.join(v1, "schema.json")
    _write_bytes(target, (json.dumps(schema(), indent=1, ensure_ascii=False)
                          + "\n").encode("utf-8"))
    written.append(target)
    return written


def _relative(v1: str, paths: Iterable[str]) -> List[str]:
    return [os.path.relpath(p, v1).replace(os.sep, "/") for p in paths]


def export_report(report_dir: Any, root: Any, *, deliver: bool = False,
                  file_manager: Any = None) -> ExportOutcome:
    """Export one report's folder, and the reports bound inside it, into
    ``<root>/report_tables/v1/``; with ``deliver``, send its files (and
    ``schema.json``, and every ``ingest_run`` file of its keys) to the
    Foundry dataset :data:`ENV_DATASET` names (:func:`send_to_dataset`).
    Never raises: a failure is the outcome's ``error``."""
    folder = str(report_dir)
    v1 = export_dir(root)
    try:
        top = _with_children(_load(folder))
        group = [top] + top.children
        built = [_build(report) for report in group]
        paths = _write_group(built, v1)
    except Exception as exc:                           # noqa: BLE001
        _log.warning("report tables: %s was not exported: %s: %s",
                     os.path.basename(os.path.normpath(folder)),
                     type(exc).__name__, exc)
        return ExportOutcome(status="error", folder=v1,
                             detail=f"{type(exc).__name__}: {exc}")
    rows = {t.name: sum(len(b.rows[t.name]) for b in built) for t in TABLES}
    outcome = ExportOutcome(status="written", folder=v1, report_key=top.key,
                            keys=[b.key for b in built],
                            files=_relative(v1, paths), rows=rows)
    if deliver:
        sent = list(paths)
        for one in built:
            history = os.path.join(v1, "ingest_run")
            sent += [os.path.join(history, name)
                     for name in sorted(os.listdir(history))
                     if name.startswith(f"{one.key}.")
                     and name.endswith(".jsonl")]
        outcome.delivery = send_to_dataset(sorted(set(sent)), v1,
                                           file_manager=file_manager)
    return outcome


def export_folder(run_root: Any, root: Any = None, *, deliver: bool = False,
                  file_manager: Any = None) -> List[ExportOutcome]:
    """Export every report under ``run_root`` (a folder run's, the cluster
    stage's ``ingest/``, or one report's folder) into ``<root>/report_tables
    /v1/``, ``root`` defaulting to ``run_root``. Each report found with its
    bound-in reports; no model is called. With ``deliver``, each report's
    files go to the Foundry dataset as after an ingest."""
    run_root = str(run_root)
    root = run_root if root is None else str(root)
    outcomes: List[ExportOutcome] = []
    for folder, dirs, names in os.walk(run_root):
        dirs[:] = sorted(d for d in dirs if d not in (
            EXPORT_DIR, "items", "descriptions"))
        if _RECORD not in names:
            continue
        dirs[:] = []                      # bound-in reports go with it
        blob = _read_json(os.path.join(folder, _RECORD))
        if isinstance(blob, dict) and blob.get("parent"):
            continue
        outcomes.append(export_report(folder, root, deliver=deliver,
                                      file_manager=file_manager))
    return outcomes


def export_outcome(out_dir: Any) -> Optional[Dict[str, Any]]:
    """What the latest ingest's export of the report in ``out_dir`` did, as
    it saved it in :data:`RUN_NAME`; None when it saved nothing (the export
    off, or a folder older than this module)."""
    run = _read_json(os.path.join(str(out_dir), RUN_NAME))
    return run.get("export") if isinstance(run, dict) else None


# ---------------------------------------------------------------------------
# delivery to a Foundry dataset
# ---------------------------------------------------------------------------

def dataset_alias() -> str:
    """The dataset alias :data:`ENV_DATASET` names, or ""."""
    return str(os.environ.get(ENV_DATASET) or "").strip()


def send_to_dataset(paths: Sequence[str], v1: str, *,
                    file_manager: Any = None) -> str:
    """Send ``paths`` (files under the export's folder ``v1``) to the
    Foundry dataset :data:`ENV_DATASET` names, in ONE save, each at
    ``<the application's top folder>/report_tables/v1/<its path under
    v1>``; the top folder is the one the conversation store saves under
    (``aec_agent_core.foundry_dataset_store``), so the save shares that
    store's lock and its outputs folder and goes up in one transaction.
    Only on a Foundry deployment with the alias set; otherwise nothing is
    imported and nothing sent. Never raises: what happened, in words.
    ``file_manager`` replaces the store's file manager (tests)."""
    alias = dataset_alias()
    if not alias:
        return "not set"
    try:
        from aec_agent_core import engine_config
        if not engine_config.is_foundry_deployment():
            return "not a Foundry deployment"
        from aec_agent_core import foundry_dataset_store as store
        top = store._app_top()
        fm = file_manager or store.DatasetFileManager(alias=alias)
    except Exception as exc:                           # noqa: BLE001
        return f"failed: {type(exc).__name__}: {exc}"
    prefix = f"{top}/{EXPORT_DIR}/{EXPORT_VERSION}"
    try:
        fm.begin_batch()
        try:
            for path in paths:
                rel = os.path.relpath(path, v1).replace(os.sep, "/")
                fm.upload_file(path, f"{prefix}/{rel}", overwrite=True)
        except BaseException:
            fm.end_batch(commit=False)
            raise
        sent = fm.end_batch()
    except Exception as exc:                           # noqa: BLE001
        busy = type(exc).__name__ == "StorageBusy"
        _log.warning("report tables: the export was not sent to the "
                     "dataset: %s: %s", type(exc).__name__, exc)
        return (f"{'busy' if busy else 'failed'}: {type(exc).__name__}: "
                f"{exc}")
    return f"sent {sent} file(s)"


# ---------------------------------------------------------------------------
# at the end of an ingest
# ---------------------------------------------------------------------------

def _save_run(folder: str, facts: Mapping[str, Any]) -> None:
    _write_bytes(os.path.join(folder, RUN_NAME),
                 (json.dumps(dict(facts), indent=1, ensure_ascii=False)
                  + "\n").encode("utf-8"))


def after_ingest(out_dir: Any, *, record: Any, children: Sequence[Any] = (),
                 source: Any = None, key_source: Any = None,
                 db_path: Any = None, setting: Any = None,
                 question_list: bool = False,
                 now: Optional[datetime] = None) -> ExportOutcome:
    """The graph's last step: the run's facts beside each record written
    (:data:`RUN_NAME`: the run id and time, the keys, the source file's
    name, size and SHA-256 off ``source``, the package versions, whether
    the question list ran), then the report exported
    (:func:`export_report`) under :func:`export_root` ``(setting, out_dir,
    db_path)`` and sent to the Foundry dataset when one is set; the
    outcome kept in the report's run file. ``children`` are the graph's
    ``(child record, its folder)`` pairs; ``key_source`` what the graph
    keyed the library row with. Off, nothing at all is written. Never
    raises."""
    out = str(out_dir)
    root = None
    try:
        root = export_root(setting, out, db_path)
        if root is None:
            return ExportOutcome(status="off")
        from report_ingest.writers import record_key

        moment = now or datetime.now(timezone.utc)
        run_id = f"{_stamp(moment)}-{secrets.token_hex(3)}"
        base = record_key(record, key_source)
        facts = {"format": RUN_FORMAT, "run_id": run_id,
                 "ingested_at": _utc(moment), "report_key": base,
                 "parent_report_key": None, **source_facts(source),
                 "app_version": _version(APP_DIST),
                 "core_version": _version(CORE_DIST),
                 "planlens_version": _version(PLANLENS_DIST),
                 "question_list": bool(question_list)}
        for child, child_dir in children:
            _save_run(str(child_dir), {
                **facts, "report_key": record_key(child, key_source, base),
                "parent_report_key": base, "question_list": False})
        _save_run(out, facts)
        outcome = export_report(out, root,
                                deliver=bool(dataset_alias()))
    except Exception as exc:                           # noqa: BLE001
        _log.warning("report tables: the export did not run: %s: %s",
                     type(exc).__name__, exc)
        outcome = ExportOutcome(status="error",
                                folder=export_dir(root) if root else "",
                                detail=f"{type(exc).__name__}: {exc}")
    try:
        run = _read_json(os.path.join(out, RUN_NAME))
        if isinstance(run, dict):
            run["export"] = outcome.to_dict()
            _save_run(out, run)
    except Exception as exc:                           # noqa: BLE001
        _log.warning("report tables: the outcome was not saved: %s", exc)
    return outcome


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m report_ingest.tables <run_root> [--root R] [--deliver]``:
    every report already ingested under ``run_root`` exported, no model
    called. Exit 1 when any report failed."""
    parser = argparse.ArgumentParser(
        prog="python -m report_ingest.tables",
        description="Export the reports ingested under a folder as the "
                    "report tables (docs/REPORT_TABLES.md).")
    parser.add_argument("run_root", nargs="?",
                        help="a folder of ingested reports, or one report's "
                             "folder")
    parser.add_argument("--root", default=None,
                        help="the folder to hold report_tables/ (default: "
                             "run_root)")
    parser.add_argument("--deliver", action="store_true",
                        help=f"also send each report to the Foundry dataset "
                             f"{ENV_DATASET} names (Foundry only)")
    parser.add_argument("--columns", action="store_true",
                        help="print the tables' columns as Markdown (the "
                             "section docs/REPORT_TABLES.md carries) and "
                             "stop")
    args = parser.parse_args(argv)
    if args.columns:
        print(columns_markdown())
        return 0
    if not args.run_root:
        parser.error("a run_root is needed (or --columns)")
    outcomes = export_folder(args.run_root, args.root,
                             deliver=args.deliver)
    for one in outcomes:
        line = f"{one.report_key or '?'}: {one.status}"
        if one.status == "written":
            line += (f", {one.rows.get('report_page', 0)} page row(s), "
                     f"{one.rows.get('report_answer', 0)} answer row(s)")
        if one.delivery:
            line += f"; dataset: {one.delivery}"
        if one.detail:
            line += f" ({one.detail})"
        print(line)
    print(f"{len(outcomes)} report(s) under {args.run_root}")
    return 1 if any(o.status == "error" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
