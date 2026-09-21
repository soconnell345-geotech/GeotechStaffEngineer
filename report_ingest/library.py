"""A folder of ingested reports as ONE library that can be asked questions.

The ingest makes a record per report. This makes those records answerable
ACROSS reports: which of them named a post, what each one recommended, which
page prints a value, where two of them disagree. Nothing here reads a PDF and
nothing here asks a model anything -- every answer is read off records that
were already written, which is what makes an answer citable.

WHAT A LIBRARY IS. A folder written by the ingest or by
:func:`report_ingest.run_folder.run_folder`::

    <root>/reports.db                      the index (writers.upsert_report)
    <root>/<ID>/report.record.json         the record
    <root>/<ID>/report.page.md             the WikiLLM page
    <root>/<ID>/report.summary.md          the summary
    <root>/<ID>/bound/<child>/...          a report bound inside that one

A report bound inside another is a report of the library in its own right,
with ``parent`` pointing at the one it came out of. It is searched like any
other and says whose appendix it was.

THE INDEX IS DERIVED, NEVER THE TRUTH. The record files are. So a folder
restored from SharePoint with no ``reports.db``, or one whose records were
re-written after the database was built, rebuilds the index from the records
on the next question -- both the ``reports`` rows the writers own and the
full-text index this module adds. The check is mtimes: the newest record,
page or summary file under the root against the stamp the last build left in
``meta``. Nothing is written back into a record, so a rebuild is always safe
and always cheap enough to do on open.

FINDING THINGS. SQLite FTS5 over chunks of the records and of the pages and
summaries beside them -- the same idiom the reference layer uses -- with a
rapidfuzz pass over the FIELD VALUES when the full-text query comes back
thin, which is what catches a firm's name spelled a second way. Every hit carries
the report it came from and the PDF pages it was read from, because a fact
from this library that cannot be checked against a page is worth nothing.

PRIVACY. This module holds no report content of its own: names, posts and
firms reach it only from records the caller points it at, and the folder of
records is the caller's to keep out of anything committed.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from report_ingest.model import (
    GENERAL_FIELDS, NATURAL_HAZARD_FIELDS, ReportRecord,
)
from report_ingest.writers import (
    DB_NAME, PAGE_NAME, RECORD_NAME, SUMMARY_NAME, open_library, parent_key,
    record_key, upsert_report,
)

__all__ = [
    "Library", "NARRATIVE_FIELDS", "REVIEW_KINDS", "MAX_ROWS",
    "SECTIONS", "LibraryError",
]

#: Every narrative field a question may name: the owner's 25 general fields
#: and 12 natural-hazard fields, in the owner's order. One tuple, taken from
#: the model, so a field added there is askable here without a second list.
NARRATIVE_FIELDS: Tuple[str, ...] = GENERAL_FIELDS + NATURAL_HAZARD_FIELDS

#: The QA kinds a PERSON should look at. A ``note`` is the record telling a
#: reader something went right; a ``skipped`` is a decision the pipeline made
#: on purpose. These six are the ones where two things disagree, a count does
#: not add up, or a page could not be read -- the entries the owner asked to
#: be able to list across a whole library.
REVIEW_KINDS: Tuple[str, ...] = (
    "conflict", "disagreement", "label_disagreement", "count_mismatch",
    "out_of_range", "unreadable",
)

#: Where a chunk of text came from, for filtering a search.
SECTIONS: Tuple[str, ...] = (
    "document", "general", "natural_hazards", "exploration", "lab",
    "calculation", "qa", "bound", "page", "summary",
)

#: How many rows any one answer returns before it says it was cut. A library
#: agent's tool result goes into a model's context, and forty rows of it is
#: already more than an answer needs.
MAX_ROWS = 40

#: How many chunks the fuzzy fallback will look at. rapidfuzz is fast enough
#: that this is about bounding the ANSWER's cost, not the search's.
MAX_FUZZY_CHUNKS = 20000

#: The fuzzy score below which a match is noise rather than a near miss.
FUZZY_FLOOR = 70

#: The shortest query the fuzzy pass will run on. At three letters every
#: value in a library is within a few edits of the query and the pass returns
#: noise dressed as a near miss.
MIN_FUZZY_QUERY = 4

#: The meta key carrying the mtime the index was last built against.
_STAMP = "library_index_mtime"

_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS library_facts (
    id                TEXT PRIMARY KEY,
    report_id         TEXT,
    folder            TEXT,
    record_path       TEXT,
    page_path         TEXT,
    summary_path      TEXT,
    parent            TEXT,
    title             TEXT,
    firm              TEXT,
    post              TEXT,
    property_type     TEXT,
    phase             TEXT,
    document_type     TEXT,
    project_number    TEXT,
    project_name      TEXT,
    report_date       TEXT,
    report_date_iso   TEXT,
    year              INTEGER,
    status            TEXT,
    confidence        TEXT,
    workflow          TEXT,
    n_pages           INTEGER,
    n_investigations  INTEGER,
    n_lab_tests       INTEGER,
    n_calculations    INTEGER,
    n_qa              INTEGER,
    n_review          INTEGER,
    n_answered        INTEGER,
    kinds             TEXT,
    load_error        TEXT
);
CREATE INDEX IF NOT EXISTS library_facts_report ON library_facts (report_id);
CREATE INDEX IF NOT EXISTS library_facts_parent ON library_facts (parent);
CREATE TABLE IF NOT EXISTS library_chunks (
    rowid     INTEGER PRIMARY KEY AUTOINCREMENT,
    id        TEXT,
    report_id TEXT,
    section   TEXT,
    subject   TEXT,
    page      INTEGER,
    pages     TEXT,
    text      TEXT
);
CREATE INDEX IF NOT EXISTS library_chunks_id ON library_chunks (id);
CREATE VIRTUAL TABLE IF NOT EXISTS library_chunks_fts USING fts5(
    subject, text, content='library_chunks', content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

#: Words a question is made of rather than about. Dropped from a full-text
#: query so that "which reports mention liquefaction" searches for the one
#: word that narrows anything.
_STOPWORDS = frozenset("""
a an the of in on at for to and or is are was were be been being it its this
that these those what which who whom whose how many much does do did done any
all some there here from with without by as into about over under between
report reports library page pages say says said tell me us we you i they them
than then so if but not no yes can could would should may might must will
""".split())

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PAGE_CITE = re.compile(r"\bp(\d+)\b")
_FRONT_ID = re.compile(r'^id:\s*"?([0-9a-f]{6,64})"?\s*$', re.MULTILINE)


class LibraryError(RuntimeError):
    """The library root is not a folder of ingested reports."""


# ---------------------------------------------------------------------------
# small shapes
# ---------------------------------------------------------------------------

def _show(value: Any) -> str:
    """One value as a page prints it, for a quantity or anything else."""
    if value is None:
        return ""
    unit = getattr(value, "unit", None)
    number = getattr(value, "value", None)
    if unit is not None and isinstance(number, (int, float)):
        return f"{number:g} {unit}".strip()
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "; ".join(_show(v) for v in value)
    return str(value)


def _pages_of(*holders: Any) -> List[int]:
    """Every 0-based page these things name, in order, without repeats."""
    out: List[int] = []
    for holder in holders:
        for page in holder or ():
            number = getattr(page, "page", page)
            if isinstance(number, int) and number not in out:
                out.append(number)
    return out


def _cited(holder: Any, field: str) -> Tuple[List[int], str]:
    """The pages and the first quote behind one narrative field."""
    rows = (getattr(holder, "citations", None) or {}).get(field) or []
    return ([row.page for row in rows],
            next((row.quote for row in rows if row.quote), ""))


def _clip(rows: List[Any], limit: int) -> Tuple[List[Any], bool]:
    return (rows[:limit], len(rows) > limit)


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------

class Library:
    """A folder of ingested reports, asked questions across all of them.

    ``root`` is the folder the ingest wrote into; ``db_path`` defaults to
    ``reports.db`` inside it. Opening is cheap -- the index is checked and
    rebuilt lazily on the first question, not in the constructor -- so a
    caller may build one per turn without paying for a folder it never asks
    about.

    Every query returns PLAIN DATA: dicts and lists of dicts, each row
    carrying the report it came from and the PDF pages behind it. Nothing
    returns a :class:`~report_ingest.model.ReportRecord`; a record runs to
    megabytes and the point of this layer is that an answer does not.
    """

    def __init__(self, root: Any, db_path: Any = None,
                 max_rows: int = MAX_ROWS) -> None:
        self.root = str(root)
        self.db_path = str(db_path) if db_path else os.path.join(self.root,
                                                                 DB_NAME)
        self.max_rows = int(max_rows)
        self._connection: Optional[sqlite3.Connection] = None
        self._records: Dict[str, ReportRecord] = {}
        self._checked = False

    # -- opening, and the index ------------------------------------------

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._records.clear()

    def __enter__(self) -> "Library":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def refresh(self, force: bool = False) -> bool:
        """Rebuild the index if it is missing or stale. True if it rebuilt.

        Stale means a record, page or summary file under the root is newer
        than the stamp the last build left behind -- which is what a folder
        restored from SharePoint, or one the ingest has just added a report
        to, looks like.
        """
        self._checked = True
        files = self._record_files()
        if not files and not os.path.isdir(self.root):
            raise LibraryError(
                f"{self.root!r} is not a folder. A library is the folder the "
                f"ingest wrote its reports into.")
        newest = self._newest_mtime(files)
        if not force and os.path.isfile(self.db_path):
            connection = self._open()
            if self._has_index(connection):
                stamp = self._stamp(connection)
                if stamp is not None and stamp >= newest:
                    return False
        self._build(files, newest)
        return True

    def _ensure(self) -> sqlite3.Connection:
        if not self._checked:
            self.refresh()
        return self._open()

    def _open(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = open_library(self.db_path)
        return self._connection

    def _record_files(self) -> List[str]:
        """Every ``report.record.json`` under the root, parents first.

        Sorted by depth so that a report is indexed before anything bound
        inside it: a child's library key is folded from its parent's, and
        the parent's has to exist by the time the child is keyed.
        """
        out: List[str] = []
        for folder, _dirs, names in os.walk(self.root):
            if RECORD_NAME in names:
                out.append(os.path.join(folder, RECORD_NAME))
        return sorted(out, key=lambda path: (path.count(os.sep), path))

    def _newest_mtime(self, files: Sequence[str]) -> float:
        newest = 0.0
        for record_file in files:
            folder = os.path.dirname(record_file)
            for name in (RECORD_NAME, PAGE_NAME, SUMMARY_NAME):
                path = os.path.join(folder, name)
                try:
                    newest = max(newest, os.path.getmtime(path))
                except OSError:
                    continue
        return newest

    @staticmethod
    def _has_index(connection: sqlite3.Connection) -> bool:
        row = connection.execute(
            "SELECT count(*) AS n FROM sqlite_master WHERE type IN "
            "('table','view') AND name IN ('library_facts', 'library_chunks',"
            " 'library_chunks_fts')").fetchone()
        return bool(row and row["n"] == 3)

    @staticmethod
    def _stamp(connection: sqlite3.Connection) -> Optional[float]:
        row = connection.execute(
            "SELECT value FROM meta WHERE key = ?", (_STAMP,)).fetchone()
        if row is None:
            return None
        try:
            return float(row["value"])
        except (TypeError, ValueError):
            return None

    def _build(self, files: Sequence[str], newest: float) -> None:
        """Read every record under the root back into a fresh index."""
        connection = self._open()
        connection.executescript(_INDEX_SCHEMA)
        connection.executescript(
            "DELETE FROM library_chunks_fts; DELETE FROM library_chunks; "
            "DELETE FROM library_facts;")
        self._records.clear()
        keys_by_file: Dict[str, str] = {}

        for record_file in files:
            folder = os.path.dirname(record_file)
            record, error = _load_record(record_file)
            if record is None:
                connection.execute(
                    "INSERT OR REPLACE INTO library_facts (id, report_id, "
                    "folder, record_path, load_error) VALUES (?,?,?,?,?)",
                    (os.path.relpath(folder, self.root), _folder_id(folder),
                     folder, record_file, error))
                continue

            base = ""
            if record.parent is not None:
                relative = record.parent.record_path or os.path.join(
                    os.pardir, os.pardir, RECORD_NAME)
                base = keys_by_file.get(
                    os.path.normpath(os.path.join(folder, relative)), "")
            key = _key_of(record, folder, base)
            keys_by_file[os.path.normpath(record_file)] = key

            upsert_report(connection, record, key,
                          _paths_in(folder), "",
                          parent=(base or parent_key(record))
                          if record.parent is not None else "")
            self._insert_facts(connection, record, key, folder)
            self._insert_chunks(connection, record, key, folder)

        connection.execute(
            "INSERT INTO library_chunks_fts (rowid, subject, text) "
            "SELECT rowid, subject, text FROM library_chunks")
        connection.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_STAMP, repr(newest)))
        connection.commit()

    def _insert_facts(self, connection: sqlite3.Connection,
                      record: ReportRecord, key: str, folder: str) -> None:
        from report_ingest.writers import (
            confidence_of, status_of, title_of, year_of,
        )

        general, hazards = record.general, record.natural_hazards
        kinds = sorted({inv.kind for inv in record.investigations}
                       | {test.kind for test in record.lab_tests}
                       | ({"calculations"} if record.calculations else set())
                       | ({"bound"} if record.bound_documents else set()))
        connection.execute(
            "INSERT OR REPLACE INTO library_facts (id, report_id, folder, "
            "record_path, page_path, summary_path, parent, title, firm, "
            "post, property_type, phase, document_type, project_number, "
            "project_name, report_date, report_date_iso, year, status, "
            "confidence, workflow, n_pages, n_investigations, n_lab_tests, "
            "n_calculations, n_qa, n_review, n_answered, kinds, load_error) "
            "VALUES (" + ",".join("?" * 30) + ")",
            (key,
             record.document.report_id or _folder_id(folder),
             folder,
             os.path.join(folder, RECORD_NAME),
             os.path.join(folder, PAGE_NAME),
             os.path.join(folder, SUMMARY_NAME),
             (record.parent.report_id if record.parent is not None else ""),
             title_of(record),
             general.geotechnicalEngineerFirm or "",
             general.postName or "",
             general.propertyType or "",
             general.projectPhase or "",
             general.documentType or "",
             general.projectNumber or record.project.number or "",
             general.projectName or record.project.name or "",
             hazards.reportDate or "",
             hazards.reportDateISO or "",
             year_of(record),
             status_of(record),
             confidence_of(record),
             record.document.workflow,
             record.document.n_pages,
             len(record.investigations),
             len(record.lab_tests),
             len(record.calculations),
             len(record.qa),
             sum(1 for entry in record.qa if entry.kind in REVIEW_KINDS),
             len(general.answered()) + len(hazards.answered()),
             json.dumps(kinds),
             ""))

    def _insert_chunks(self, connection: sqlite3.Connection,
                       record: ReportRecord, key: str, folder: str) -> None:
        report_id = record.document.report_id or _folder_id(folder)
        rows = [(key, report_id, section, subject,
                 (pages[0] if pages else None), json.dumps(pages), text)
                for section, subject, pages, text
                in _chunks(record, folder)]
        connection.executemany(
            "INSERT INTO library_chunks (id, report_id, section, subject, "
            "page, pages, text) VALUES (?,?,?,?,?,?,?)", rows)

    # -- resolving a report ----------------------------------------------

    def _resolve(self, report_id: Any) -> Optional[str]:
        """The library key for whatever a caller called a report."""
        wanted = str(report_id or "").strip()
        if not wanted:
            return None
        connection = self._ensure()
        for sql in ("SELECT id FROM library_facts WHERE id = ?",
                    "SELECT id FROM library_facts WHERE report_id = ?",
                    "SELECT id FROM library_facts WHERE "
                    "lower(report_id) = lower(?)",
                    "SELECT id FROM library_facts WHERE "
                    "lower(project_name) = lower(?)"):
            row = connection.execute(sql, (wanted,)).fetchone()
            if row is not None:
                return row["id"]
        row = connection.execute(
            "SELECT id FROM library_facts WHERE folder LIKE ? "
            "ORDER BY length(folder) LIMIT 1",
            (f"%{os.sep}{wanted}",)).fetchone()
        return row["id"] if row is not None else None

    def _record(self, key: str) -> Optional[ReportRecord]:
        """The whole record for one library key, loaded once per session."""
        if key in self._records:
            return self._records[key]
        connection = self._ensure()
        row = connection.execute(
            "SELECT record_path FROM library_facts WHERE id = ?",
            (key,)).fetchone()
        if row is None:
            return None
        record, _error = _load_record(row["record_path"])
        if record is not None:
            self._records[key] = record
        return record

    def _report_id(self, key: str) -> str:
        row = self._ensure().execute(
            "SELECT report_id FROM library_facts WHERE id = ?",
            (key,)).fetchone()
        return row["report_id"] if row is not None else key

    def _not_found(self, report_id: Any) -> Dict[str, Any]:
        known = [row["report_id"] for row in self._ensure().execute(
            "SELECT report_id FROM library_facts ORDER BY report_id LIMIT 25")]
        return {"error": f"the library holds no report called "
                         f"{str(report_id)!r}",
                "reports_in_the_library": known}

    # -- the queries ------------------------------------------------------

    def list_reports(self, post: str = "", property_type: str = "",
                     phase: str = "", firm: str = "",
                     date_from: str = "", date_to: str = "",
                     document_type: str = "", has_kind: str = "",
                     limit: int = 0) -> Dict[str, Any]:
        """The reports of the library that match every filter given.

        Every filter is a CONTAINS match, case-insensitive, except
        ``has_kind`` (an exploration kind, a laboratory test kind,
        ``calculations`` or ``bound``, matched exactly) and the dates, which
        are ISO ``YYYY-MM-DD`` bounds against the date the report prints for
        itself. A filter left empty does not filter.
        """
        connection = self._ensure()
        where, args = ["load_error = ''"], []
        for column, value in (("post", post),
                              ("property_type", property_type),
                              ("phase", phase), ("firm", firm),
                              ("document_type", document_type)):
            if str(value or "").strip():
                where.append(f"lower({column}) LIKE lower(?)")
                args.append(f"%{str(value).strip()}%")
        if str(date_from or "").strip():
            where.append("report_date_iso != '' AND report_date_iso >= ?")
            args.append(str(date_from).strip())
        if str(date_to or "").strip():
            where.append("report_date_iso != '' AND report_date_iso <= ?")
            args.append(str(date_to).strip())
        if str(has_kind or "").strip():
            where.append("kinds LIKE ?")
            args.append(f'%"{str(has_kind).strip()}"%')

        rows = connection.execute(
            "SELECT * FROM library_facts WHERE " + " AND ".join(where)
            + " ORDER BY report_id", args).fetchall()
        kept, cut = _clip([self._report_row(row) for row in rows],
                          limit or self.max_rows)
        return {"reports": kept, "n": len(rows), "truncated": cut,
                "filters": {name: value for name, value in (
                    ("post", post), ("property_type", property_type),
                    ("phase", phase), ("firm", firm),
                    ("date_from", date_from), ("date_to", date_to),
                    ("document_type", document_type),
                    ("has_kind", has_kind)) if value}}

    def _report_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        out = {
            "report": row["report_id"], "key": row["id"],
            "title": row["title"], "firm": row["firm"], "post": row["post"],
            "property_type": row["property_type"], "phase": row["phase"],
            "document_type": row["document_type"],
            "date": row["report_date"] or row["report_date_iso"] or "",
            "year": row["year"], "n_pages": row["n_pages"],
            "explorations": row["n_investigations"],
            "lab_tests": row["n_lab_tests"],
            "calculations": row["n_calculations"],
            "questions_answered": row["n_answered"],
            "needs_review": row["n_review"],
            "status": row["status"], "confidence": row["confidence"],
            "kinds": json.loads(row["kinds"] or "[]"),
        }
        if row["parent"]:
            out["bound_inside"] = row["parent"]
        return out

    def find(self, text: str, k: int = 8, section: str = "",
             report_id: str = "") -> Dict[str, Any]:
        """Search the library: full text first, then fuzzy on what is left.

        Returns at most ``k`` hits, each with the report it came from, the
        PDF pages behind it, a snippet and whether it was found by the
        full-text index or by the fuzzy pass.
        """
        query = str(text or "").strip()
        if not query:
            return {"query": "", "hits": [], "n": 0,
                    "error": "give something to search for"}
        k = max(1, min(int(k or 8), self.max_rows))
        key = self._resolve(report_id) if report_id else None
        if report_id and key is None:
            return dict(self._not_found(report_id), query=query, hits=[], n=0)

        hits = self._search_fts(query, k, section, key)
        if len(hits) < k:
            seen = {hit["_rowid"] for hit in hits}
            hits += self._search_fuzzy(query, k - len(hits), section, key,
                                       seen)
        for hit in hits:
            hit.pop("_rowid", None)
        return {"query": query, "hits": hits, "n": len(hits),
                "searched": section or "everything"}

    def _search_fts(self, query: str, k: int, section: str,
                    key: Optional[str]) -> List[Dict[str, Any]]:
        match = _fts_query(query)
        if not match:
            return []
        connection = self._ensure()
        where, args = ["library_chunks_fts MATCH ?"], [match]
        if section:
            where.append("c.section = ?")
            args.append(section)
        if key:
            where.append("c.id = ?")
            args.append(key)
        args.append(k)
        try:
            rows = connection.execute(
                "SELECT c.rowid AS rid, c.id, c.report_id, c.section, "
                "c.subject, c.pages, c.text, "
                "snippet(library_chunks_fts, 1, '', '', ' ... ', 18) AS snip, "
                "bm25(library_chunks_fts, 3.0, 1.0) AS rank "
                "FROM library_chunks_fts JOIN library_chunks c "
                "ON c.rowid = library_chunks_fts.rowid "
                "WHERE " + " AND ".join(where)
                + " ORDER BY rank LIMIT ?", args).fetchall()
        except sqlite3.OperationalError:
            return []                    # a query FTS5 will not parse
        return [self._hit(row, row["snip"], -float(row["rank"]), "fts")
                for row in rows]

    def _search_fuzzy(self, query: str, k: int, section: str,
                      key: Optional[str],
                      seen: Iterable[int]) -> List[Dict[str, Any]]:
        """rapidfuzz over the FIELD VALUES, for what the index did not match.

        The fallback that matters: a firm, a post, a project or a hole
        spelled a second way is one edit from the record and nothing at all
        to a tokeniser. It runs over VALUES rather than over whole chunks
        because a twenty-character query against four hundred characters of
        text scores as a mismatch however close the name inside it is.

        Absent rapidfuzz this returns nothing rather than failing -- the
        full-text answer is still an answer. A query under
        :data:`MIN_FUZZY_QUERY` characters is not fuzzed at all: at three
        letters every value in the library is a near miss.
        """
        try:
            from rapidfuzz import fuzz, process
        except ImportError:                          # pragma: no cover
            return []
        text = str(query or "").strip()
        if len(text) < MIN_FUZZY_QUERY:
            return []
        skip = set(seen)
        pool = [row for row in self._fuzzy_pool(section, key)
                if row[1]["_rowid"] not in skip]
        if not pool:
            return []
        matches = process.extract(text, [choice for choice, _hit in pool],
                                  scorer=fuzz.WRatio, limit=k,
                                  score_cutoff=FUZZY_FLOOR)
        out: List[Dict[str, Any]] = []
        for _choice, score, index in matches:
            hit = dict(pool[index][1])
            hit["score"] = round(float(score), 1)
            out.append(hit)
        return out

    def _fuzzy_pool(self, section: str, key: Optional[str]
                    ) -> List[Tuple[str, Dict[str, Any]]]:
        """``(value, hit)`` for everything the fuzzy pass may match.

        Two sources: the identity columns of the index -- a title, a firm, a
        post, a project, a date, each one short -- and the VALUE half of
        every chunk that holds one.
        """
        connection = self._ensure()
        pool: List[Tuple[str, Dict[str, Any]]] = []

        if not section or section == "document":
            where = "load_error = ''" + (" AND id = ?" if key else "")
            for row in connection.execute(
                    "SELECT id, report_id, title, firm, post, project_name, "
                    "project_number, property_type, phase, document_type, "
                    "report_date FROM library_facts WHERE " + where,
                    (key,) if key else ()):
                for column in ("title", "firm", "post", "project_name",
                               "project_number", "property_type", "phase",
                               "document_type", "report_date"):
                    value = row[column]
                    if not value:
                        continue
                    pool.append((str(value), {
                        "report": row["report_id"], "key": row["id"],
                        "section": "document", "subject": column,
                        "pages": [], "snippet": str(value), "score": 0.0,
                        "found_by": "fuzzy", "_rowid": -1}))

        where, args = ["1 = 1"], []
        if section:
            where.append("section = ?")
            args.append(section)
        if key:
            where.append("id = ?")
            args.append(key)
        args.append(MAX_FUZZY_CHUNKS)
        for row in connection.execute(
                "SELECT rowid AS rid, id, report_id, section, subject, "
                "pages, text FROM library_chunks WHERE "
                + " AND ".join(where) + " LIMIT ?", args):
            pool.append((_fuzzy_value(row), {
                "report": row["report_id"], "key": row["id"],
                "section": row["section"], "subject": row["subject"],
                "pages": json.loads(row["pages"] or "[]"),
                "snippet": " ".join(str(row["text"]).split())[:240],
                "score": 0.0, "found_by": "fuzzy", "_rowid": row["rid"]}))
        return pool

    def _hit(self, row: sqlite3.Row, snippet: str, score: float,
             how: str) -> Dict[str, Any]:
        return {"report": row["report_id"], "key": row["id"],
                "section": row["section"], "subject": row["subject"],
                "pages": json.loads(row["pages"] or "[]"),
                "snippet": " ".join(str(snippet or "").split())[:240],
                "score": score, "found_by": how, "_rowid": row["rid"]}

    def facts(self, report_id: str,
              fields: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """What one report answered, for any of the 37 narrative fields.

        ``fields`` names the ones wanted; with none it returns every field
        the report actually answered. A field the report did not answer is
        listed under ``not_answered`` rather than returned empty, because a
        report that did not say and a reader that did not read must not look
        the same.
        """
        key = self._resolve(report_id)
        if key is None:
            return self._not_found(report_id)
        record = self._record(key)
        if record is None:
            return {"error": f"the record for {report_id!r} could not be read"}
        rid = self._report_id(key)

        asked = [str(name).strip() for name in (fields or ()) if str(name).strip()]
        unknown = [name for name in asked if name not in NARRATIVE_FIELDS]
        wanted = [name for name in asked if name in NARRATIVE_FIELDS] or \
            list(NARRATIVE_FIELDS)

        out: List[Dict[str, Any]] = []
        missing: List[str] = []
        for name in wanted:
            holder = (record.general if name in GENERAL_FIELDS
                      else record.natural_hazards)
            value = getattr(holder, name, None)
            if value is None:
                missing.append(name)
                continue
            pages, quote = _cited(holder, name)
            row = {"report": rid, "field": name, "value": _show(value),
                   "pages": pages}
            if quote:
                row["quote"] = quote
            out.append(row)

        kept, cut = _clip(out, self.max_rows)
        by_kind: Dict[str, int] = {}
        for entry in record.qa:
            by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
        answer = {
            "report": rid, "key": key,
            "title": _title_of(record),
            "fields": kept, "truncated": cut,
            "not_answered": missing if asked else missing[:self.max_rows],
            "counts": record.counts(),
            "qa": {"total": len(record.qa), "by_kind": by_kind,
                   "needs_review": sum(1 for entry in record.qa
                                       if entry.kind in REVIEW_KINDS)},
        }
        if unknown:
            answer["not_a_field"] = unknown
        return answer

    def compare(self, field: str,
                report_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """One narrative field across several reports, as a table.

        With no ``report_ids`` it compares every report in the library. A
        report that did not answer the field is named under ``missing``, so
        the table never pretends a silence was an answer.
        """
        name = str(field or "").strip()
        if name not in NARRATIVE_FIELDS:
            return {"error": f"{name!r} is not one of the 37 narrative "
                             f"fields", "fields": list(NARRATIVE_FIELDS)}
        keys: List[str] = []
        unknown: List[str] = []
        for wanted in (report_ids or ()):
            key = self._resolve(wanted)
            (keys if key else unknown).append(key or str(wanted))
        if not report_ids:
            keys = [row["id"] for row in self._ensure().execute(
                "SELECT id FROM library_facts WHERE load_error = '' "
                "ORDER BY report_id")]

        rows: List[Dict[str, Any]] = []
        missing: List[str] = []
        for key in keys:
            record = self._record(key)
            if record is None:
                continue
            rid = self._report_id(key)
            holder = (record.general if name in GENERAL_FIELDS
                      else record.natural_hazards)
            value = getattr(holder, name, None)
            if value is None:
                missing.append(rid)
                continue
            pages, quote = _cited(holder, name)
            row = {"report": rid, "value": _show(value), "pages": pages}
            if quote:
                row["quote"] = quote
            rows.append(row)
        kept, cut = _clip(rows, self.max_rows)
        answer = {"field": name, "rows": kept, "n": len(rows),
                  "truncated": cut, "missing": missing}
        if unknown:
            answer["not_in_the_library"] = unknown
        return answer

    def explorations(self, report_id: str,
                     kind: str = "") -> Dict[str, Any]:
        """One report's explorations, with their depths and their samples."""
        key = self._resolve(report_id)
        if key is None:
            return self._not_found(report_id)
        record = self._record(key)
        if record is None:
            return {"error": f"the record for {report_id!r} could not be read"}
        rid = self._report_id(key)
        wanted = str(kind or "").strip().lower()
        chosen = [inv for inv in record.investigations
                  if not wanted or inv.kind == wanted]
        kept, cut = _clip(chosen, self.max_rows)
        return {"report": rid, "key": key, "kind": wanted or "any",
                "n": len(chosen), "truncated": cut,
                "explorations": [self._exploration(rid, inv) for inv in kept]}

    def _exploration(self, rid: str, inv: Any) -> Dict[str, Any]:
        layers, layers_cut = _clip(inv.layers, self.max_rows)
        samples, samples_cut = _clip(inv.samples, self.max_rows)
        out: Dict[str, Any] = {
            "report": rid,
            "exploration": inv.investigation_id,
            "kind": inv.kind,
            "depth_unit": inv.depth_unit,
            "units_known": inv.units_known,
            "total_depth": _show(inv.total_depth),
            "ground_elevation": _show(inv.elevation),
            "dates": " to ".join(x for x in (inv.date_started,
                                             inv.date_finished) if x),
            "pages": list(inv.pages),
            "n_layers": len(inv.layers),
            "layers": [{"top": _show(layer.top), "bottom": _show(layer.bottom),
                        "description": layer.description, "uscs": layer.uscs}
                       for layer in layers],
            "n_samples": len(inv.samples),
            "samples": [{"sample": sample.sample_id,
                         "top": _show(sample.top),
                         "bottom": _show(sample.bottom),
                         "kind": sample.kind, "uscs": sample.uscs,
                         "water_content": sample.water_content}
                        for sample in samples],
            "spt": [{"depth": _show(drive.depth_top), "n": drive.n,
                     "blows": list(drive.blows), "sample": drive.sample_id}
                    for drive in inv.spt[:self.max_rows]],
            "water": [{"depth": _show(level.depth), "when": level.when}
                      for level in inv.water[:self.max_rows]],
        }
        if layers_cut or samples_cut:
            out["truncated"] = True
        if inv.pit is not None:
            out["pit"] = {"length": _show(inv.pit.length),
                          "width": _show(inv.pit.width),
                          "depth": _show(inv.pit.depth)}
        if inv.cpt is not None:
            out["cpt_points"] = inv.cpt.n_points
        if inv.dcp is not None:
            out["dcp_points"] = inv.dcp.n_points
        if inv.remarks:
            out["remarks"] = inv.remarks[:400]
        return out

    def lab_summary(self, report_id: str, kind: str = "") -> Dict[str, Any]:
        """One report's laboratory testing: how much of each, and each test."""
        key = self._resolve(report_id)
        if key is None:
            return self._not_found(report_id)
        record = self._record(key)
        if record is None:
            return {"error": f"the record for {report_id!r} could not be read"}
        rid = self._report_id(key)
        wanted = str(kind or "").strip().lower()
        chosen = [test for test in record.lab_tests
                  if not wanted or test.kind == wanted]
        by_kind: Dict[str, int] = {}
        for test in record.lab_tests:
            by_kind[test.kind] = by_kind.get(test.kind, 0) + 1
        kept, cut = _clip(chosen, self.max_rows)
        return {"report": rid, "key": key, "kind": wanted or "any",
                "n": len(chosen), "truncated": cut, "by_kind": by_kind,
                "tests": [{
                    "report": rid, "test": test.kind,
                    "exploration": test.investigation_id or "",
                    "sample": test.sample_id or "",
                    "depth": _show(test.depth_top),
                    "standard": test.standard,
                    "linked_exploration": test.linked_investigation_id,
                    "linked_sample": test.linked_sample_id,
                    "values": _result_values(test.result),
                    "pages": list(test.pages)} for test in kept]}

    def calculations(self, report_id: str) -> Dict[str, Any]:
        """What one report worked out: the printouts, as the pages print."""
        key = self._resolve(report_id)
        if key is None:
            return self._not_found(report_id)
        record = self._record(key)
        if record is None:
            return {"error": f"the record for {report_id!r} could not be read"}
        rid = self._report_id(key)
        kept, cut = _clip(record.calculations, self.max_rows)
        return {"report": rid, "key": key, "n": len(record.calculations),
                "truncated": cut,
                "calculations": [{
                    "report": rid, "works_out": calc.kind,
                    "program": calc.program or "not stated",
                    "method": calc.method, "for": calc.subject,
                    "summary": calc.summary,
                    "inputs": [str(row) for row in calc.inputs[:12]],
                    "results": [str(row) for row in calc.results[:12]],
                    "exploration": calc.linked_investigation_id,
                    "pages": list(calc.pages)} for calc in kept]}

    def disagreements(self, report_id: str = "") -> Dict[str, Any]:
        """The QA entries a person should look at, for one report or all.

        The six kinds where two things disagree, a count does not add up or
        a page could not be read. A ``note`` says something went right and a
        ``skipped`` was a decision on purpose; neither is here.
        """
        if report_id:
            key = self._resolve(report_id)
            if key is None:
                return self._not_found(report_id)
            keys = [key]
        else:
            keys = [row["id"] for row in self._ensure().execute(
                "SELECT id FROM library_facts WHERE n_review > 0 "
                "ORDER BY report_id")]

        rows: List[Dict[str, Any]] = []
        for key in keys:
            record = self._record(key)
            if record is None:
                continue
            rid = self._report_id(key)
            for entry in record.qa:
                if entry.kind not in REVIEW_KINDS:
                    continue
                rows.append({"report": rid, "kind": entry.kind,
                             "where": entry.where, "detail": entry.detail,
                             "values": list(entry.values[:5]),
                             "pages": list(entry.pages)})
        kept, cut = _clip(rows, self.max_rows)
        return {"entries": kept, "n": len(rows), "truncated": cut,
                "reports": sorted({row["report"] for row in kept}),
                "kinds_looked_for": list(REVIEW_KINDS)}

    def where_is(self, text: str, k: int = 8) -> Dict[str, Any]:
        """Which report and which page prints this.

        The same search as :meth:`find`, shaped as PLACES rather than hits:
        one row per report and page, deduplicated, so the answer is a list
        of somewhere to look.
        """
        found = self.find(text, k=max(int(k or 8), 8))
        if found.get("error"):
            return found
        places: List[Dict[str, Any]] = []
        seen = set()
        for hit in found["hits"]:
            pages = hit["pages"] or [None]
            for page in pages:
                mark = (hit["report"], page)
                if mark in seen:
                    continue
                seen.add(mark)
                places.append({"report": hit["report"], "page": page,
                               "section": hit["section"],
                               "subject": hit["subject"],
                               "snippet": hit["snippet"],
                               "found_by": hit["found_by"]})
        kept, cut = _clip(places, min(int(k or 8), self.max_rows))
        return {"query": found["query"], "locations": kept, "n": len(places),
                "truncated": cut}

    def library_stats(self) -> Dict[str, Any]:
        """What the library holds: how many of what, and over what years."""
        connection = self._ensure()
        row = connection.execute(
            "SELECT count(*) AS n, "
            "sum(parent != '') AS bound, "
            "sum(n_pages) AS pages, "
            "sum(n_investigations) AS investigations, "
            "sum(n_lab_tests) AS lab_tests, "
            "sum(n_calculations) AS calculations, "
            "sum(n_qa) AS qa, sum(n_review) AS review, "
            "min(nullif(year, 0)) AS first_year, max(year) AS last_year "
            "FROM library_facts WHERE load_error = ''").fetchone()
        chunks = connection.execute(
            "SELECT count(*) AS n FROM library_chunks").fetchone()["n"]
        failed = [dict(report=r["report_id"], error=r["load_error"])
                  for r in connection.execute(
                      "SELECT report_id, load_error FROM library_facts "
                      "WHERE load_error != ''")]
        kinds: Dict[str, int] = {}
        for value in connection.execute(
                "SELECT kinds FROM library_facts WHERE load_error = ''"):
            for name in json.loads(value["kinds"] or "[]"):
                kinds[name] = kinds.get(name, 0) + 1
        answer = {
            "root": self.root, "db": self.db_path,
            "reports": int(row["n"] or 0),
            "bound_inside_another": int(row["bound"] or 0),
            "pages": int(row["pages"] or 0),
            "totals": {"explorations": int(row["investigations"] or 0),
                       "lab_tests": int(row["lab_tests"] or 0),
                       "calculations": int(row["calculations"] or 0),
                       "qa_entries": int(row["qa"] or 0),
                       "needing_review": int(row["review"] or 0)},
            "years": {"first": row["first_year"], "last": row["last_year"]},
            "kinds": dict(sorted(kinds.items())),
            "indexed_chunks": int(chunks or 0),
        }
        for label, column in (("by_document_type", "document_type"),
                              ("by_status", "status"),
                              ("by_confidence", "confidence"),
                              ("firms", "firm"), ("posts", "post"),
                              ("property_types", "property_type"),
                              ("phases", "phase")):
            answer[label] = {
                r["value"]: r["n"] for r in connection.execute(
                    f"SELECT {column} AS value, count(*) AS n FROM "
                    f"library_facts WHERE load_error = '' AND {column} != '' "
                    f"GROUP BY {column} ORDER BY n DESC, value")}
        if failed:
            answer["could_not_be_read"] = failed
        return answer


# ---------------------------------------------------------------------------
# reading a record folder
# ---------------------------------------------------------------------------

def _folder_id(folder: str) -> str:
    return os.path.basename(os.path.normpath(folder)) or "report"


def _paths_in(folder: str) -> Dict[str, str]:
    names = {"record": RECORD_NAME, "summary": SUMMARY_NAME,
             "page": PAGE_NAME, "diggs": "report.diggs.xml"}
    return {key: os.path.join(folder, name) for key, name in names.items()
            if os.path.isfile(os.path.join(folder, name))}


def _load_record(path: str) -> Tuple[Optional[ReportRecord], str]:
    """One record off disk, or None and why not.

    A record that will not load is a row of the index saying so rather than
    an exception: one unreadable folder must not take a library of three
    hundred reports down with it.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    try:
        return ReportRecord.model_validate(blob), ""
    except Exception as exc:                     # a record this build cannot read
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


def _key_of(record: ReportRecord, folder: str, parent_base: str) -> str:
    """This record's library key: the page's, if it wrote one.

    ``report.page.md`` carries the key the ingest computed from the SOURCE
    FILE's own bytes, which is the one the library row already uses and the
    one a re-ingest will produce again. It is preferred over recomputing,
    because recomputing without the PDF falls back to the record's identity
    and would give a second key for the same report.
    """
    page = os.path.join(folder, PAGE_NAME)
    try:
        with open(page, encoding="utf-8") as handle:
            head = handle.read(2000)
    except OSError:
        head = ""
    match = _FRONT_ID.search(head)
    if match:
        return match.group(1)
    return record_key(record, None, parent_base)


def _title_of(record: ReportRecord) -> str:
    from report_ingest.writers import title_of
    return title_of(record)


def _result_values(result: Any) -> Dict[str, Any]:
    """The scalar values one laboratory result printed, for a summary row."""
    if result is None:
        return {}
    try:
        blob = result.model_dump(mode="json")
    except Exception:                            # pragma: no cover - defensive
        return {}
    out: Dict[str, Any] = {}
    for name, value in blob.items():
        if name in ("kind", "prov", "title") or value in (None, "", [], {}):
            continue
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            out[name] = value
        if len(out) >= 10:
            break
    return out


# ---------------------------------------------------------------------------
# the chunks the search runs over
# ---------------------------------------------------------------------------

def _chunks(record: ReportRecord,
            folder: str) -> List[Tuple[str, str, List[int], str]]:
    """``(section, subject, pages, text)`` for everything searchable.

    From the RECORD, because the record is what carries pages: a hit that
    cannot name the page it came off is not a citation. The page and summary
    markdown beside it are indexed too, block by block, so anything those
    print -- a takeaway, a key parameter, a table row -- is findable as well;
    those blocks take their pages from the ``p12`` marks the writers print.
    """
    out: List[Tuple[str, str, List[int], str]] = []
    general, hazards = record.general, record.natural_hazards

    identity = " ".join(str(x) for x in (
        general.projectName, record.project.name, general.projectNumber,
        record.project.number, record.project.client,
        general.geotechnicalEngineerFirm, general.primeContractor,
        general.primeAe, general.postName, general.propertyType,
        general.projectPhase, general.documentType, record.document.report_id,
        hazards.reportDate) if x)
    if identity.strip():
        out.append(("document", "identity", [], identity))

    for name in GENERAL_FIELDS:
        value = getattr(general, name, None)
        if value is None:
            continue
        pages, quote = _cited(general, name)
        out.append(("general", name, pages,
                    f"{name}: {_show(value)}" + (f" -- {quote}" if quote
                                                 else "")))
    for name in NATURAL_HAZARD_FIELDS:
        value = getattr(hazards, name, None)
        if value is None:
            continue
        pages, quote = _cited(hazards, name)
        out.append(("natural_hazards", name, pages,
                    f"{name}: {_show(value)}" + (f" -- {quote}" if quote
                                                 else "")))
    for value in general.bearingCapacityValues:
        out.append(("general", "bearingCapacityValues",
                    _pages_of(value.citation),
                    f"bearing capacity {_show(value.value)} "
                    f"{value.foundation_type} {value.condition}".strip()))
    for stratum in general.strataList:
        out.append(("general", "strataList", _pages_of(stratum.citation),
                    f"stratum {stratum.name} {stratum.description} "
                    f"{stratum.uscs}".strip()))

    for inv in record.investigations:
        words = [inv.investigation_id, inv.kind, _show(inv.total_depth),
                 inv.station, inv.remarks]
        words += [f"{layer.uscs} {layer.description}" for layer in inv.layers]
        words += [f"N {drive.n}" for drive in inv.spt if drive.n is not None]
        words += [f"water at {_show(level.depth)} {level.when}"
                  for level in inv.water]
        out.append(("exploration", inv.investigation_id, list(inv.pages),
                    " ".join(str(x) for x in words if x)))

    for test in record.lab_tests:
        values = _result_values(test.result)
        out.append(("lab", test.kind, list(test.pages),
                    " ".join(str(x) for x in (
                        test.kind, test.investigation_id, test.sample_id,
                        _show(test.depth_top), test.standard, test.lab,
                        " ".join(f"{k} {v}" for k, v in values.items()))
                        if x)))

    for calc in record.calculations:
        out.append(("calculation", calc.kind, list(calc.pages),
                    " ".join(str(x) for x in (
                        calc.kind.replace("_", " "), calc.program,
                        calc.method, calc.subject, calc.summary,
                        " ".join(str(row) for row in calc.results[:12]))
                        if x)))

    for entry in record.qa:
        out.append(("qa", entry.kind, list(entry.pages),
                    f"{entry.kind} {entry.where} {entry.detail} "
                    + " ".join(entry.values[:5])))

    for child in record.bound_documents:
        out.append(("bound", child.bound_id, _page_span(child.pages),
                    f"bound document {child.title} {child.firm} {child.date} "
                    f"{child.kind} {child.document_type} pages {child.pages}"))
    if record.parent is not None:
        out.append(("bound", "parent", _page_span(record.parent.pages),
                    f"bound inside {record.parent.report_id} at pages "
                    f"{record.parent.pages}"))

    out += _markdown_chunks(os.path.join(folder, PAGE_NAME), "page")
    out += _markdown_chunks(os.path.join(folder, SUMMARY_NAME), "summary")
    return [(section, subject, pages, " ".join(text.split())[:2000])
            for section, subject, pages, text in out if text.strip()]


def _page_span(span: str) -> List[int]:
    """``"112-184"`` as the first page of the range, which is where to look."""
    match = re.match(r"\s*(\d+)", str(span or ""))
    return [int(match.group(1))] if match else []


def _markdown_chunks(path: str,
                     section: str) -> List[Tuple[str, str, List[int], str]]:
    """One written page, block by block, with the pages its blocks cite.

    The front matter is skipped (its fields are indexed off the record with
    better pages), each blank-line-separated block is a chunk, and a table is
    one chunk per row so that a hit points at the row rather than the table.
    The pages are the ``p12`` marks the writers print beside an answer.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return []
    body = text
    if body.startswith("---"):
        parts = body.split("\n---", 1)
        body = parts[1] if len(parts) == 2 else body
    out: List[Tuple[str, str, List[int], str]] = []
    heading = ""
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block:
            continue
        first = block.splitlines()[0].strip()
        if first.startswith("#"):
            heading = first.lstrip("# ").strip()
        if block.startswith("|"):
            for line in block.splitlines():
                line = line.strip()
                if not line.startswith("|") or set(line) <= set("|- "):
                    continue
                out.append((section, heading or "table",
                            [int(n) for n in _PAGE_CITE.findall(line)],
                            line.strip("| ").replace("|", " ")))
            continue
        out.append((section, heading or section,
                    [int(n) for n in _PAGE_CITE.findall(block)], block))
    return out


def _fuzzy_value(row: sqlite3.Row) -> str:
    """The VALUE half of one chunk, short enough to fuzzy-match against.

    A narrative chunk is written ``field: value -- quote``; the quote is the
    page's own wording of the same thing and repeating it doubles the length
    for nothing. Anything else is its subject and the head of its text.
    """
    text = " ".join(str(row["text"] or "").split())
    if row["section"] in ("general", "natural_hazards") and ": " in text:
        name, _sep, rest = text.partition(": ")
        return f"{name} {rest.split(' -- ')[0]}"[:160]
    return f"{row['subject']} {text}"[:160]


def _fts_query(text: str) -> str:
    """A question as an FTS5 MATCH expression.

    Every word that narrows anything, quoted so that a hyphen or a full stop
    inside an identifier cannot be read as syntax, joined with OR so that a
    question phrased in a sentence still ranks the chunks that answer it.
    """
    words = [word for word in _WORD.findall(str(text or "").lower())
             if len(word) > 1 and word not in _STOPWORDS]
    if not words:
        words = _WORD.findall(str(text or "").lower())
    return " OR ".join(f'"{word}"' for word in dict.fromkeys(words))
