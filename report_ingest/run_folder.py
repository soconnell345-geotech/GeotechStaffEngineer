"""A folder of reports, read one after another into one library.

The same graph the app's sub-agent runs, driven headless over every PDF in a
folder. This is the report-library use: a few hundred reports go in, one
``reports.db`` comes out, and a library agent searches the pages beside the
owner's other records.

WHAT IT GUARANTEES, because a run of three hundred reports will be interrupted.

**It resumes.** Each report writes into its own folder and a finished report
is skipped on the next run -- as is a finished ITEM inside an unfinished
report, which is the graph's own doing. An interrupted run costs what it had
already spent and nothing more.

**One report cannot stop the run.** A file that will not open, a reader that
raises, a DIGGS file that will not write: the failure is recorded against that
report and the next one starts. The failures are listed at the end and in
``INDEX.md``, because a run that quietly read 297 of 300 is worse than one
that says so.

**The library is shared.** Every report upserts into ONE database, keyed by a
hash of its own file, so re-running the folder updates rows instead of
doubling them.

PRIVACY. The index this writes carries the file STEMS, because on a private
corpus the stem is the report's ID. Nothing here is committed anywhere: the
outputs land where the caller says, and the corpus rule -- IDs only in
anything committed -- is the caller's to keep.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["run_folder", "FolderRun", "ReportRun", "load_di_result"]


@dataclass
class ReportRun:
    """One report's outcome in a folder run."""

    report_id: str
    source: str
    out_dir: str
    ok: bool = True
    error: str = ""
    skipped: bool = False
    #: The library key this report got, and the report it turned out to be a
    #: copy of. Two files with the same bytes are ONE document and share one
    #: library row; that is right, and it is said out loud here so a folder
    #: of 300 files that makes 297 rows is not a mystery.
    key: str = ""
    duplicate_of: str = ""
    n_pages: int = 0
    workflow: str = ""
    investigations: int = 0
    lab_tests: int = 0
    qa: int = 0
    answered: int = 0
    model_calls: int = 0
    dollars: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FolderRun:
    """What a whole folder came to."""

    folder: str
    out_dir: str
    db: str
    reports: List[ReportRun] = field(default_factory=list)
    started: str = ""

    @property
    def failures(self) -> List[ReportRun]:
        return [row for row in self.reports if not row.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folder": self.folder, "out_dir": self.out_dir, "db": self.db,
            "started": self.started,
            "n_reports": len(self.reports),
            "n_failed": len(self.failures),
            "reports": [row.to_dict() for row in self.reports],
        }


def load_di_result(di_dir: Any, stem: str) -> Optional[Any]:
    """The Azure Document Intelligence result for one report, or None.

    Both spellings the corpus has produced are read: ``<stem>.json.gz``, which
    is what this repo's copies are, and the uncompressed ``DI_data_<stem>.json``
    that Funhouse writes. A result that will not read is None -- the report is
    then read from its own text layer, which is a worse answer and not a
    failure.
    """
    if not di_dir:
        return None
    import gzip

    folder = str(di_dir)
    names = [f"{stem}.json.gz", f"{stem}.json",
             f"DI_data_{stem}.json.gz", f"DI_data_{stem}.json"]
    for name in names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as probe:
                gzipped = probe.read(2) == b"\x1f\x8b"
            opener: Any = gzip.open if gzipped else open
            with opener(path, "rt", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, EOFError, json.JSONDecodeError):
            return None
        from planlens.document.azure_di import AzureLayout
        return AzureLayout(result)
    return None


def run_folder(folder: Any, engine_factory: Callable[[], Any], *,
               out_dir: Any,
               pattern: str = ".pdf",
               db_path: Any = None,
               budgets: Any = None,
               questions: Optional[Sequence[str]] = None,
               di_dir: Any = None,
               resume: bool = True,
               max_reports: Optional[int] = None,
               report_ids: Optional[Sequence[str]] = None,
               log: Callable[[str], None] = print) -> FolderRun:
    """Read every PDF in ``folder`` into one library under ``out_dir``.

    ``engine_factory`` is called once per report, so each report's cost is
    metered on its own. ``report_ids`` limits the run to those file stems.
    Everything else is the graph's: ``budgets`` are its ceilings and
    ``questions`` are asked of every report in the folder.
    """
    from report_ingest.graph import ingest_report

    source_dir = str(folder)
    out = str(out_dir)
    os.makedirs(out, exist_ok=True)
    db = str(db_path) if db_path else os.path.join(out, "reports.db")
    run = FolderRun(folder=source_dir, out_dir=out, db=db,
                    started=date.today().isoformat())

    wanted = set(report_ids or ())
    files = sorted(name for name in os.listdir(source_dir)
                   if name.lower().endswith(pattern.lower()))
    stems = [(os.path.splitext(name)[0], os.path.join(source_dir, name))
             for name in files]
    if wanted:
        stems = [(stem, path) for stem, path in stems if stem in wanted]
    if max_reports:
        stems = stems[:int(max_reports)]

    log(f"report ingest over {len(stems)} report(s) in {source_dir}; "
        f"library {db}")
    keys: Dict[str, str] = {}
    for index, (stem, path) in enumerate(stems, start=1):
        report_out = os.path.join(out, stem)
        record_file = os.path.join(report_out, "report.record.json")
        if resume and os.path.isfile(record_file):
            row = _row_from_record(stem, path, report_out, record_file)
            row.skipped = True
            run.reports.append(row)
            log(f"  [{index}/{len(stems)}] {stem}: already done, skipping")
            continue

        started = time.time()
        try:
            record = ingest_report(
                path, engine_factory(), out_dir=report_out,
                budgets=budgets, report_id=stem, resume=resume,
                db_path=db, questions=list(questions or []),
                di_result=load_di_result(di_dir, stem))
        except KeyboardInterrupt:
            log("  interrupted; what is finished is on disk and a later call "
                "resumes")
            break
        except Exception as exc:                 # keep the folder going
            run.reports.append(ReportRun(
                report_id=stem, source=path, out_dir=report_out, ok=False,
                error=f"{type(exc).__name__}: {exc}",
                seconds=round(time.time() - started, 1)))
            log(f"  [{index}/{len(stems)}] {stem}: FAILED -- "
                f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        from report_ingest.writers import record_key
        key = record_key(record, path)
        row = ReportRun(
            key=key, duplicate_of=keys.get(key, ""),
            report_id=stem, source=path, out_dir=report_out,
            n_pages=record.document.n_pages,
            workflow=record.document.workflow,
            investigations=len(record.investigations),
            lab_tests=len(record.lab_tests),
            qa=len(record.qa),
            answered=len(record.general.answered()
                         + record.natural_hazards.answered()),
            model_calls=record.document.model_calls,
            dollars=record.document.dollars,
            seconds=round(time.time() - started, 1))
        keys.setdefault(key, stem)
        run.reports.append(row)
        if row.duplicate_of:
            log(f"  [{index}/{len(stems)}] {stem}: the same file as "
                f"{row.duplicate_of}; they share one library row")
        log(f"  [{index}/{len(stems)}] {stem}: {row.n_pages} pp, "
            f"{row.investigations} exploration(s), {row.lab_tests} lab "
            f"test(s), {row.answered} question(s) answered, "
            f"{row.model_calls} model call(s), {row.seconds:.0f} s")

    _write_index(run, out)
    log(f"\nwrote {os.path.join(out, 'INDEX.md')} and the library at {db}")
    if run.failures:
        log(f"{len(run.failures)} report(s) FAILED: "
            + ", ".join(row.report_id for row in run.failures))
    return run


def _row_from_record(stem: str, path: str, out_dir: str,
                     record_file: str) -> ReportRun:
    """The row for a report that was already read, off its own record."""
    row = ReportRun(report_id=stem, source=path, out_dir=out_dir)
    try:
        with open(record_file, encoding="utf-8") as handle:
            blob = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return row
    document = blob.get("document") or {}
    row.n_pages = int(document.get("n_pages") or 0)
    row.workflow = str(document.get("workflow") or "")
    row.model_calls = int(document.get("model_calls") or 0)
    row.dollars = float(document.get("dollars") or 0.0)
    row.investigations = len(blob.get("investigations") or [])
    row.lab_tests = len(blob.get("lab_tests") or [])
    row.qa = len(blob.get("qa") or [])
    general = blob.get("general") or {}
    hazards = blob.get("natural_hazards") or {}
    row.answered = sum(1 for value in list(general.values())
                       + list(hazards.values())
                       if value not in (None, [], {}, ""))
    return row


def _write_index(run: FolderRun, out: str) -> None:
    """One table of what the folder came to, beside the library."""
    lines = [f"# Report ingest over `{run.folder}`", "",
             f"{len(run.reports)} report(s), {len(run.failures)} failed. "
             f"Library: `{run.db}`. Run {run.started}.", "",
             "| Report | Pages | Workflow | Explorations | Lab tests | "
             "Questions answered | QA | Model calls |",
             "|---|---|---|---|---|---|---|---|"]
    for row in run.reports:
        if not row.ok:
            lines.append(f"| {row.report_id} | | FAILED | | | | | |")
            continue
        lines.append(
            f"| {row.report_id} | {row.n_pages} | {row.workflow} | "
            f"{row.investigations} | {row.lab_tests} | {row.answered} | "
            f"{row.qa} | {row.model_calls} |")
    copies = [row for row in run.reports if row.duplicate_of]
    if copies:
        lines += ["", "## The same document twice", "",
                  "These files have the same bytes as another in the folder, "
                  "so they are one document and share one library row.", ""]
        for row in copies:
            lines.append(f"- `{row.report_id}` is a copy of "
                         f"`{row.duplicate_of}`")
    if run.failures:
        lines += ["", "## What failed", ""]
        for row in run.failures:
            lines.append(f"- `{row.report_id}`: {row.error}")
    with open(os.path.join(out, "INDEX.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as handle:
        json.dump(run.to_dict(), handle, indent=2)
